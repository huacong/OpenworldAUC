import os
import time
import yaml
import clip
import shutil
import pickle
import random
import argparse
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import json
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from skimage.filters import threshold_otsu
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils.util_algo import metrics_old, metrics_new
from datasets import WrapperDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, average_precision_score
import csv

__all__ = ["GMoP"]

_tokenizer = _Tokenizer()

import torch
import torch.nn as nn
from abc import abstractmethod


class AUCLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 gamma=1):
        super(AUCLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
    
    def _check_input(self, pred, target):
        assert pred.max() <= 1 and pred.min() >= 0
        assert target.min() >= 0
        assert pred.shape[0] == target.shape[0]

    def forward(self, pred, target, pos_weight, neg_weight):
        self._check_input(pred, target)
        Y = target.float()
        pos_score = pred[Y == 1]
        neg_score = pred[Y == 0]
        pair_diff = 1 - (pos_score.unsqueeze(1) - neg_score.unsqueeze(0))
        weights = pos_weight.unsqueeze(1) * neg_weight.unsqueeze(0)
        loss = weights * (pair_diff**2)
        return loss.mean()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).float()

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)].type(self.dtype) @ self.text_projection

        return x

#### Out-of-Distribution Prompter ####
class OODPrompter(nn.Module):
    def __init__(self, cfg, log, device, clip_model):
        super().__init__()
        
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        self.n_cls = 0
        ctx_init = None
        # ctx_init = 'a photo of a' # caltech101
        n_ctx = int(cfg["method"]["ood_n_ctx"])
        self.n_ctx = n_ctx
        self.device = device

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init       
            self.n_ctx = n_ctx
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype).to(device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        log.info(f'Initial context: "{prompt_prefix}"')
        log.info(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        ######
        self.prompt_prefix = prompt_prefix

        
    def get_prefix_suffix_token(self, classnames):
        clip_model = self.clip_model
        self.n_cls = len(classnames)
        prompt_prefix = self.prompt_prefix
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])  # CLS, EOS
        
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts

#### CoOp Prompter ####
class CoOpPrompter(nn.Module):
    def __init__(self, cfg, log, device, clip_model):
        super().__init__()
        
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        self.n_cls = 0
        ctx_init = None
        # ctx_init = 'a photo of a' # caltech101
        n_ctx = int(cfg["method"]["n_ctx"])
        self.n_ctx = n_ctx
        self.device = device

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init       
            self.n_ctx = n_ctx
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype).to(device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        log.info(f'Initial context: "{prompt_prefix}"')
        log.info(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        self.prompt_prefix = prompt_prefix
        
    def get_prefix_suffix_token(self, classnames):
        clip_model = self.clip_model
        self.n_cls = len(classnames)
        prompt_prefix = self.prompt_prefix
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])  # CLS, EOS
        
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts
    
class ClipPrompter(nn.Module):
    def __init__(self, cfg, log, device, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.device = device
        self.classtokens = None
        self.n_classes = 0
        self.clip_model.eval()
        if cfg['dataset']['name'] == "fgvc":
            ctx_init = [cfg["dataset"]["ctx_init"]]
        else:
            ctx_init = [cfg["dataset"]["ctx_init"]] #cfg["method"]["ctx_init"].split(" ")

        self.prefixs  = []
        for prefix in ctx_init:
            prefix = prefix.replace("_", " ")
            self.prefixs += [prefix]
        for prefix in self.prefixs:
            log.info(f'Initial context: "{prefix}"')
       
        
    def get_prefix_suffix_token(self, classnames):
        clip_model = self.clip_model
        prefix_classnames = []
        #### Naive version: a photo of {}
        for prefix in self.prefixs:
            prefix_classnames += [prefix.format(name.replace("_", " ")) for name in classnames]
        classtokens = clip.tokenize(prefix_classnames).to(self.device)
        self.classtokens = classtokens.detach()
        self.n_classes = len(classnames) 


    def forward(self, images):
        images = images.type(self.dtype).to(self.device)
        logits_per_image, logits_per_text = self.clip_model(images, self.classtokens)
        logits_per_image = logits_per_image.view(images.shape[0], -1, self.n_classes)
        logits_per_image = logits_per_image.mean(1)
        return logits_per_image ###

def count_parameters(model):
    total = 0
    name_forbidden = 'clip_model'
    for name, param in model.named_parameters():
        if name_forbidden not in name and param.requires_grad:
            total += param.numel()
    return total


class GMoP(object):
    def __init__(self, args, cfg, logger, device, clip_model, dataset):
        # Logger
        self.model_path = cfg["log"]["model"]
        self.predict_path = cfg["log"]["prediction"]
        self.args = args
        self.lam = self.args.lam
        self.alpha = self.args.alpha
        self.K = self.args.K
        self.seed = args.seed
        self.imb_domain = args.imb_domain
        
        # Computing Device
        self.device = device 

        # CLIP Model
        self.clip_model = clip_model 
        self.text_encoder = TextEncoder(self.clip_model)
        self.image_encoder = self.clip_model.visual
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

        # Prompter Learner
        self.clip_prompter = ClipPrompter(cfg, logger, device, clip_model).to(device)
        
        self.coop_prompter = CoOpPrompter(cfg, logger, device, clip_model).to(device)
        
        self.ood_prompter = nn.ModuleList([
            OODPrompter(cfg, logger, device, clip_model)
            for i in range(self.K)
        ]).to(device)

        

        self.ood_optimizer = torch.optim.SGD(self.ood_prompter.parameters(), lr=cfg["method"]['lr'] )
        self.ood_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.ood_optimizer,  cfg["method"]['train_epoch'])
        self.iid_optimizer = torch.optim.SGD(self.coop_prompter.parameters(), lr=cfg["method"]['lr'] )
        self.iid_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.iid_optimizer,  cfg["method"]['train_epoch'])
        self.record = []
        
        
        ### auc_criterion
        self.criterion = AUCLoss(
            num_classes=2, # number of classes
            gamma=1.0 # safe margin
        ) # create loss criterion

        # Init OOD Detector
        self.n_classes = len(dataset.classnames)
        self.classnames = dataset.classnames
        self.n_divs = (self.n_classes * (self.K - 1) + self.K - 1) // self.K
        self.tokenized_prompts, self.ood_masks, self.ood_labels = [], [], []
        self.ood_classes = []
        self.ood_recoveries = []
        self.thresholds = []
        self.ood_classnames = []

        # Process mask, label and classname for OOD detector  ## 34 50 self.K = 3
        logger.info("OOD Detector: {} classes in {} classes".format(self.n_divs, self.n_classes))
        for i in range(self.K):
            classname, ood_mask, ood_label= [], [], []
            ood_recovery = []
            for j in range(self.n_classes):
                if j % self.K != i:
                    ood_label.append(len(classname))
                    ood_mask.append(1)
                    ood_recovery.append(j)
                    classname.append(dataset.classnames[j])
                else:
                    ood_label.append(-1)
                    ood_mask.append(0)
            if len(classname) < self.n_divs:
                ood_label.append(len(classname))
                ood_mask.append(1)
                ood_recovery.append(i)
                classname.append(dataset.classnames[i])
            logger.info("OOD Detector #{}: {} classes in {} classes".format(i, len(classname), self.n_classes))
            self.ood_classes.append(set(classname))
            self.ood_classnames.append(classname)
            self.ood_masks.append(torch.tensor(ood_mask).long().to(self.device))
            self.ood_labels.append(torch.tensor(ood_label).long().to(self.device))
            self.ood_recoveries.append(torch.tensor(ood_recovery).long().to(self.device))
            self.ood_prompter[i].get_prefix_suffix_token(classname)
            self.tokenized_prompts.append(self.ood_prompter[i].tokenized_prompts)
            
        self.thresholds = []
        self.threshold  = 0
        
        # Setup OOD margin
        if cfg["method"]["ood_margin"] == "auto":
            energy_func = lambda v: -np.sum(np.array(v) * np.log(np.array(v)))
            self.ood_margin = energy_func([1 / self.n_divs] * self.n_divs) - energy_func([0.4 / (self.n_divs - 1)] * (self.n_divs - 1) + [0.6])
        else:
            self.ood_margin = float(cfg["method"]["ood_margin"])
        logger.info("OOD Margin: {} => {}".format(cfg["method"]["ood_margin"], self.ood_margin))
        
        # Evaluate Information
        self.evaluate_OOD_cache = False
        self.OOD_scores = None
        self.CLIP_logits = None
    
    def load_model(self,save_dir,epoch):
        coop_ctx = torch.load(os.path.join(save_dir, f"coop_prompter_epoch{epoch}.pth"))
        self.coop_prompter.ctx.data.copy_(coop_ctx)

        ood_ctx_list = torch.load(os.path.join(save_dir, f"ood_prompter_epoch{epoch}.pth"))
        for p, ctx in zip(self.ood_prompter, ood_ctx_list):
            p.ctx.data.copy_(ctx)

        print(f"Prompters loaded from {save_dir}")

    def save_model(self,save_dir,epoch):
        try:
            torch.save(self.coop_prompter.ctx.detach().cpu(), os.path.join(save_dir, f"coop_prompter_epoch{epoch}.pth"))
            torch.save([p.ctx.detach().cpu() for p in self.ood_prompter], os.path.join(save_dir,f"ood_prompter_epoch{epoch}.pth"))
            print(f"Prompters for epoch {epoch} saved to {save_dir}")
        except Exception as e:
            print(f"Failed to save models for epoch {epoch}: {e}")

    def delete_previous_models(self,save_dir, epoch):
        
        for file in os.listdir(save_dir):
            if f"epoch{epoch}" not in file:
                os.remove(os.path.join(save_dir, file))
                print(f"Deleted old model: {file}")
                
    def setup_ood_prompter(self, loader):
        classnames = loader #loader.dataset.classnames
        self.tokenized_prompts = []
        for i in range(self.K):
            classname = [c for c in self.ood_classnames[i]]
            for c in classnames:
                if c not in self.ood_classnames[i]:
                    classname.append(c)
            self.ood_prompter[i].get_prefix_suffix_token(classname)
            self.tokenized_prompts.append(self.ood_prompter[i].tokenized_prompts)
            
    
    def recover_ood_prompter(self):
        self.tokenized_prompts = []
        for i in range(self.K):
            self.ood_prompter[i].get_prefix_suffix_token(self.ood_classnames[i])
            self.tokenized_prompts.append(self.ood_prompter[i].tokenized_prompts)
    
    def setup_coop_prompter(self, loader):        
        self.coop_prompter.get_prefix_suffix_token(loader.dataset.classnames)
        coop_tokenized_prompts = self.coop_prompter.tokenized_prompts
        return coop_tokenized_prompts

    def freeze_encoder(self, logger):
        logger.info("Turning off gradients in both the image and the text encoder")

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def train(self, cfg, logger, base_train_loader, base_test_loader, new_test_loader, test_loader):
        self.freeze_encoder(logger)
        method_cfg = cfg["method"]
        train_loader = base_train_loader
        self.evaluate_OOD_cache = False
        best_opt_auc = 0.0
        save_epoch = 0
        for epoch in range(self.args.epoch): #method_cfg["train_epoch"]
            last_time = time.time()
            loss_id, loss_auc = self.train_epoch(epoch, cfg, logger, train_loader)
            if epoch % self.args.print_freq == 0: 
                message =  "Epoch:[{:3d}/{:3d}]({:.2f}s) IDLoss:{:.2f} AUCLoss:{:.2f} EnergyLoss:{:.2f}"
                logger.info(message.format(epoch, self.args.epoch, time.time() - last_time, loss_id, loss_auc,))
            if epoch + 1 == self.args.epoch:
                self.save_model(save_dir=cfg["log"]["model"],epoch=epoch)
                
        self.predict(cfg, logger, base_test_loader=base_test_loader, new_test_loader=new_test_loader, test_loader=test_loader)
        self.evaluate_OPT_metric(cfg, logger)
        return save_epoch
        
    def eval(self,cfg,logger,base_test_loader,new_test_loader,test_loader,epoch,cross_name=None):
        self.load_model(save_dir=cfg["log"]["model"],epoch=epoch)
        self.predict(cfg, logger, base_test_loader=base_test_loader, new_test_loader=new_test_loader, test_loader=test_loader) 
        base_acc, new_acc, auroc, opt_auc = self.eval_time_evaluate_OPT_metric(cfg,logger)
        results = {
                'K': self.args.K,
                'epoch': epoch,
                'lambda': self.args.lam,
                'alpha': self.args.alpha,
                'warmup': self.args.warm_up,
                'base_acc': round(base_acc*100, 2),
                'new_acc': round(new_acc*100, 2),
                'auroc': round(auroc*100,2),
                'openworldauc': round(opt_auc*100,2)
            }
        file_name = f"Ours_{cfg['dataset']['name']}_{self.args.mask}.csv"
        headers = ['K', 'epoch', 'lambda', 'alpha', 'warmup', 'base_acc', 'new_acc', 'auroc', 'openworldauc']  # 添加了缺失的逗号
        file_name = os.path.join(f"./output_search/output_{cfg['dataset']['name']}_csv", file_name)  # 修正了嵌套引号
        csv_directory = os.path.dirname(file_name)
        if not os.path.exists(csv_directory): 
            os.makedirs(csv_directory)
        file_exists = os.path.exists(file_name)
        with open(file_name, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)
    
        print(f"RESULTS has been appended to '{file_name}'.")
        
    def eval_test(self,cfg,logger,base_test_loader,new_test_loader,test_loader,epoch,cross_name=None):
        self.load_model(save_dir=cfg["log"]["model"],epoch=epoch)
        self.predict(cfg, logger, base_test_loader=base_test_loader, new_test_loader=new_test_loader, test_loader=test_loader) 
        base_acc, new_acc, auroc, opt_auc = self.eval_time_evaluate_OPT_metric(cfg,logger)
        results = {
                'SEED': self.seed,
                'base_acc': round(base_acc*100, 2),
                'new_acc': round(new_acc*100, 2),
                'auroc': round(auroc*100,2),
                'openworldauc': round(opt_auc*100,2),
                'epoch': epoch
            }
        file_name = f"Ours_{cfg['dataset']['name']}_{self.args.mask}_{cross_name}.csv"
        headers = ['SEED', 'base_acc', 'new_acc', 'auroc', 'openworldauc', 'epoch']
        file_name = os.path.join(f'./output_0_{self.args.mask}_csv2', file_name)
        file_exists = os.path.exists(file_name)
        with open(file_name, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)

        print(f"Seed {results['SEED']} data has been appended to '{file_name}'.")

    def train_epoch(self, epoch, cfg, logger, train_loader):
        
        self.ood_prompter.train()
        self.coop_prompter.train()
        self.clip_prompter.eval()

        ####
        self.recover_ood_prompter()
        coop_tokenized_prompts  = self.setup_coop_prompter(train_loader)
        with torch.no_grad():
            self.clip_prompter.get_prefix_suffix_token(train_loader.dataset.classnames)
            self.clip_prompter.eval()

        avg_id_loss, avg_auc_loss, avg_warm_loss = [], [], []

        for idx, (images, target, _ ) in enumerate(train_loader):
            with torch.no_grad():
                # Get Image Features
                images, target = images.type(self.dtype).to(self.device), target.to(self.device)
                image_features = self.image_encoder(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            ######## ood_setting
            prompts = [self.ood_prompter[i]() for i in range(self.K)]
            text_features = [self.text_encoder(prompts[i], self.tokenized_prompts[i]) for i in range(self.K)]
            text_features = [text_features[i] / text_features[i].norm(dim=-1, keepdim=True) for i in range(self.K)]
            logit_scale = self.logit_scale.exp().detach()
            logits = [logit_scale * image_features.float() @ text_features[i].T.float() for i in range(self.K)]
            

            ######## coop_logits
            coop_prompts = self.coop_prompter() 
            coop_text_features = self.text_encoder(coop_prompts, coop_tokenized_prompts)
            coop_text_features = coop_text_features / coop_text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp().detach()
            coop_logits = logit_scale * image_features.float() @ coop_text_features.T.float()
            
            
            ####### clip_logits
            with torch.no_grad():
                clip_logits = self.clip_prompter(images)
            
            loss_ood_auc, loss_ood_warm =  0, 0
            ce_loss = F.cross_entropy(coop_logits.half(),target.long()).half()

            for i in range(self.K):
                batch_label = torch.index_select(self.ood_labels[i],0,target.long()).long() #batch_label [32]
                iid_sample = batch_label >= 0 #iid_sample [32] True or False
                ood_sample = batch_label < 0 #ood_sample [32] True or False
                if torch.sum(iid_sample) == 0 or torch.sum(ood_sample) == 0:
                    continue
                
                coop_score = torch.sigmoid(F.softmax(coop_logits,dim=1)[torch.arange(coop_logits.size(0)), target])
                pos_weight = coop_score[iid_sample]
                clip_score = torch.sigmoid(F.softmax(clip_logits,dim=1)[torch.arange(clip_logits.size(0)), target]) # [32,C]
                neg_weight = clip_score[ood_sample]



                #####################计算auc_loss
                soft_logits = F.softmax(logits[i], dim=1)

                pred_pos, _ = soft_logits[batch_label>=0,:].max(axis = 1) ###
                pred_neg, _ = soft_logits[batch_label<0,:].max(axis = 1) ###
                
                pred = torch.cat((pred_pos, pred_neg), dim=0)

                auc_target_pos = torch.ones(pred_pos.size(0), dtype=torch.long).to(self.device)
                auc_target_neg = torch.zeros(pred_neg.size(0), dtype=torch.long).to(self.device)
                auc_target = torch.cat((auc_target_pos, auc_target_neg), dim=0)
                
                loss_ood_auc += self.criterion(pred,auc_target,pos_weight,neg_weight) 
                loss_ood_warm += F.cross_entropy(logits[i],batch_label,ignore_index=-1)
                             
            loss_id = ce_loss
            loss_ood_auc, loss_ood_warm =  loss_ood_auc / self.K, loss_ood_warm / self.K
           
            loss = loss_id + self.alpha*loss_ood_warm + self.lam*loss_ood_auc

            self.ood_optimizer.zero_grad()
            self.iid_optimizer.zero_grad()
            loss.backward()
            self.ood_optimizer.step()
            self.iid_optimizer.step()
            
            if idx + 1 == len(train_loader):
                self.iid_scheduler.step()
                self.ood_scheduler.step()

            avg_id_loss.append(loss_id.item())
            avg_auc_loss.append(loss_ood_auc.item())
            avg_warm_loss.append(loss_ood_warm.item())

        return np.mean(avg_id_loss), np.mean(avg_auc_loss)

    
    def save_prediction(self, predicts, name="last"):
        pred_path = os.path.join(self.predict_path, f"{name}.pkl")
        last_pred_path = os.path.join(self.predict_path, "last.pkl")
        pickle.dump(predicts, file=open(pred_path, 'wb+'))
        shutil.copyfile(pred_path, last_pred_path)
    
    def predict(self, cfg, logger, base_test_loader, new_test_loader, test_loader):
        self.ood_prompter.eval()
        self.coop_prompter.eval()
        self.clip_prompter.eval()
        
        total_inference_time = 0.0
        self.setup_ood_prompter(test_loader)
        with torch.no_grad():
            start_time = time.time()
            ood_prompts = [self.ood_prompter[i]() for i in range(self.K)]
            ood_text_features = [self.text_encoder(ood_prompts[i], self.tokenized_prompts[i]).to(self.device) for i in range(self.K)]
            ood_text_features = [ood_text_features[i] / ood_text_features[i].norm(dim=-1, keepdim=True) for i in range(self.K)]
            end_time = time.time()
            total_inference_time += end_time - start_time
            
        with torch.no_grad():
            start_time = time.time()
            coop_tokenized_prompts  = self.setup_coop_prompter(base_test_loader)
            coop_prompts = self.coop_prompter()
            coop_text_features = self.text_encoder(coop_prompts, coop_tokenized_prompts)
            coop_text_features = coop_text_features / coop_text_features.norm(dim=-1, keepdim=True)
            end_time = time.time()
            total_inference_time += end_time - start_time
        with torch.no_grad():
            self.clip_prompter.get_prefix_suffix_token(new_test_loader.dataset.classnames)
            self.clip_prompter.eval()


        self.opt_ood_scores = {0:[], 1:[]}
        self.opt_ood_labels = {0:[], 1:[]}

        self.opt_close_set_preds = {0:[],1:[]}
        self.opt_close_set_labels = {0:[],1:[]}

        with torch.no_grad():
            for new_class_label, loader in enumerate((base_test_loader,new_test_loader)):
                ## new_class_label = 0, base class set;
                ## new_class_label = 1, new class set
                scores = []
                preds = []
                targets = []
                if new_class_label:
                    logger.info("Forward pass through the New Class set...")
                else:
                    logger.info("Forward pass through the Base Class set....")
                
                for ids, (images, labels, cnames) in enumerate(loader):
                    images = images.type(self.dtype).to(self.device)
                    

                    image_features = self.image_encoder(images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True) # image_features [256,512]
                    logit_scale = self.logit_scale.exp() 
                    if new_class_label:
                        logits = self.clip_prompter(images) 
                    else:
                        logits = logit_scale * image_features.float() @ coop_text_features.T.float()
                    
                    
                    ood_logits  = [ logit_scale * image_features.float() @ ood_text_features[i].T.float() for i in range(self.K) ]
                    

                    
                    ood_probas = torch.cat([
                        F.softmax(ood_logits[i], dim=1)[:, :self.n_divs].unsqueeze(-1)
                        for i in range(self.K)
                    ], -1) ## [256,34,3]
                    
                    scores.append(ood_probas.cpu()) #[256,34,3]
                    preds.append(logits.cpu()) ## [256,50]
                    targets.append(labels.cpu()) #[256,]

                scores = torch.cat(scores, 0) ##### scores.shape [1666,34,3]
                scores = torch.nan_to_num(scores, nan=0.0)
                scores = torch.clamp(scores, min=0, max=1)
                
                # Compute OOD Auroc
                probs  = torch.max(scores.view(scores.shape[0], -1), 1)[0].cpu().numpy().tolist()
                self.opt_ood_scores[new_class_label].extend(probs)
                self.opt_ood_labels[new_class_label].extend([1-new_class_label]*len(probs))
                ### new_class_label = 0, base class set则为1, 表明要给予更高的分

                preds = torch.cat(preds,0) #####[1666,50]
                targets = torch.cat(targets,0) ####[1666]
                self.opt_close_set_preds[new_class_label].extend(preds.cpu().numpy().tolist())
                self.opt_close_set_labels[new_class_label].extend(targets.cpu().numpy().tolist())
                



    def evaluate_OPT_metric(self,cfg,logger):
        opt_ood_score = np.array(self.opt_ood_scores[0] + self.opt_ood_scores[1])
        opt_ood_labels = np.array(self.opt_ood_labels[0] + self.opt_ood_labels[1])
        auroc = compute_auroc(opt_ood_score, opt_ood_labels)
        logger.info('AUROC: {:.5f}'.format(auroc))

        base_set_preds = np.array(self.opt_close_set_preds[0])
        base_set_labels = np.array(self.opt_close_set_labels[0])
        base_acc = closed_set_acc(base_set_preds, base_set_labels)
        logger.info('Base Acc: {:.5f}'.format(base_acc))

        new_set_preds = np.array(self.opt_close_set_preds[1])
        new_set_labels = np.array(self.opt_close_set_labels[1])
        new_acc = closed_set_acc(new_set_preds, new_set_labels)
        logger.info('New Acc: {:.5f}'.format(new_acc))

        opt_ood_score_base_set = opt_ood_score[opt_ood_labels.astype('bool')] ### ood_label为1
        opt_ood_score_new_set = opt_ood_score[~opt_ood_labels.astype('bool')]
        base_set_preds_cls = base_set_preds.argmax(axis=-1)
        new_set_preds_cls = new_set_preds.argmax(axis=-1)

        opt_auc = compute_opt_auc(opt_ood_score_base_set, opt_ood_score_new_set, base_set_preds_cls, base_set_labels, new_set_preds_cls, new_set_labels)
        logger.info('opt_auc: {:.5f}'.format(opt_auc))
        return opt_auc
    
    def eval_time_evaluate_OPT_metric(self,cfg,logger):
        opt_ood_score = np.array(self.opt_ood_scores[0] + self.opt_ood_scores[1])
        opt_ood_labels = np.array(self.opt_ood_labels[0] + self.opt_ood_labels[1])
        auroc = compute_auroc(opt_ood_score, opt_ood_labels)
        logger.info('AUROC: {:.5f}'.format(auroc))

        base_set_preds = np.array(self.opt_close_set_preds[0])
        base_set_labels = np.array(self.opt_close_set_labels[0])
        base_acc = closed_set_acc(base_set_preds, base_set_labels)
        logger.info('Base Acc: {:.5f}'.format(base_acc))

        new_set_preds = np.array(self.opt_close_set_preds[1])
        new_set_labels = np.array(self.opt_close_set_labels[1])
        new_acc = closed_set_acc(new_set_preds, new_set_labels)
        logger.info('New Acc: {:.5f}'.format(new_acc))

        opt_ood_score_base_set = opt_ood_score[opt_ood_labels.astype('bool')] ### ood_label为1
        opt_ood_score_new_set = opt_ood_score[~opt_ood_labels.astype('bool')]
        base_set_preds_cls = base_set_preds.argmax(axis=-1)
        new_set_preds_cls = new_set_preds.argmax(axis=-1)

        opt_auc = eval_time_compute_opt_auc(opt_ood_score_base_set, opt_ood_score_new_set, base_set_preds_cls, base_set_labels, new_set_preds_cls, new_set_labels)
        logger.info('opt_auc: {:.5f}'.format(opt_auc))
        return base_acc, new_acc, auroc, opt_auc




def closed_set_acc(preds, labels):
    preds = preds.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    return acc
    
def compute_auroc(open_set_preds, open_set_labels):
    auroc = roc_auc_score(open_set_labels, open_set_preds)
    return auroc

def compute_opt_auc(base_score, new_score, base_preds, base_labels, new_preds, new_labels):
    base_score, new_score = base_score.tolist(), new_score.tolist()
    #### 如果base不对, 就加大new
    min_new = min(new_score) - 1e-5
    max_base = max(base_score) + 1e-5
    base_correct = (base_preds == base_labels).tolist()
    new_correct = (new_preds == new_labels).tolist()
    ### y_score = 调整base score + 调整new_score
    base_tilde_score = [value if hit else min_new for value, hit in zip(base_score, base_correct)]
    new_tilde_score = [value if hit else max_base for value, hit in zip(new_score, new_correct)]
    y_score =  base_tilde_score + new_tilde_score
    y_true = [1] * len(base_tilde_score) + [0]*len(new_tilde_score)
    
    opt_auc = roc_auc_score(y_true, y_score)
    return opt_auc

def eval_time_compute_opt_auc(base_score, new_score, base_preds, base_labels, new_preds, new_labels):
    base_score, new_score = base_score.tolist(), new_score.tolist()
    min_new = min(new_score) - 1e-5
    max_base = max(base_score) + 1e-5
    base_correct = (base_preds == base_labels).tolist()
    new_correct = (new_preds == new_labels).tolist()
    base_tilde_score = [value if hit else min_new for value, hit in zip(base_score, base_correct)]
    new_tilde_score = [value if hit else max_base for value, hit in zip(new_score, new_correct)]
    y_score =  base_tilde_score + new_tilde_score
    y_true = [1] * len(base_tilde_score) + [0]*len(new_tilde_score)

    opt_auc = roc_auc_score(y_true, y_score)
    return opt_auc


def compute_openauc(x1, x2, pred, labels):
    """
    :param x1: open set score for each known class sample (B_k,)
    :param x2: open set score for each unknown class sample (B_u,)
    :param pred: predicted class for each known class sample (B_k,)
    :param labels: correct class for each known class sample (B_k,)
    :return: Open Set Classification Rate
    """
    x1, x2, correct = x1.tolist(), x2.tolist(), (pred == labels).tolist()
    m_x2 = max(x2) + 1e-5
    y_score = [value if hit else m_x2 for value, hit in zip(x1, correct)] + x2
    y_true = [0] * len(x1) + [1] * len(x2)
    open_auc = roc_auc_score(y_true, y_score)
    return open_auc