import os
import random
import argparse
import yaml
import logging
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from utils.util_algo import *
from utils.util_data import *
from models import GMoP
from datasets import build_dataset

_tokenizer = _Tokenizer()

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  default="/data/huacong/OpenworldAUC/configs/GMoP/Vit-B16.yaml", dest='config',  help='settings of methods in yaml format')
    parser.add_argument('--dataset', default="/data/huacong/OpenworldAUC/configs/datasets/food101.yaml", dest='dataset', help='settings of dataset in yaml format')
    parser.add_argument('--seed',    type=int, default=1, metavar='N', help='fix random seed')
    parser.add_argument('--lam',    type=float, default=1.0, metavar='N', help='fix random seed')
    parser.add_argument('--alpha',    type=float, default=0.5, metavar='N', help='alpha')
    parser.add_argument('--logdir',  type=str, default="./results")
    parser.add_argument('--epoch',    type=int, default=100, metavar='N')
    parser.add_argument('--K',    type=int, default=3, metavar='N')
    parser.add_argument('--eval-epoch', type=int, default=10, metavar='N')
    parser.add_argument('--print-freq', type=int, default=1, metavar='N')
    parser.add_argument('--warm-up', type=int, default=0,metavar='N', help='warm-up epochs for classifier')
    parser.add_argument('--mask',    type=str, default='all_mask', metavar='N', help='if mask') ### all_mask 和 no_mask
    parser.add_argument('--imb-domain',  type=str, default='base', metavar='N', help='base or new')
    parser.add_argument('--save',    type=str, default='save', metavar='N', help='if mask')
    args = parser.parse_args()
    return args    

def main(mode):
    # Set Cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = get_arguments()
    set_random_seed(args.seed)
    assert(os.path.exists(args.config))
    assert(os.path.exists(args.dataset))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg.update( yaml.load(open(args.dataset, 'r'), Loader=yaml.Loader) )
    logging.basicConfig(level=logging.INFO)
    log_file_path = args.logdir + f"/{cfg['method']['name']}/{cfg['dataset']['name']}_{args.mask}_{args.lam}_{args.alpha}_{args.warm_up}/{args.seed}/K_{args.K}/log_opt.txt"
    log_directory = os.path.dirname(log_file_path)
    cfg["log"] = {
        "root": log_directory, 
        "model": os.path.join(log_directory, "model"), 
        "prediction": os.path.join(log_directory, "prediction")
    }
    if not os.path.exists(log_directory): 
        os.makedirs(log_directory)
        os.makedirs(cfg["log"]["model"])
        os.makedirs(cfg["log"]["prediction"])
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.info(args)
    logger.info(cfg)

    # Load clip
    clip_model, transform = clip.load(cfg['method']["backbone"])
    clip_model = clip_model.to(device)

    # Prepare dataset
    logger.info('Preparing dataset')
    ### train
    base_dataset = build_dataset(dataset=cfg['dataset']['name'], root=cfg['dataset']['root'], subsample='base', num_shots=cfg['dataset']['shots'], transform=transform, type='train', seed=args.seed)
    

    ### test
    base_test_dataset = build_dataset(dataset=cfg['dataset']['name'], root=cfg['dataset']['root'], subsample='base', num_shots=cfg['dataset']['shots'], transform=transform, type='test', seed=args.seed, imb_domain = args.imb_domain)
    #base_classnames = base_test_dataset.classnames
    new_test_dataset  = build_dataset(dataset=cfg['dataset']['name'], root=cfg['dataset']['root'], subsample='new',  num_shots=cfg['dataset']['shots'], transform=transform, type='test', seed=args.seed, imb_domain = args.imb_domain)
    #new_classnames = new_test_dataset.classnames
    test_dataset =  build_dataset(dataset=cfg['dataset']['name'], root=cfg['dataset']['root'], subsample='all',  num_shots=-1, transform=transform, type='test', seed=args.seed, imb_domain = args.imb_domain)
    

    ### dataloader
    base_loader = DataLoader(dataset=base_dataset, batch_size=cfg['method']['train_batch_size'], shuffle=True,  num_workers=16)
    
    base_test_loader = DataLoader(dataset=base_test_dataset, batch_size=cfg['method']['eval_batch_size'],  shuffle=False, num_workers=16)
    new_test_loader = DataLoader(dataset=new_test_dataset, batch_size=cfg['method']['eval_batch_size'],  shuffle=False, num_workers=16)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg['method']['eval_batch_size'],  shuffle=False, num_workers=16)
    
    logger.info("Training set: {}  Base Testing: {} New Testing: {} Testing set: {}".format(len(base_dataset), len(base_test_dataset), len(new_test_dataset), len(test_dataset)))
    
    model = None
    if cfg["method"]["name"].startswith("GMoP"): 
        model = GMoP(args=args, cfg=cfg, logger=logger, device=device, clip_model=clip_model, dataset=base_dataset)
        
    if mode == 'train':
        model.train(cfg=cfg, logger=logger, base_train_loader=base_loader, base_test_loader=base_test_loader, new_test_loader=new_test_loader, test_loader=test_loader) # test dataset == 'all'
    elif mode == 'test':
        model.eval(cfg=cfg, logger=logger,base_test_loader=base_test_loader, new_test_loader=new_test_loader, test_loader=test_loader, epoch=args.eval_epoch)

if __name__ == '__main__':
    main(mode='train')