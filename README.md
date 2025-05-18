# OpenworldAUC

This is the official code for the paper “OpenworldAUC: Towards Unified Evaluation and Optimization for Open-world Prompt Tuning.” accepted by International Conference on Machine Learning (ICML2025). This paper is available at [here](https://arxiv.org/pdf/2505.05180).

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2505.05180) [![video](https://img.shields.io/badge/Video-Presentation-F9D371)](https://github.com/huacong/ReconBoost) [![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://github.com/huacong/ReconBoost) [![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://github.com/huacong/OpenworldAUC)

**Paper Title: OpenworldAUC: Towards Unified Evaluation and Optimization for Open-world Prompt Tuning.**   

**Authors: [Cong Hua](https://huacong.github.io/),  [Qianqian Xu*](https://qianqianxu010.github.io/), [Zhiyong Yang](https://joshuaas.github.io/), [Zitai Wang](https://wang22ti.com/), [Shilong Bao](https://statusrank.github.io/),  [Qingming Huang*](https://people.ucas.ac.cn/~qmhuang)**   

**Prompt tuning** adapts Vision-Language Models like **CLIP** to open-world tasks with minimal training costs. In this direction, one typical paradigm evaluates model performance separately on known classes (*i.e.*, base domain) and unseen classes (*i.e.*, new domain). However, real-world scenarios require models to handle inputs without prior domain knowledge. This practical challenge has spurred the development of **Open-world Prompt Tuning**, which demands a unified evaluation of two stages: 1) detecting whether an input belongs to the base or new domain (P1), and 2) classifying the sample into its correct class (P2). What's more, as domain distributions are generally unknown, a proper metric should be insensitive to varying base/new sample ratios (P3). However, we find that current metrics, including HM, overall accuracy, and AUROC, fail to satisfy these three properties simultaneously. To bridge this gap, we propose OpenworldAUC, a unified metric that jointly assesses detection and classification through pairwise instance comparisons. To optimize OpenworldAUC effectively, we introduce **Gated Mixture-of-Prompts (GMoP)**, which employs domain-specific prompts and a gating mechanism to dynamically balance detection and classification. Theoretical guarantees ensure generalization of GMoP under practical conditions. Experiments on 15 benchmarks in open-world scenarios show GMoP achieves SOTA performance on OpenworldAUC and other metrics.

![pipeline](docs\pipeline.jpg)

## 🚀 Quick Start

#### 1. Prepare datasets

Prepare the datasets according to the instructions in [DATASETS](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) and put them in the `DATA` directory.

#### 2. Prepare Python Environment

Clone GMoP repository, create conda environment, and then install the required packages.

```bash
git clone https://github.com/huacong/OpenworldAUC.git
cd OpenworldAUC
conda create -n GMoP python==3.8
conda activate GMoP
pip install -r requirements.txt
```

#### 3. Run GMoP Approach

```bash
python main.py --seed 1 --lam 1.0 --alpha 1.0 --epoch 10 --dataset configs/datasets/imagenet.yaml
python main.py --seed 1 --lam 1.0 --alpha 0.5 --epoch 40 --dataset configs/datasets/caltech101.yaml
python main.py --seed 1 --lam 1.0 --alpha 0.5 --epoch 40 --dataset configs/datasets/oxford_pets.yaml
```

The outputs are in `results/GMoP-ViT-B16/{dataset_name}_{mask setting}_{lam}_{alpha}/{seed}/`

## 🖋️ Citation

If you find this repository useful in your research, please cite the following papers:

```
@inproceedings{hua2025openworldauc,
title={OpenworldAUC: Towards Unified Evaluation and Optimization for Open-world Prompt Tuning}, 
author={Cong Hua and Qianqian Xu and Zhiyong Yang and Zitai Wang and Shilong Bao and Qingming Huang},
booktitle={The Forty-second International Conference on Machine Learning},
year={2025}
}
```

## 📧 Contact us

If you have any detailed questions or suggestions, you can email us: huacong23z@ict.ac.cn. We will reply in 1-2 business days. Thanks for your interest in our work!

## 🌟 Acknowledgements

- Our code is based on the official PyTorch implementation of [DeCoOp](https://github.com/WNJXYK/decoop).
- The evaluation code is based on [OpenAUC](https://github.com/wang22ti/OpenAUC).

