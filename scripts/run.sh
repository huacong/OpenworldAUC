#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py --seed 1 --lam 1.0 --alpha 1.0 --epoch 10 --dataset OpenworldAUC/configs/datasets/imagenet.yaml
CUDA_VISIBLE_DEVICES=1 python main.py --seed 1 --lam 1.0 --alpha 0.5 --epoch 40 --dataset OpenworldAUC/configs/datasets/caltech101.yaml
CUDA_VISIBLE_DEVICES=2 python main.py --seed 1 --lam 1.0 --alpha 0.5 --epoch 40 --dataset OpenworldAUC/configs/datasets/oxford_pets.yaml