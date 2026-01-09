#!/bin/bash

idx=$1

seed_list=(500 501 502)

for seed in "${seed_list[@]}"; do
    CUDA_VISIBLE_DEVICES=${idx} python main_zq.py --epochs 10000 --lr 0.001 --save_best --nn_type MymodelAttention --batch_size 64 --seed ${seed}
done

python kaggle_submit.py
