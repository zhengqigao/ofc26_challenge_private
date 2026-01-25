#!/bin/bash

# Exit immediately if any command fails
set -e

# Check if enough arguments are passed
if [ $# -lt 4 ]; then
    echo "Usage: $0 <gpu_idx> <hidden_embed_dim> <hidden_dim> <num_layers>"
    exit 1
fi

idx="$1"
hidden_embed_dim="$2"
hidden_dim="$3"
num_layers="$4"

# Array of seeds
seeds=(300)

# 1. Runs with lr=0.001, no scheduler
for seed in "${seeds[@]}"; do
    CUDA_VISIBLE_DEVICES="${idx}" python main_zq.py \
        --epochs 1000 \
        --lr 0.001 \
        --save_best \
        --nn_type MymodelAttention \
        --batch_size 64 \
        --seed "${seed}" \
        --hidden_embed_dim "${hidden_embed_dim}" \
        --hidden_dim "${hidden_dim}" \
        --num_layers "${num_layers}" \
        --cosmos_ratio 10 \
        --batch_size 128 

    CUDA_VISIBLE_DEVICES="${idx}" python main_zq.py \
        --epochs 1000 \
        --lr 0.001 \
        --save_best \
        --nn_type MymodelAttention \
        --batch_size 64 \
        --seed "${seed}" \
        --hidden_embed_dim "${hidden_embed_dim}" \
        --hidden_dim "${hidden_dim}" \
        --num_layers "${num_layers}" \
        --cosmos_ratio 0 \
        --batch_size 128 \
        --resume_from "./model/model_MymodelAttention.pt"
done