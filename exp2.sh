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
seeds=(200)

# 1. Runs with lr=0.001, no scheduler
for seed in "${seeds[@]}"; do
    CUDA_VISIBLE_DEVICES="${idx}" python main_zq.py \
        --epochs 5000 \
        --lr 0.001 \
        --save_best \
        --nn_type MymodelAttention \
        --batch_size 512 \
        --seed "${seed}" \
        --hidden_embed_dim "${hidden_embed_dim}" \
        --hidden_dim "${hidden_dim}" \
        --num_layers "${num_layers}" \
        --cosmos_ratio 20 \
        --batch_size 256 

    CUDA_VISIBLE_DEVICES="${idx}" python main_zq.py \
        --epochs 2000 \
        --lr 0.0005 \
        --save_best \
        --nn_type MymodelAttention \
        --seed "${seed}" \
        --hidden_embed_dim "${hidden_embed_dim}" \
        --hidden_dim "${hidden_dim}" \
        --num_layers "${num_layers}" \
        --cosmos_ratio 0 \
        --batch_size 128 \
        --resume_from "./model/model_MymodelAttention.pt"
done