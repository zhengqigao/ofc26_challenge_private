#!/bin/bash

LR=0.01
EPOCHS=5000
BATCH_SIZE=128
SAVE_BEST="--save_best"

run_model=$1 # can be BasicFNN, LinearGateNet, GatedBasicFNN, ResidualFNN, AttentionFNN, ChannelWiseFNN, LightweightFNN, HybridFNN, DeepResidualFNN
device=$2 

CUDA_VISIBLE_DEVICES=${device} python main.py --nn_type ${run_model} --lr $LR --epochs $EPOCHS --batch_size $BATCH_SIZE --save_model_name ${run_model}.pt $SAVE_BEST

# python main.py --nn_type LinearGateNet --lr $LR --epochs $EPOCHS --batch_size $BATCH_SIZE --save_model_name LinearGateNet.pt $SAVE_BEST

# python main.py --nn_type GatedBasicFNN --lr $LR --epochs $EPOCHS --batch_size $BATCH_SIZE --save_model_name GatedBasicFNN.pt $SAVE_BEST

# python main.py --nn_type ResidualFNN --lr $LR --epochs $EPOCHS --batch_size $BATCH_SIZE --save_model_name ResidualFNN.pt $SAVE_BEST

# python main.py --nn_type AttentionFNN --lr $LR --epochs $EPOCHS --batch_size $BATCH_SIZE --save_model_name AttentionFNN.pt $SAVE_BEST

# python main.py --nn_type ChannelWiseFNN --lr $LR --epochs $EPOCHS --batch_size $BATCH_SIZE --save_model_name ChannelWiseFNN.pt $SAVE_BEST

# python main.py --nn_type LightweightFNN --lr $LR --epochs $EPOCHS --batch_size $BATCH_SIZE --save_model_name LightweightFNN.pt $SAVE_BEST

# python main.py --nn_type HybridFNN --lr $LR --epochs $EPOCHS --batch_size $BATCH_SIZE --save_model_name HybridFNN.pt $SAVE_BEST

# python main.py --nn_type DeepResidualFNN --lr $LR --epochs $EPOCHS --batch_size $BATCH_SIZE --save_model_name DeepResidualFNN.pt $SAVE_BEST
