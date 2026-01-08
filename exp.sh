#!/bin/bash

LR=0.01
EPOCHS=5000
BATCH_SIZE=32
SAVE_BEST="--save_best"

python main.py --nn_type BasicFNN --lr $LR --epochs $EPOCHS --batch_size $BATCH_SIZE --save_model_name BasicFNN.pt $SAVE_BEST

python main.py --nn_type LinearGateNet --lr $LR --epochs $EPOCHS --batch_size $BATCH_SIZE --save_model_name LinearGateNet.pt $SAVE_BEST

python main.py --nn_type GatedBasicFNN --lr $LR --epochs $EPOCHS --batch_size $BATCH_SIZE --save_model_name GatedBasicFNN.pt $SAVE_BEST

python main.py --nn_type ResidualFNN --lr $LR --epochs $EPOCHS --batch_size $BATCH_SIZE --save_model_name ResidualFNN.pt $SAVE_BEST

python main.py --nn_type AttentionFNN --lr $LR --epochs $EPOCHS --batch_size $BATCH_SIZE --save_model_name AttentionFNN.pt $SAVE_BEST

python main.py --nn_type ChannelWiseFNN --lr $LR --epochs $EPOCHS --batch_size $BATCH_SIZE --save_model_name ChannelWiseFNN.pt $SAVE_BEST

python main.py --nn_type LightweightFNN --lr $LR --epochs $EPOCHS --batch_size $BATCH_SIZE --save_model_name LightweightFNN.pt $SAVE_BEST

python main.py --nn_type HybridFNN --lr $LR --epochs $EPOCHS --batch_size $BATCH_SIZE --save_model_name HybridFNN.pt $SAVE_BEST

python main.py --nn_type DeepResidualFNN --lr $LR --epochs $EPOCHS --batch_size $BATCH_SIZE --save_model_name DeepResidualFNN.pt $SAVE_BEST
