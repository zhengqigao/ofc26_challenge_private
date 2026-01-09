#!/bin/bash

python main.py --epochs 20000 --lr 0.001  --save_best --nn_type SpectralCNN --batch_size 64  --seed 1
python main.py --epochs 20000 --lr 0.001 --save_best --nn_type SpectralCNN --batch_size 64  --seed 2
python main.py --epochs 20000 --lr 0.001 --save_best --nn_type SpectralCNN --batch_size 64  --seed 3
python main.py --epochs 20000 --lr 0.001 --save_best --nn_type SpectralCNN --batch_size 64  --seed 4
python main.py --epochs 20000 --lr 0.001 --save_best --nn_type SpectralCNN --batch_size 64  --seed 5


python kaggle_submit.py
