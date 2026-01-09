#!/bin/bash

python main.py --epochs 5000 --lr 0.005 --save_best --nn_type BasicFNN --batch_size 32  --seed 1
python main.py --epochs 5000 --lr 0.005 --save_best --nn_type BasicFNN --batch_size 32  --seed 2
python main.py --epochs 5000 --lr 0.005 --save_best --nn_type BasicFNN --batch_size 32  --seed 3
python main.py --epochs 5000 --lr 0.005 --save_best --nn_type BasicFNN --batch_size 32  --seed 4
python main.py --epochs 5000 --lr 0.005 --save_best --nn_type BasicFNN --batch_size 32  --seed 5
python main.py --epochs 5000 --lr 0.005 --save_best --nn_type BasicFNN --batch_size 32  --seed 6
python main.py --epochs 5000 --lr 0.005 --save_best --nn_type BasicFNN --batch_size 32  --seed 7
python main.py --epochs 5000 --lr 0.005 --save_best --nn_type BasicFNN --batch_size 32  --seed 8
python main.py --epochs 5000 --lr 0.005 --save_best --nn_type BasicFNN --batch_size 32  --seed 9
python main.py --epochs 5000 --lr 0.005 --save_best --nn_type BasicFNN --batch_size 32  --seed 10

python kaggle_submit.py
