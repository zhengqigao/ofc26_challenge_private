Run the command 

```shell
git clone git@github.com:zhengqigao/ofc26_challenge_private.git
cd ofc26_challenge_private
mkdir submission
python main.py --epochs 5000 --lr 0.001 --save_best --nn_type BasicFNN --batch_size 32
```

It will generate a csv file under the folder `submission` which can be used to direcly upload to Kaggle.