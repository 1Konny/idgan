#! /bin/bash

python dvae_main.py --dataset celeba --dec_dist gaussian --c_dim 20 --beta 6.4 --max_iter 1e6 --name dvae_celeba
python train.py --config celebA-64.yaml --dvae_name dvae_celeba --name idgan_celeba_64
python train.py --config celebA-256.yaml --dvae_name dvae_celeba --name idgan_celeba_256
python train.py --config celebA-HQ.yaml --dvae_name dvae_celeba --name idgan_celeba_hq
