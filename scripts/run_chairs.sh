#! /bin/bash

python dvae_main.py --dataset chairs --dec_dist gaussian --c_dim 20 --beta 10 --max_iter 1e6 --name dvae_chairs
python train.py --config chairs-64.yaml --dvae_name dvae_chairs --name idgan_chairs_64
