#! /bin/bash

python dvae_main.py --dataset dsprites --dec_dist bernoulli --c_dim 10 --beta 8 --max_iter 5e5 --name dvae_dsprites
python train.py --config dsprites.yaml --dvae_name dvae_dsprites --name idgan_dsprites
