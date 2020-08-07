#! /bin/bash

python dvae_main.py --dataset cars --dec_dist gaussian --c_dim 20 --beta 5 --max_iter 1e6 --name dvae_cars
python train.py --config cars-64.yaml --dvae_name dvae_cars --name idgan_cars_64
