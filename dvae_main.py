import argparse
import random
import numpy as np
import torch

from gan_training.dvae.solver import Solver


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter', default=3e5, type=float, help='maximum training iteration')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')

    parser.add_argument('--c_dim', default=10, type=int, help='dimension of the representation c')
    parser.add_argument('--beta', default=4, type=float, help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--dec_dist', default='gaussian', type=str, choices=['bernoulli', 'gaussian'], help='decoder distribution')

    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='CelebA', type=str, choices=['dsprites', 'celeba', 'chairs', 'cars'], help='dataset name')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')

    parser.add_argument('--name', default='main', type=str, help='run name')
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--load_ckpt', default=-1, type=int, help='-1: load last | 0: do not load | X: load X-th')

    parser.add_argument('--log_line_iter', default=100, type=int, help='')
    parser.add_argument('--log_img_iter', default=1000, type=int, help='')
    parser.add_argument('--ckpt_save_iter', default=10000, type=int, help='')


    args = parser.parse_args()
    print(args)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    net = Solver(args)
    net.train()
