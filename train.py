import random
import numpy as np
import argparse
import os
from os import path
from tqdm import tqdm
import time
import copy
import torch
from torch import nn
from gan_training import utils
from gan_training.train import Trainer, update_average
from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import DisentEvaluator
from gan_training.config import (
    load_config, build_models, build_optimizers, build_lr_scheduler,
)

# Arguments
parser = argparse.ArgumentParser(
    description='Train a GAN with different regularization strategies.'
)
parser.add_argument('--config_dir', default='./configs', type=str, help='Path to configs directory')
parser.add_argument('--output_dir', default='./outputs', type=str, help='Path to outputs directory')
parser.add_argument('--dvae_name', default='none', type=str, help='Name of the experiment for pre-training disentangling VAE')
parser.add_argument('--config', type=str, help='Name of base config file')
parser.add_argument('--name', type=str, help='Name of the experiment')
parser.add_argument('--nf', '--nfilter', default=-1, type=int, help='Base number of filters')
parser.add_argument('--bs', '--batch_size', default=-1, type=int, help='Batch size')
parser.add_argument('--reg_param', default=-1, type=float, help='R1 regularization parameter')
parser.add_argument('--w_info', default=-1, type=float, help='weighting constant on ID Loss')
parser.add_argument('--mi', '--max_iter', default=-1, type=int, help='Max training iteration')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda')
parser.add_argument('--seed', default=1, type=int, help='Random Seed')

args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

config_path = os.path.join(args.config_dir, args.config)
config = load_config(config_path)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)

# = = = = = Customized Configurations = = = = = #
out_dir = os.path.join(args.output_dir, args.name)
config['training']['out_dir'] = out_dir
if args.nf > 0:
    config['generator']['kwargs']['nfilter'] = args.nf 
    config['discriminator']['kwargs']['nfilter'] = args.nf 
if args.bs > 0:
    config['training']['batch_size'] = args.bs
if args.reg_param > 0:
    config['training']['reg_param'] = args.reg_param
if args.w_info > 0:
    config['training']['w_info'] = args.w_info
if args.mi > 0:
    max_iter = config['training']['max_iter'] = args.mi
else:
    max_iter = config['training']['max_iter']
if args.dvae_name != 'none':
    config['dvae']['runname'] = args.dvae_name
# = = = = = Customized Configurations = = = = = #

# Short hands
batch_size = config['training']['batch_size']
d_steps = config['training']['d_steps']
restart_every = config['training']['restart_every']
inception_every = config['training']['inception_every']
save_every = config['training']['save_every']
backup_every = config['training']['backup_every']

out_dir = config['training']['out_dir']
checkpoint_dir = path.join(out_dir, 'chkpts')

# Create missing directories
if not path.exists(out_dir):
    os.makedirs(out_dir)
if not path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Logger
checkpoint_io = CheckpointIO(
    checkpoint_dir=checkpoint_dir
)

device = torch.device("cuda:0" if is_cuda else "cpu")

train_dataset = get_dataset(
    name=config['data']['type'],
    data_dir=config['data']['train_dir'],
    size=config['data']['img_size'],
)
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config['training']['nworkers'],
        shuffle=True, pin_memory=True, sampler=None, drop_last=True
)

# Create models
dvae, generator, discriminator = build_models(config)
dvae_ckpt_path = os.path.join(args.output_dir, config['dvae']['runname'], 'chkpts', config['dvae']['ckptname'])
dvae_ckpt = torch.load(dvae_ckpt_path)['model_states']['net']
dvae.load_state_dict(dvae_ckpt)

tqdm.write('{}'.format(dvae))
tqdm.write('{}'.format(generator))
tqdm.write('{}'.format(discriminator))

# Put models on gpu if needed
dvae = dvae.to(device)
generator = generator.to(device)
discriminator = discriminator.to(device)

g_optimizer, d_optimizer = build_optimizers(
    generator, discriminator, dvae, config
)

# Use multiple GPUs if possible
dvae = nn.DataParallel(dvae)
generator = nn.DataParallel(generator)
discriminator = nn.DataParallel(discriminator)

# Register modules to checkpoint
checkpoint_io.register_modules(
    generator=generator,
    discriminator=discriminator,
    g_optimizer=g_optimizer,
    d_optimizer=d_optimizer,
)

# Logger
logger = Logger(
    log_dir=path.join(out_dir, 'logs'),
    img_dir=path.join(out_dir, 'imgs'),
    monitoring=config['training']['monitoring'],
    monitoring_dir=path.join(out_dir, 'monitoring')
)

# Distributions
cdist = get_zdist('gauss', config['dvae']['c_dim'], device=device)
zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                  device=device)

# Save for tests
ntest = batch_size
x_real, ytest = utils.get_nsamples(train_loader, ntest)
ztest = zdist.sample((ntest,))
ctest = cdist.sample((ntest,))
ztest_ = torch.cat([ztest, ctest], 1)
utils.save_images(x_real, path.join(out_dir, 'real.png'))

# Test generator
if config['training']['take_model_average']:
    generator_test = copy.deepcopy(generator)
    checkpoint_io.register_modules(generator_test=generator_test)
else:
    generator_test = generator

# Evaluator
dis_evaluator = DisentEvaluator(generator=generator_test, zdist=zdist, cdist=cdist,
                                batch_size=batch_size, device=device, dvae=dvae)

# Train
tstart = t0 = time.time()
it = epoch_idx = -1

# Load checkpoint if existant
it = checkpoint_io.load('model.pt')
if it != -1:
    logger.load_stats('stats.p')

# Reinitialize model average if needed
if (config['training']['take_model_average']
        and config['training']['model_average_reinit']):
    update_average(generator_test, generator, 0.)

# Learning rate anneling
g_scheduler = build_lr_scheduler(g_optimizer, config, last_epoch=it)
d_scheduler = build_lr_scheduler(d_optimizer, config, last_epoch=it)

# Trainer
trainer = Trainer(
    dvae, generator, discriminator, g_optimizer, d_optimizer,
    reg_param=config['training']['reg_param'],
    w_info = config['training']['w_info'],
    beta = config['training']['beta'],
    beta_step = config['training']['beta_step'],
    target_kl = config['training']['target_kl'],
)

# Training loop
tqdm.write('Start training...')
pbar = tqdm(total=max_iter)
if it > 0:
    pbar.update(it)

out = False
while not out:
    epoch_idx += 1
    tqdm.write('Start epoch %d...' % epoch_idx)

    for x_real, _ in train_loader:
        it += 1
        pbar.update(1)
        g_scheduler.step()
        d_scheduler.step()

        d_lr = d_optimizer.param_groups[0]['lr']
        g_lr = g_optimizer.param_groups[0]['lr']
        logger.add('learning_rates', 'discriminator', d_lr, it=it)
        logger.add('learning_rates', 'generator', g_lr, it=it)

        x_real = x_real.to(device)

        # Discriminator updates
        z = zdist.sample((batch_size,))
        dloss, reg, cs, avg_kl = trainer.discriminator_trainstep(x_real, z)
        logger.add('losses', 'discriminator', dloss, it=it)
        logger.add('losses', 'regularizer', reg, it=it)
        logger.add('losses', 'avg_kl', avg_kl, it=it)
        logger.add('losses', 'beta', trainer.beta, it=it)

        # Generators updates
        if ((it + 1) % d_steps) == 0:
            z = zdist.sample((batch_size,))
            gloss, encloss = trainer.generator_trainstep(z, cs)
            logger.add('losses', 'generator', gloss, it=it)
            logger.add('losses', 'encoder', encloss, it=it)

            if config['training']['take_model_average']:
                update_average(generator_test, generator,
                               beta=config['training']['model_average_beta'])

        # Print stats
        g_loss_last = logger.get_last('losses', 'generator')
        d_loss_last = logger.get_last('losses', 'discriminator')
        e_loss_last = logger.get_last('losses', 'encoder')
        d_reg_last = logger.get_last('losses', 'regularizer')
        tqdm.write('[epoch %0d, it %4d] g_loss = %.4f, d_loss = %.4f, e_loss = %.4f, reg=%.4f avg_kl=%.4f beta=%.4f'
              % (epoch_idx, it, g_loss_last, d_loss_last, e_loss_last, d_reg_last, avg_kl, trainer.beta))

        # (i) Sample if necessary
        if (it % config['training']['sample_every']) == 0:
            tqdm.write('Creating samples...')

            # samples
            x_point, x_singlez, x_singlec, x_fcfz = dis_evaluator.create_samples(ztest, ctest)
            logger.add_imgs(x_point, 'prior_point-wise_z_and_c', it)
            logger.add_imgs(x_singlez, 'prior_single_z_and_grid_c', it)
            logger.add_imgs(x_singlec, 'prior_single_c_and_grid_z', it)
            logger.add_imgs(x_fcfz, 'prior_1st_column_c_and_1st_row_z', it)

            # traverses
            x_traverse = dis_evaluator.traverse_c1z1(ztest[:1], ctest[:1])
            logger.add_imgs(x_traverse, 'prior_traverse', it)

            # samples
            ztest_ = torch.randn_like(ztest)
            x_point, x_singlez, x_singlec, x_fcfz = dis_evaluator.create_samples(ztest_, cs[1])
            logger.add_imgs(x_point, 'mu_point-wise_z_and_c', it)
            logger.add_imgs(x_singlez, 'mu_single_z_and_grid_c', it)
            logger.add_imgs(x_singlec, 'mu_single_c_and_grid_z', it)
            logger.add_imgs(x_fcfz, 'mu_1st_column_c_and_1st_row_z', it)

            # traverses
            x_traverse = dis_evaluator.traverse_c1z1(ztest_[:1], cs[1][:1])
            logger.add_imgs(x_traverse, 'mu_traverse', it)

        # (iii) Backup if necessary
        if ((it + 1) % backup_every) == 0:
            tqdm.write('Saving backup...')
            checkpoint_io.save(it, 'model_%08d.pt' % it)
            logger.save_stats('stats_%08d.p' % it)
            checkpoint_io.save(it, 'model.pt')
            logger.save_stats('stats.p')

        # (iv) Save checkpoint if necessary
        if time.time() - t0 > save_every:
            tqdm.write('Saving checkpoint...')
            checkpoint_io.save(it, 'model.pt')
            logger.save_stats('stats.p')
            t0 = time.time()

            if (restart_every > 0 and t0 - tstart > restart_every):
                exit(3)

        if it >= max_iter:
            tqdm.write('Saving backup...')
            checkpoint_io.save(it, 'model_%08d.pt' % it)
            logger.save_stats('stats_%08d.p' % it)
            checkpoint_io.save(it, 'model.pt')
            logger.save_stats('stats.p')
            out = True
            break
