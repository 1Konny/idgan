from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from math import sqrt
from gan_training.metrics import inception_score

from scipy.stats import truncnorm

def truncated_z_sample(batch_size, z_dim, truncation=1., seed=None):
    values = truncnorm.rvs(-2, 2, size=(batch_size, z_dim))
    return torch.from_numpy(truncation * values)


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


def generator_postprocess(x):
    return x.add(1).div(2).clamp(0, 1)

def decoder_postprocess(x):
    return torch.sigmoid(x)

gp = generator_postprocess
dp = decoder_postprocess

class Evaluator(object):
    def __init__(self, generator, zdist, batch_size=64,
                 inception_nsamples=60000, device=None, dvae=None):
        self.generator = generator
        self.zdist = zdist
        self.inception_nsamples = inception_nsamples
        self.batch_size = batch_size
        self.device = device
        self.dvae = dvae

    def compute_inception_score(self):
        self.generator.eval()
        imgs = []
        while(len(imgs) < self.inception_nsamples):
            ztest = self.zdist.sample((self.batch_size,))

            samples = self.generator(ztest, ytest)
            samples = [s.data.cpu().numpy() for s in samples]
            imgs.extend(samples)

        imgs = imgs[:self.inception_nsamples]
        score, score_std = inception_score(
            imgs, device=self.device, resize=True, splits=10
        )

        return score, score_std

    def create_samples(self, z):
        self.generator.eval()
        batch_size = z.size(0)

        # Sample x
        with torch.no_grad():
            x = self.generator(z)
        return x


class DisentEvaluator(object):
    def __init__(self, generator, zdist, cdist, batch_size=64,
                 device=None, dvae=None):
        self.generator = generator
        self.zdist = zdist
        self.cdist = cdist
        self.batch_size = batch_size
        self.device = device
        self.dvae = dvae


    @torch.no_grad()
    def save_samples(self, zdist, cdist, out_dir, batch_size=64, N=50000):
        import os
        from torchvision.utils import save_image
        self.generator.eval()

        samples = []
        quotient, remainder = N//batch_size, N%batch_size
        with tqdm(range(quotient)) as pbar:
            for _ in pbar:
                z = self.zdist.sample((batch_size,))
                c = self.cdist.sample((batch_size,))
                z_ = torch.cat([z, c], 1)
                sample = gp(self.generator(z_)).data.cpu()
                samples.append(sample)     
                pbar.set_description('generating samples...')

        if remainder > 0:
            z = self.zdist.sample((remainder,))
            c = self.cdist.sample((remainder,))
            z_ = torch.cat([z, c], 1)
            sample = gp(self.generator(z_)).data.cpu()
            samples.append(sample)     

        samples = torch.cat(samples, 0)
        with tqdm(enumerate(samples)) as pbar:
            for i, sample in pbar:
                path = os.path.join(out_dir, '{}.png'.format(i+1))
                save_image(sample, path, 1, 0)
                pbar.set_description('now saving samples...')

        return None

    @torch.no_grad()
    def create_samples(self, z, c):
        self.generator.eval()
        bs =  z.size(0)
        nrow = ncol = int(sqrt(bs))
        bs = nrow*ncol
        z = z[:bs]
        c = c[:bs]

        # point-wise synthesis of z and c
        z_ = torch.cat([z, c], 1)
        x_point = gp(self.generator(z_))
        x_point = make_grid(x_point, nrow=ncol, padding=2, pad_value=1)

        # single z
        z_ = torch.cat([z[:1].repeat(bs, 1), c], 1)
        x_singlez = gp(self.generator(z_))
        x_singlez = make_grid(x_singlez, nrow=ncol, padding=2, pad_value=1)

        # single c
        z_ = torch.cat([z, c[:1].repeat(bs, 1)], 1)
        x_singlec = gp(self.generator(z_))
        x_singlec = make_grid(x_singlec, nrow=ncol, padding=2, pad_value=1)

        # 1st column c and 1st row z
        cc = c.view(nrow, ncol, -1).permute(1, 0, 2).contiguous().view(bs, -1)[:ncol].unsqueeze(1).repeat(1, ncol, 1).view(bs, -1)
        zz = z[:ncol].unsqueeze(0).repeat(nrow, 1, 1).view(bs, -1)
        z_ = torch.cat([zz, cc], 1)
        x_fcfz = gp(self.generator(z_))
        x_fcfz = make_grid(x_fcfz, nrow=ncol, padding=2, pad_value=1)

        return x_point, x_singlez, x_singlec, x_fcfz

    @torch.no_grad()
    def traverse_c1z1(self, z=None, c=None, limit=3, ncol=7, dims=-1):
        # traverse with a single c and z
        self.generator.eval()
        interpolation = torch.linspace(-limit, limit, ncol)
        if z is None:
            z = self.zdist.sample((1,))
        if c is None:
            c = self.cdist.sample((1,))

        idgan_samples = []
        idgan_samples_p = []
        dvae_samples = []
        dvae_samples_p = []
        c_ori = c.clone()
        for c_dim in range(self.cdist.dim):
            if dims != -1 and c_dim not in dims:
                continue

            c = c_ori.clone()
            c_ = c_ori.clone()
            c_zero = torch.zeros_like(c)
            for val in interpolation:
                c[:, c_dim] = val
                z_ = torch.cat([z, c], 1)
                idgan_sample = F.adaptive_avg_pool2d(gp(self.generator(z_)), (64, 64)).data.cpu()
                idgan_samples.append(idgan_sample)
                dvae_sample = dp(self.dvae(c=c, decode_only=True)).data.cpu()
                dvae_samples.append(dvae_sample)

                c_zero[:, c_dim] = val
                c_p = c_ + c_zero
                z_p_ = torch.cat([z, c_p], 1)
                idgan_sample_p = F.adaptive_avg_pool2d(gp(self.generator(z_p_)), (64, 64)).data.cpu()
                idgan_samples_p.append(idgan_sample_p)
                dvae_sample_p = dp(self.dvae(c=c_p, decode_only=True)).data.cpu()
                dvae_samples_p.append(dvae_sample_p)

        idgan_samples = torch.cat(idgan_samples, dim=0)
        idgan_samples = make_grid(idgan_samples, nrow=ncol, padding=2, pad_value=1)
        dvae_samples = torch.cat(dvae_samples, dim=0)
        dvae_samples = make_grid(dvae_samples, nrow=ncol, padding=2, pad_value=1)

        idgan_samples_p = torch.cat(idgan_samples_p, dim=0)
        idgan_samples_p = make_grid(idgan_samples_p, nrow=ncol, padding=2, pad_value=1)
        dvae_samples_p = torch.cat(dvae_samples_p, dim=0)
        dvae_samples_p = make_grid(dvae_samples_p, nrow=ncol, padding=2, pad_value=1)

        x = torch.stack([idgan_samples, dvae_samples, idgan_samples_p, dvae_samples_p])
        x = make_grid(x, nrow=4, padding=4, pad_value=0)

        return x
