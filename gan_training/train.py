# coding: utf-8
import math
import torch
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd


class Trainer(object):
    def __init__(self, dvae, generator, discriminator, g_optimizer, d_optimizer,
                 reg_param, w_info, beta, beta_step, target_kl):
        self.dvae = dvae
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.reg_param = reg_param
        self.w_info = w_info
        self.beta = beta
        self.beta_step = beta_step
        self.target_kl = target_kl

    def generator_trainstep(self, z, cs):
        toogle_grad(self.generator, True)
        toogle_grad(self.dvae, True)
        toogle_grad(self.discriminator, False)
        self.generator.train()
        self.discriminator.train()
        self.dvae.train()
        self.dvae.zero_grad()
        self.g_optimizer.zero_grad()

        loss = 0.
        c, c_mu, c_logvar = cs
        z_ = torch.cat([z, c], 1)
        x_fake = self.generator(z_)
        d_fake, mu_fake, logstd_fake = self.discriminator(x_fake)

        gloss = self.compute_loss(d_fake, 1)
        loss += gloss

        chs = self.dvae(x_fake, encode_only=True)
        encloss = self.compute_infomax(cs, chs)
        loss += self.w_info*encloss

        loss.backward()
        self.g_optimizer.step()

        return gloss.item(), encloss.item()

    def discriminator_trainstep(self, x_real, z):
        toogle_grad(self.generator, False)
        toogle_grad(self.dvae, False)
        toogle_grad(self.discriminator, True)
        self.generator.train()
        self.discriminator.train()
        self.dvae.train()
        self.d_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()
        reg = 0.

        d_real, mu_real, logstd_real = self.discriminator(x_real)
        dloss_real = self.compute_loss(d_real, 1)
        dloss_real.backward(retain_graph=True)
        reg += self.reg_param * compute_grad2(d_real, x_real).mean()
        kl_real = kl_loss(mu_real, logstd_real).mean()

        # On fake data
        with torch.no_grad():
            c, c_mu, c_logvar = cs = self.dvae(x_real, encode_only=True)
            z_ = torch.cat([z, c], 1)
            x_fake = self.generator(z_)

        x_fake.requires_grad_()
        d_fake, mu_fake, logstd_fake = self.discriminator(x_fake)
        dloss_fake = self.compute_loss(d_fake, 0)
        dloss_fake.backward(retain_graph=True)
        kl_fake = kl_loss(mu_fake, logstd_fake).mean()
        avg_kl = 0.5*(kl_real+kl_fake)
        reg += self.beta * avg_kl
        reg.backward()
        self.update_beta(avg_kl)

        self.d_optimizer.step()
        toogle_grad(self.discriminator, False)

        # Output
        dloss = (dloss_real + dloss_fake)

        return dloss.item(), reg.item(), cs, avg_kl.item()

    def update_beta(self, avg_kl):
        with torch.no_grad():
            new_beta = self.beta - self.beta_step * (self.target_kl - avg_kl)
            new_beta = max(new_beta, 0)
            # print('setting beta from %.2f to %.2f' % (self.reg_param, new_beta))
            self.beta = new_beta

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        loss = F.binary_cross_entropy_with_logits(d_out, targets)
        return loss

    def compute_infomax(self, cs, chs):
        c, c_mu, c_logvar = cs
        ch, ch_mu, ch_logvar = chs
        loss = (math.log(2*math.pi) + ch_logvar + (c-ch_mu).pow(2).div(ch_logvar.exp()+1e-8)).div(2).sum(1).mean()
        return loss


# Utility functions
def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def update_average(model_tgt, model_src, beta):
    toogle_grad(model_src, False)
    toogle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)


def kl_loss(mu, logstd):
    # mu and logstd are b x k x d x d
    # make them into b*d*d x k

    dim = mu.shape[1]
    mu = mu.permute(0, 2, 3, 1).contiguous()
    logstd = logstd.permute(0, 2, 3, 1).contiguous()
    mu = mu.view(-1, dim)
    logstd = logstd.view(-1, dim)

    std = torch.exp(logstd)
    kl = torch.sum(-logstd + 0.5 * (std**2 + mu**2), dim=-1) - (0.5 * dim)

    return kl
