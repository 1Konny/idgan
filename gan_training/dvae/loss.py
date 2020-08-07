import torch
import torch.nn.functional as F


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(batch_size)
    elif distribution == 'gaussian':
        x_recon = torch.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum').div(batch_size)
    else:
        raise NotImplementedError

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0

    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    kld = (-0.5*(1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean(0, True)
    return kld
