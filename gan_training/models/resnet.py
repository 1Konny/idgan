import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed
import numpy as np


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, mode='channel', eps=1e-5):
        assert mode in ['channel', 'spatial']
        super(AdaptiveInstanceNorm, self).__init__()

        self.mode = mode
        self.eps = eps
        self.normalizer = self._get_normalizer(mode)
        self.mean = None
        self.std = None

    def forward(self, x):
        x_normalised = self.normalizer(x)
        out = (x_normalised).mul(self.std).add(self.mean)
        return out

    def spatial_normalization(self, x):
        assert x.ndimension() == 4
        x_mean = x.mean(1, keepdim=True)
        x_var = x.var(1, unbiased=False, keepdim=True)
        x_normalised = (x-x_mean).div((x_var+self.eps).sqrt())
        return x_normalised

    def channel_normalization(self, x):
        assert x.ndimension() == 4
        x_normalised = F.instance_norm(x)
        return x_normalised

    def update(self, mean, std):
        self.mean = mean
        self.std = std

    def _get_normalizer(self, mode):
        return self.channel_normalization if mode == 'channel' else self.spatial_normalization


class SubMapper(nn.Module):
    def __init__(self, in_channels, out_channels, adain):
        super(SubMapper, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.adain = adain
        self.mapping = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*2, 1),
                )
    def forward(self, w_base):
        w = self.mapping(w_base)
        mean = w[:, :self.out_channels] 
        std = w[:, self.out_channels:]
        self.adain.update(mean, std)
        return w_base


class Mapper(nn.Module):
    def __init__(self, submappers, z2_dim, hidden_dim=256):
        super(Mapper, self).__init__()
        self.z2_dim = z2_dim
        self.hidden_dim = hidden_dim
        self.base_mapping = nn.Sequential(
                nn.Linear(z2_dim, hidden_dim),
                nn.LeakyReLU(0.2, True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2, True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2, True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2, True),
                View((-1, hidden_dim, 1, 1))
                )
        self.submappers = nn.Sequential(*submappers)
    def forward(self, z2):
        base_w = self.base_mapping(z2)
        for submapper in self.submappers:
            submapper(base_w)
        return None


class GeneratorAdaIN(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, nfilter_max=512, **kwargs):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        self.z_dim = z_dim

        # Submodules

        nlayers = int(np.log2(size / s0))
        map_base_dim = min(nf * 2**(nlayers-0), nf_max)
        self.nf0 = min(nf_max, nf * 2**nlayers)

        self.const_base = nn.Parameter(torch.ones(1, map_base_dim, 4, 4))
        self.const_bias = nn.Parameter(torch.ones(1, map_base_dim, 1, 1))
        self.embedding = nn.Embedding(nlabels, embed_size)
        self.fc = nn.Linear(z_dim + embed_size, self.nf0*s0*s0)

        blocks = []
        submappers = []
        for i in range(nlayers):
            nf0 = min(nf * 2**(nlayers-i), nf_max)
            nf1 = min(nf * 2**(nlayers-i-1), nf_max)
            adain = AdaptiveInstanceNorm()
            submappers += [SubMapper(map_base_dim, nf0, adain)]
            blocks += [
                adain,
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2)
            ]

        adain = AdaptiveInstanceNorm()
        submappers += [SubMapper(map_base_dim, nf, adain)]
        blocks += [
            adain,
            ResnetBlock(nf, nf),
        ]

        self.mapping = Mapper(submappers, z_dim+nlabels, hidden_dim=map_base_dim)
        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, z, y):
        assert(z.size(0) == y.size(0))
        batch_size = z.size(0)

        if y.dtype is torch.int64:
            yembed = self.embedding(y)
        else:
            yembed = y

        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)
        yz = torch.cat([z, yembed], dim=1)
        self.mapping(yz)

        input = self.const_base + self.const_bias
        out = self.resnet(input)
        out = self.conv_img(actvn(out))
        out = F.tanh(out)

        return out


class Generator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, nfilter_max=512, **kwargs):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        self.z_dim = z_dim

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        self.embedding = nn.Embedding(nlabels, embed_size)
        self.fc = nn.Linear(z_dim + embed_size, self.nf0*s0*s0)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2**(nlayers-i), nf_max)
            nf1 = min(nf * 2**(nlayers-i-1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2)
            ]

        blocks += [
            ResnetBlock(nf, nf),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, z, y):
        assert(z.size(0) == y.size(0))
        batch_size = z.size(0)

        if y.dtype is torch.int64:
            yembed = self.embedding(y)
        else:
            yembed = y

        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)

        yz = torch.cat([z, yembed], dim=1)
        out = self.fc(yz)
        out = out.view(batch_size, self.nf0, self.s0, self.s0)

        out = self.resnet(out)

        out = self.conv_img(actvn(out))
        out = F.tanh(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, nfilter_max=1024):
        super().__init__()
        self.embed_size = embed_size
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [
            ResnetBlock(nf, nf)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(3, 1*nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0*s0*s0, nlabels)

        self.hiddens = dict()
        def named_hook(name):
            def hook(module, input, output):
                self.hiddens[name] = output.view(output.size(0), -1)
                return None
            return hook
#        self.resnet.register_forward_hook(named_hook('ds'))

    def forward(self, x, y, zs=None):
        assert(x.size(0) == y.size(0))
        batch_size = x.size(0)

        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0*self.s0*self.s0)
        out_temp = out

#        ds_ratio = None
#        if zs is not None:
#            bs_half = int(batch_size//2)
#            #out1, out2 = out[:bs_half], out[bs_half:bs_half+bs_half]
#            out1, out2 = self.hiddens['ds'][:bs_half], self.hiddens['ds'][bs_half:bs_half+bs_half]
#            z1, z2 = zs[:bs_half], zs[bs_half:bs_half+bs_half]
#            ds_ratio = (out1-out2).view(bs_half, -1).norm(p=2, dim=1).div((z1-z2).norm(p=2, dim=1)).mean()

        out = self.fc(actvn(out))

        index = Variable(torch.LongTensor(range(out.size(0))))
        if y.is_cuda:
            index = index.cuda()
        out = out[index, y]

        #return out, ds_ratio
        return out, out_temp


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out
