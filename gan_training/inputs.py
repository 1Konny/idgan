import os
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor, type, transform=None):
        self.type = type
        self.data_tensor = data_tensor
        self.transform = transform

        def idt(x):
            return x

        if type == 'dsprites':
            self.transform2 = idt
        elif type == 'cdsprites':
            self.num_values = 8
            self.color_values = torch.linspace(0, 1, self.num_values)
            self.transform2 = self.colorize
        elif type == 'ndsprites':
            self.transform2 = self.apply_noise
        elif type == 'sdsprites':
            from PIL import Image
            if not os.path.exists('The_Scream.jpg'):
                os.system('wget https://upload.wikimedia.org/wikipedia/commons/f/f4/The_Scream.jpg')
            scream = Image.open('The_Scream.jpg')
            scream.thumbnail((350, 274, 3))
            self.scream = scream
            self.transform2 = self.apply_scream
            self.totensor = transforms.ToTensor()
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        img = self.data_tensor[index]
        img = self.transform2(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.zeros((1,))

    def __len__(self):
        return self.data_tensor.size(0)

    def colorize(self, x):
        r, g, b = x.repeat(3, 1, 1).split(1, 0)
        r *= random.choice(self.color_values)
        g *= random.choice(self.color_values)
        b *= random.choice(self.color_values)
        x = torch.cat([r, g, b], 0)
        return x

    def apply_noise(self, x):
        x = x.repeat(3, 1, 1)
        x = torch.where(x==0, torch.rand_like(x), x).clamp(0,1)
        return x

    def apply_scream(self, x):
        x_crop = random.randint(0, self.scream.size[0]-64)
        y_crop = random.randint(0, self.scream.size[1]-64)
        scream_cropped = self.scream.crop((x_crop, y_crop, x_crop+64, y_crop+64))
        th_scream_cropped = (self.totensor(scream_cropped) + torch.rand((3, 1, 1))).div(2).clamp(0, 1)
        x = torch.where(x<0.5, th_scream_cropped, 1-th_scream_cropped)
        return x

    def save_samples(self, N, root='./true_samples'):
        from torchvision.utils import save_image
        save_dir = os.path.join(root, self.type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        sample_idxs = random.choices(list(range(len(self))), k=N)
        for i, sample_idx in enumerate(sample_idxs):
            path = os.path.join(save_dir, '{}.jpg'.format(i+1))
            save_image(self.__getitem2__(sample_idx), path, 1, 0)

        return None


def npy_loader(path):
    img = np.load(path)

    if img.dtype == np.uint8:
        img = img.astype(np.float32)
        img = img/127.5 - 1.
    elif img.dtype == np.float32:
        img = img * 2 - 1.
    else:
        raise NotImplementedError

    img = torch.Tensor(img)
    if len(img.size()) == 4:
        img.squeeze_(0)

    return img


def get_dataset(name, data_dir, size=64, lsun_categories=None):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size())),
    ])

    if name == 'image':
        dataset = datasets.ImageFolder(data_dir, transform)
    elif name == 'npy':
        # Only support normalization for now
        dataset = datasets.DatasetFolder(data_dir, npy_loader, ('npy',))
    elif name == 'synthetic':
        def transform(x):
            return x*2-1
        data_path = os.path.join(data_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        data_tensor = np.load(data_path, encoding='bytes')
        data_tensor = torch.from_numpy(data_tensor['imgs']).unsqueeze(1).float()
        dataset = CustomTensorDataset(data_tensor, type='dsprites', transform=transform)
    else:
        raise NotImplemented

    return dataset
