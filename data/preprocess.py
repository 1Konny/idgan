import os
import sys
from PIL import Image
from pathlib import Path
from torchvision import transforms


def preprocess_celeba(path):
    crop = transforms.CenterCrop((160, 160))
    resample = Image.LANCZOS
    img = Image.open(path)
    img = crop(img)

    img_256_path = celeba_256_dir / path.name 
    img.resize((256, 256), resample=resample).save(img_256_path)

    img_128_path = celeba_128_dir / path.name 
    img.resize((128, 128), resample=resample).save(img_128_path)

    img_64_path = celeba_64_dir / path.name 
    img.resize((64, 64), resample=resample).save(img_64_path)

    return None


def preprocess_chairs(path):
    crop = transforms.CenterCrop((450, 450))
    resample = Image.LANCZOS
    img = Image.open(path)
    img = crop(img)

    img_64_path = chairs_64_dir / path.parts[-3] / path.name 
    img_64_path.parent.mkdir(parents=True, exist_ok=True)
    img.resize((64, 64), resample=resample).save(img_64_path)

    return None


def preprocess_cars(meta):
    resample = Image.LANCZOS

    path, (x1, y1, x2, y2) = meta
    img = Image.open(path).crop((x1, y1, x2, y2))
    if min(img.size) < 64:
        return

    img_64_path = cars_64_dir / path.parts[-2] / path.parts[-1] 
    img_64_path.parent.mkdir(parents=True, exist_ok=True)
    img.resize((64, 64), resample=resample).save(img_64_path)

    return None


if __name__ == '__main__':
    dataset = sys.argv[1]
    dataroot = Path('data')

    if dataset == 'celeba':
        preprocess = preprocess_celeba
        celeba_64_dir = dataroot / 'CelebA_64' / 'images' 
        celeba_64_dir.mkdir(parents=True, exist_ok=True)
        celeba_128_dir = dataroot / 'CelebA_128' / 'images' 
        celeba_128_dir.mkdir(parents=True, exist_ok=True)
        celeba_256_dir = dataroot / 'CelebA_256' / 'images' 
        celeba_256_dir.mkdir(parents=True, exist_ok=True)

        os.system('unzip -q %s -d %s' % ((dataroot / 'img_align_celeba.zip'), dataroot))
        paths = list((dataroot / 'img_align_celeba').glob('*.jpg'))

    elif dataset == 'celeba-hq':
        preprocess = preprocess_celeba

        os.system('unzip -q %s -d %s' % ((dataroot / 'data1024x1024.zip'), (dataroot / 'CelebA-HQ')))
        paths = []

    elif dataset == 'chairs':
        preprocess = preprocess_chairs
        chairs_64_dir = dataroot / 'Chairs_64'

        os.system('tar -xf %s -C %s' % ((dataroot / 'rendered_chairs.tar'), dataroot))
        paths = list((dataroot / 'rendered_chairs').glob('**/*.png'))

    elif dataset == 'cars':
        preprocess = preprocess_cars
        cars_64_dir = dataroot / 'Cars_64'
        cars_64_dir.mkdir(parents=True, exist_ok=True)

        os.system('tar -xf %s -C %s' % ((dataroot / 'cars_train.tgz'), dataroot))
        os.system('tar -xf %s -C %s' % ((dataroot / 'cars_test.tgz'), dataroot))
        os.system('tar -xf %s -C %s' % ((dataroot / 'car_devkit.tgz'), dataroot))

        from scipy import io
        train_annos = io.loadmat(dataroot / 'devkit' / 'cars_train_annos.mat')
        test_annos = io.loadmat(dataroot / 'devkit' / 'cars_test_annos.mat')

        paths = []
        for anno in train_annos['annotations'][0]:
            path = dataroot / 'cars_train' / anno[-1][0]
            x1, x2, y1, y2 = anno[0][0][0], anno[1][0][0], anno[2][0][0], anno[3][0][0]
            paths.append([path, (x1, x2, y1, y2)])
        for anno in test_annos['annotations'][0]:
            path = dataroot / 'cars_test' / anno[-1][0]
            x1, x2, y1, y2 = anno[0][0][0], anno[1][0][0], anno[2][0][0], anno[3][0][0]
            paths.append([path, (x1, x2, y1, y2)])

    from multiprocessing import Pool
    with Pool(16) as pool:
        pool.map(preprocess, paths)
