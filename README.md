# ID-GAN
Pytorch implementation on "High-fidelity Synthesis with Disentangled Representation" (https://arxiv.org/abs/2001.04296). <br>
For ID-GAN augmented with Variational Discriminator Bottleneck (VDB) or VGAN, please refer to the vgan [branch](https://github.com/1Konny/idgan/tree/vgan).

# Usage

## Prepare datasets 
- Create `data` directory, and put the necessary datasets inside here.
```
mkdir data
```

- dSprites dataset.
```
cd data
git clone https://github.com/deepmind/dsprites-dataset.git
cd dsprites-dataset
rm -rf .git* *.md LICENSE *.ipynb *.gif *.hdf5
```

- CelebA dataset.
1. Go to the official website [(link)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and download `img_align_celeba.zip` file to `data` directory.
```
data
|- preprocess.py
|_ img_align_celeba.zip
```
2. Preprocess the data.
```
python data/preprocess.py celeba
```

- CelebA-HQ dataset.
1. Go to the google drive [(link)](https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P) and download `data1024x1024.zip` file to `data` directory.
```
data
|- preprocess.py
|_ data1024x1024.zip
```
2. Preprocess the data.
```
python data/preprocess.py celeba-hq
```

- 3D Chairs dataset.
1. Go to the official website [(link)](https://www.di.ens.fr/willow/research/seeing3Dchairs/) and download `rendered_chairs.tar` file to `data` directory.
```
data
|- preprocess.py
|_ rendered_chairs.tar
```
2. Preprocess the data.
```
python data/preprocess.py chairs 
```

- 3D Cars dataset.
1. Go to the official website [(link)](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) and download `cars_train.tgz`, `cars_test.tgz`, and `car_devkit.tgz` files to `data` directory.
```
data
|- preprocess.py
|- cars_train.tgz 
|- cars_test.tgz 
|_ car_devkit.tgz 
```
2. Preprocess the data.
```
python data/preprocess.py cars 
```

## Train 
- You can run pre-defined commands as follows
```
bash scripts/run_dsprites.sh
bash scripts/run_celeba.sh
bash scripts/run_chairs.sh
bash scripts/run_cars.sh
```

- Stage 1: Train VAEs.
```
python dvae_main.py --dataset [dataset_name] --name [dvae_run_name] --c_dim [c_dim] --beta [beta]
```
, where `[dataset_name]` can be one of `dsprites`, `celeba`, `cars`, and `chairs`.
please refer to `dvae_main.py` for the details.

- Stage 2: Train ID-GAN through information distillation loss.
```
python train.py --config [config_name] --dvae_name [dvae_run_name] --name [idgan_run_name]
```
please refer to `configs` directory for `[config_name]`.

## Results
Results, including checkpoints, tensorboard logs, and images can be found in `outputs` directory.

# Acknowledgement
This code is built on the repos as follows:
1. Beta-VAE: [https://www.github.com/1Konny](https://www.github.com/1Konny)
2. GAN with R2 regularization: [https://github.com/LMescheder/GAN_stability](https://github.com/LMescheder/GAN_stability)
3. VGAN: [https://github.com/akanazawa/vgan](https://github.com/akanazawa/vgan) 

# Citation
If you find our work useful for your research, please cite our paper.
```
@article{lee2020highfidelity, 
    title={High-Fidelity Synthesis with Disentangled Representation}, 
    author={Wonkwang Lee and Donggyun Kim and Seunghoon Hong and Honglak Lee}, 
    year={2020}, 
    journal={arXiv preprint arXiv:2001.04296}, 
}
```
