# Disentangled Representation Learning and Generation with Manifold Optimization

## Abstract
Disentanglement is a useful property in representation learning which increases the interpretability of generative models such as Variational Auto-Encoders (VAE), Generative Adversarial Models, and their many variants. Typically in such models, an increase in disentanglement performance is traded-off with generation quality. In the context of latent space models, this work presents a representation learning framework that explicitly promotes disentanglement by encouraging orthogonal directions of variations. The proposed objective is the sum of an auto-encoder error term along with a Principal Component Analysis reconstruction error in the feature space. This has an interpretation of a Restricted Kernel Machine with an interconnection matrix on the Stiefel manifold. Our analysis shows that such a construction promotes disentanglement by matching the principal directions in latent space with the directions of orthogonal variation in data space. The training algorithm involves a stochastic optimization method on the Stiefel manifold, which increases only marginally the computing time compared to an analogous VAE. Our theoretical discussion and various experiments show that the proposed model improves over many VAE variants in terms of both generation quality and disentangled representation learning.

[paper](https://arxiv.org/abs/2006.07046)

## Usage
After downloading the repository, navigate to the folder and install required python packages with the provided rkm_env.yml file as follows.

### Install packages in conda enviornment
Run the following in terminal. This will create a conda envionment named 'rkm_env'.

```
conda env create -f rkm_env.yml
```

### Train
Activate the conda enviornment `conda activate rkm_env` and run one of the following commands, for example:
```
python train_rkm.py --dataset_name mnist --h_dim 10 --loss deterministic --max_epochs 1000
python train_rkm.py --dataset_name dsprites --h_dim 5 --loss splitloss --max_epochs 1000
```

#### Help
```
usage: train_rkm.py [-h] [--dataset_name DATASET_NAME] [--h_dim H_DIM]
                    [--capacity CAPACITY] [--mb_size MB_SIZE]
                    [--x_fdim1 X_FDIM1] [--x_fdim2 X_FDIM2] [--c_accu C_ACCU]
                    [--noise_level NOISE_LEVEL] [--loss LOSS] [--lr LR]
                    [--lrg LRG] [--max_epochs MAX_EPOCHS] [--proc PROC]
                    [--workers WORKERS] [--shuffle SHUFFLE]

St-RKM Model

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
                        Dataset name: mnist/fashion-
                        mnist/svhn/dsprites/3dshapes/cars3d (default: mnist)
  --h_dim H_DIM         Dim of latent vector (default: 10)
  --capacity CAPACITY   Conv_filters of network (default: 64)
  --mb_size MB_SIZE     Mini-batch size (default: 256)
  --x_fdim1 X_FDIM1     Input x_fdim1 (default: 256)
  --x_fdim2 X_FDIM2     Input x_fdim2 (default: 50)
  --c_accu C_ACCU       Input weight on recons_error (default: 1)
  --noise_level NOISE_LEVEL
                        Noise-level (default: 0.001)
  --loss LOSS           loss type: deterministic/noisyU/splitloss (default:
                        deterministic)
  --lr LR               Input learning rate for ADAM optimizer (default:
                        0.0002)
  --lrg LRG             Input learning rate for Cayley_ADAM optimizer
                        (default: 0.0001)
  --max_epochs MAX_EPOCHS
                        Input max_epoch (default: 1000)
  --proc PROC           device type: cuda or cpu (default: cuda)
  --workers WORKERS     Number of workers for dataloader (default: 16)
  --shuffle SHUFFLE     shuffle dataset: True/False (default: True)

```

## Pretrained models

* Please download the [pre_trained_models](https://www.dropbox.com/sh/jkw9adrm8bpxywd/AAD6YaN22b-1r0qm07IUua9va?dl=0) from the link provided.
* Place it under following directory structure and specify one of the filenames in the `rkm_gen.py` for generation related to that model.
* The directory structure underneath must be as follows
``` python
out
  |____ mnist
            |___ file1.tar
                 file2.tar
                    .
                    .
                    .
  |____ svhn
            |___ file1.tar
                 file2.tar
                    .
                    .
                    .
  |____ 3dshapes
            |___ file1.tar
                 file2.tar
                    .
                    .
                    .
        .
        .
        .
```
* To evaluate a pre-trained model, run for instance `python rkm_gen.py --dataset_name mnist --filename mnist_Trained_rkm_20200320-0837`
