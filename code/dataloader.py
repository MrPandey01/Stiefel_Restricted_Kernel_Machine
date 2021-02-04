import os
import glob
import PIL
import scipy.io as sio
import h5py
import numpy as np
from sklearn.utils.extmath import cartesian
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from utils import Resize


def get_dataloader(args):
    print('Loading data...')
    if args.dataset_name == 'mnist':
        return get_mnist_dataloader(args=args)

    elif args.dataset_name == 'fashion-mnist':
        return get_fashion_mnist_dataloaders(args=args)

    elif args.dataset_name == 'svhn':
        return get_svhn_dataloader(args=args)

    elif args.dataset_name == 'cars3d':
        return get_cars3d_dataloader(args=args)

    elif args.dataset_name == '3dshapes':
        return get_3dshapes_dataloader(args=args)

    elif args.dataset_name == 'dsprites':
        return get_dsprites_dataloader(args=args)


def get_mnist_dataloader(args, path_to_data='mnist'):
    """MNIST dataloader with (28, 28) images."""

    all_transforms = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(path_to_data, train=True, download=True, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(train_loader))[0].size()
    return train_loader, c*x*y, c


def get_fashion_mnist_dataloaders(args, path_to_data='fashion-mnist'):
    """FashionMNIST dataloader with (28, 28) images."""

    all_transforms = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.FashionMNIST(path_to_data, train=True, download=True, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(train_loader))[0].size()
    return train_loader, c*x*y, c


def get_svhn_dataloader(args, path_to_data='svhn'):
    """SVHN dataloader with (28, 28) images."""

    all_transforms = transforms.Compose([transforms.Resize(28),
                                         transforms.ToTensor()])
    train_data = datasets.SVHN(path_to_data, split='train', download=True, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(train_loader))[0].size()
    return train_loader, c*x*y, c


def get_cars3d_dataloader(args, path_to_data='cars3d'):
    """Cars3D dataloader with (64, 64, 3) images."""

    name = '{}/data/cars/'.format(path_to_data)
    if not os.path.exists(name):
        print('Data at the given path doesn\'t exist. Downloading now...')
        os.system(" mkdir cars3d/;"
                  " wget -O cars3d/nips2015-analogy-data.tar.gz http://www.scottreed.info/files/nips2015-analogy-data.tar.gz ;"
                  " cd cars3d/; tar xzf nips2015-analogy-data.tar.gz")

    all_transforms = transforms.Compose([transforms.ToTensor()])

    cars3d_data = cars3dDataset(path_to_data, transform=all_transforms)
    cars3d_loader = DataLoader(cars3d_data, batch_size=args.mb_size,
                                 shuffle=args.shuffle, pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(cars3d_loader))[0].size()
    return cars3d_loader, c*x*y, c


def get_dsprites_dataloader(args, path_to_data='dsprites'):
    """DSprites dataloader (64, 64) images"""

    name = '{}/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'.format(path_to_data)
    if not os.path.exists(name):
        print('Data at the given path doesn\'t exist. Downloading now...')
        os.system("  mkdir dsprites;"
                  "  wget -O dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")

    transform = transforms.Compose([transforms.ToTensor()])

    dsprites_data = DSpritesDataset(name, transform=transform)
    dsprites_loader = DataLoader(dsprites_data, batch_size=args.mb_size,
                                 shuffle=args.shuffle, pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(dsprites_loader))[0].size()
    return dsprites_loader, c*x*y, c


def get_3dshapes_dataloader(args, path_to_data='3dshapes'):
    """3dshapes dataloader with images rescaled to (28,28,3)"""

    name = '{}/3dshapes.h5'.format(path_to_data)
    if not os.path.exists(name):
        print('Data at the given path doesn\'t exist. ')
        os.system("  mkdir 3dshapes;"
                  "  wget -O 3dshapes/3dshapes.h5 https://storage.googleapis.com/3d-shapes/3dshapes.h5")

    transform = transforms.Compose([Resize(28), transforms.ToTensor()])

    d3shapes_data = d3shapesDataset(name, transform=transform)
    d3shapes_loader = DataLoader(d3shapes_data, batch_size=args.mb_size,
                                 shuffle=args.shuffle, pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(d3shapes_loader))[0].size()
    return d3shapes_loader, c*x*y, c


class DSpritesDataset(Dataset):
    """DSprites dataloader class"""

    lat_names = ('shape', 'scale', 'orientation', 'posX', 'posY')
    lat_sizes = np.array([3, 6, 40, 32, 32])

    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        dat = np.load(path_to_data)
        self.imgs = dat['imgs'][::subsample]
        self.lv = dat['latents_values'][::subsample]
        # self.lc = dat['latents_classes'][::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = self.imgs[idx] * 255
        sample = sample.reshape(sample.shape + (1,))

        if self.transform:
            sample = self.transform(sample)
        return sample, self.lv[idx]


class d3shapesDataset(Dataset):
    """3dshapes dataloader class"""

    lat_names = ('floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation')
    lat_sizes = np.array([10, 10, 10, 8, 4, 15])

    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        dataset = h5py.File(path_to_data, 'r')
        self.imgs = dataset['images'][::subsample]
        self.lat_val = dataset['labels'][::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = self.imgs[idx] / 255
        if self.transform:
            sample = self.transform(sample)
        return sample, self.lat_val[idx]


class cars3dDataset(Dataset):
    """Cars3D dataloader class

    The data set was first used in the paper "Deep Visual Analogy-Making"
    (https://papers.nips.cc/paper/5845-deep-visual-analogy-making) and can be
    downloaded from http://www.scottreed.info/. The images are rescaled to 64x64.

    The ground-truth factors of variation are:
    0 - elevation (4 different values)
    1 - azimuth (24 different values)
    2 - object type (183 different values)

    Reference: Code adapted from
    https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/cars3d.py
    """
    lat_names = ('elevation', 'azimuth', 'object_type')
    lat_sizes = np.array([4, 24, 183])

    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        self.imgs = self._load_data()[::subsample]
        self.lat_val = cartesian([np.array(list(range(i))) for i in self.lat_sizes])[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if self.transform:
            sample = self.transform(self.imgs[idx])
        return sample.float(), self.lat_val[idx]

    def _load_data(self):
        dataset = np.zeros((24 * 4 * 183, 64, 64, 3))
        all_files = glob.glob("cars3d/data/cars/*.mat")
        for i, filename in enumerate(all_files):
            data_mesh = self._load_mesh(filename)
            factor1 = np.array(list(range(4)))
            factor2 = np.array(list(range(24)))
            all_factors = np.transpose([
                np.tile(factor1, len(factor2)),
                np.repeat(factor2, len(factor1)),
                np.tile(i,
                        len(factor1) * len(factor2))
            ])
            dataset[np.arange(i, 24*4*183, 183)] = data_mesh
        return dataset

    def _load_mesh(self, filename):
        """Parses a single source file and rescales contained images."""
        mesh = np.einsum("abcde->deabc", sio.loadmat(filename)["im"])
        flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
        rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3))
        for i in range(flattened_mesh.shape[0]):
            pic = PIL.Image.fromarray(flattened_mesh[i, :, :, :])
            pic.thumbnail((64, 64), PIL.Image.ANTIALIAS)
            rescaled_mesh[i, :, :, :] = np.array(pic)
        return rescaled_mesh * 1. / 255
