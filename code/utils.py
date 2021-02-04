import torch
import imageio
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
from skimage import img_as_ubyte
from tensorflow.keras.applications.inception_v3 import preprocess_input
import cv2
import glob
import os
import skimage.transform
import numpy as np
from numpy import cov, trace, iscomplexobj, asarray
from scipy.linalg import sqrtm
import scipy.misc
from skimage.transform import resize

rcParams['animation.convert_path'] = r'/usr/bin/convert'
rcParams['animation.ffmpeg_path'] = r'/usr/bin/ffmpeg'


class create_dirs:
    """ Creates directories for logging, Checkpoints and saving trained models """

    def __init__(self, name, ct):
        self.name = name
        self.ct = ct
        self.dircp = 'checkpoint.pth_{}.tar'.format(self.ct)
        self.dirout = '{}_Trained_rkm_{}.tar'.format(self.name, self.ct)

    def create(self):
        if not os.path.exists('cp/{}'.format(self.name)):
            os.makedirs('cp/{}'.format(self.name))

        if not os.path.exists('log/{}'.format(self.name)):
            os.makedirs('log/{}'.format(self.name))

        if not os.path.exists('out/{}'.format(self.name)):
            os.makedirs('out/{}'.format(self.name))

    def save_checkpoint(self, state, is_best):
        if is_best:
            torch.save(state, 'cp/{}/{}'.format(self.name, self.dircp))


def convert_to_imshow_format(image):
    # Convert from CHW to HWC
    if image.shape[0] == 1:
        return image[0, :, :]
    else:
        if np.any(np.where(image < 0)):
            # First convert back to [0,1] range from [-1,1] range
            image = image / 2 + 0.5
        return image.transpose(1, 2, 0)


def _get_traversal_range(max_traversal, mean=0, std=1):
    """Return the corresponding traversal range in absolute terms."""

    if max_traversal < 0.5:
        max_traversal = (1 - 2 * max_traversal) / 2  # from 0.45 to 0.05
        max_traversal = stats.norm.ppf(max_traversal, loc=mean, scale=std)  # from 0.05 to -1.645

    # Symmetrical traversals
    return (-1 * max_traversal, max_traversal)


class Lin_View(nn.Module):
    """ Unflatten linear layer to be used in Convolution layer"""

    def __init__(self, c, a, b):
        super(Lin_View, self).__init__()
        self.c, self.a, self.b = c, a, b

    def forward(self, x):
        try:
            return x.view(x.size(0), self.c, self.a, self.b)
        except:
            return x.view(1, self.c, self.a, self.b)


class Resize:
    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        if isinstance(size, int):
            self._size = (size, size)
        else:
            self._size = size

    def __call__(self, img: np.ndarray):
        resize_image = skimage.transform.resize(img, self._size)
        # the resize will return a float32 array
        return skimage.util.img_as_float32(resize_image)


def scatter_w_hist(h):
    """ 2D scatter plot of latent variables"""
    fig = plt.figure()
    grid = plt.GridSpec(4, 4, hspace=0.0, wspace=0.0)
    main_ax = fig.add_subplot(grid[:-1, 1:])
    y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
    x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

    main_ax.scatter(h[:, 0].detach().numpy(), h[:, 1].detach().numpy(), s=1)
    _, binsx, _ = x_hist.hist(h[:, 0].detach().numpy(), 40, histtype='stepfilled', density=True,
                              orientation='vertical')
    _, binsy, _ = y_hist.hist(h[:, 1].detach().numpy(), 40, histtype='stepfilled', density=True,
                              orientation='horizontal')
    x_hist.invert_yaxis()
    y_hist.invert_xaxis()
    plt.setp(main_ax.get_xticklabels(), visible=False)
    plt.setp(main_ax.get_yticklabels(), visible=False)
    plt.show()


class fid:
    """ Calculates the Frechet Inception Distance """

    def __init__(self, gt, gen):
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        self.gt = gt
        self.gen = gen
        self.model = InceptionV3(include_top=False, pooling='avg',
                                 input_shape=(299, 299, 3))  # prepare the inception v3 model

    # Scale an array of images to a new size
    def scale_images(self, images, new_shape):
        images_list = list()
        for image in images:
            # Resize with nearest neighbor interpolation
            new_image = resize(image, new_shape, 0)
            # Store
            images_list.append(new_image)
        return asarray(images_list)

    # Calculate frechet inception distance
    def calculate_fid(self, model, images1, images2):
        # Calculate activations
        act1 = model.predict(images1)
        act2 = model.predict(images2)
        # Calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)

        # Calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)

        # Calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))

        # Check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real

        # Calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def comp_fid(self):
        fid = np.empty(10)
        im_vae1 = []
        files = glob.glob("{}/*.png".format(self.gt))  # your image path
        for myFile in files:
            image = cv2.imread(myFile)
            im_vae1.append(image)

        im_vae1 = np.array(im_vae1, dtype='float32')
        images1 = im_vae1.astype('float32')
        images1 = self.scale_images(images1, (299, 299, 3))
        images1 = preprocess_input(images1)

        for i in range(10):
            im_vae2 = []
            files2 = glob.glob("{}/{}/*.png".format(self.gen, i))  # your image path
            for myFile in files2:
                image = cv2.imread(myFile)
                im_vae2.append(image)
            im_vae2 = np.array(im_vae2, dtype='float32')

            # convert integer to floating point values
            images2 = im_vae2.astype('float32')

            # resize images
            images2 = self.scale_images(images2, (299, 299, 3))
            print('Scaled', images1.shape, images2.shape)

            # pre-process images
            images2 = preprocess_input(images2)

            # calculate fid
            fid[i] = self.calculate_fid(self.model, images1, images2)
            print('FID: {}'.format(fid[i]))

        print('Mean: {}, Std: {}'.format(np.mean(fid), np.std(fid)))
        return fid


def gen_gt_imgs(dataset_name, xtrain):
    """ Save Ground-truth images on HDD for use in FID scores """
    if not os.path.exists('gt_imgs/{}'.format(dataset_name)):
        os.makedirs('gt_imgs/{}'.format(dataset_name))
        print('Saving Ground-Truth Images')
        for i, sample_batched in enumerate(xtrain):
            xt = sample_batched[0].numpy()
            for j in range(xt.shape[0]):
                imageio.imwrite('gt_imgs/{}/gt_img{}{}.png'.format(dataset_name, i, j),
                                  img_as_ubyte(convert_to_imshow_format(xt[j, :, :, :])))
            if i == 16:  # stop after printing 8k images
                break
        print('GT images saved in: gt_imgs/{}\n'.format(dataset_name))
