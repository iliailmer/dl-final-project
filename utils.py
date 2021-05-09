import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.io import imread
from torch import nn

from model import DecoderBlock
from resu import ReSU

Swish = nn.SiLU
Activations: defaultdict = defaultdict(nn.Module)
Activations["relu"] = nn.ReLU(inplace=True)
Activations["swish"] = Swish()
Activations["resu"] = ReSU()


def rle_decode(mask_rle, shape=(1280, 1918, 1)):
    """
    from Kaggle kernel:
    https://www.kaggle.com/julichitai/gb-carvana-cars-segmentation
    """
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [
        np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])
    ]
    starts -= 1
    ends = starts + lengths
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    img = img.reshape(shape)
    return img


def plot_images_carvana(nrows, image_paths, df=None):
    fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 9))
    for i in range(nrows):
        idx = np.random.randint(0, (len(image_paths)))
        ax[i, 0].imshow(imread(image_paths[idx]))
        ax[i, 0].grid(False)
        ax[i, 0].axis(False)
        ax[i, 0].set_title("Image")
        ax[i, 1].imshow(rle_decode(df.iloc[idx, 1]))
        ax[i, 1].grid(False)
        ax[i, 1].axis(False)
        ax[i, 1].set_title("Mask")
    plt.tight_layout()


def model_surgery(model, act: str):
    for name, layer in model.named_children():
        if isinstance(layer, DecoderBlock):
            layer.block[0].activation = Activations[act]
            layer.block[2] = Activations[act]
    return model


def plot_images(nrows, image_paths, mask_paths, df=None):
    fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 9))
    for i in range(nrows):
        idx = np.random.randint(0, (len(image_paths)))
        ax[i, 0].imshow(imread(image_paths[idx]))
        ax[i, 0].grid(False)
        ax[i, 0].axis(False)
        ax[i, 0].set_title("Image")
        ax[i, 1].imshow(imread(mask_paths[idx]))
        ax[i, 1].grid(False)
        ax[i, 1].axis(False)
        ax[i, 1].set_title("Mask")
    plt.tight_layout()


def compare_pred_gt(image, mask, prediction):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    for idx, (k, v) in enumerate(
        dict(Image=image, Mask=mask, Prediction=prediction).items()
    ):
        ax[idx].imshow(v)
        ax[idx].grid(False)
        ax[idx].axis(False)
        ax[idx].set_title(k)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
