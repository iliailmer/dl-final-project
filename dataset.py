import os

import albumentations as albu
import pandas as pd
from albumentations import (
    Compose,
    HorizontalFlip,
    Normalize,
    Resize,
    Rotate,
    VerticalFlip,
)
from albumentations.pytorch import ToTensorV2
from skimage.io import imread
from torch.utils.data import Dataset

from utils import rle_decode


class TrainValData(Dataset):
    def __init__(self, df: pd.DataFrame, prefix: str, transforms=None):
        super().__init__()
        self.tfm = transforms
        self.df = df
        self.prefix = prefix

    def __getitem__(self, idx):
        path, mask = self.df.iloc[idx]
        image = imread(os.path.join(self.prefix, path))
        mask = rle_decode(mask_rle=mask)
        if self.tfm:
            aug = self.tfm(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]
        return dict(
            image=image, mask=mask.reshape((1, mask.shape[0], mask.shape[1]))
        )

    def __len__(self):
        return len(self.df)


class TrainValDataSat(Dataset):
    def __init__(self, df: pd.DataFrame, transforms=None):
        super().__init__()
        self.tfm = transforms
        self.df = df

    def __getitem__(self, idx):
        img_path, mask_path = self.df.iloc[idx]
        image = imread(img_path)
        mask = imread(mask_path)[..., 0]  # we need single channel for mask
        mask = mask / 255
        if self.tfm:
            aug = self.tfm(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]
        return dict(
            image=image, mask=mask.reshape((1, mask.shape[0], mask.shape[1]))
        )

    def __len__(self):
        return len(self.df)


def get_train_transforms(p=0.5):
    return Compose(
        [
            Resize(64, 64),
            albu.RandomRotate90(p=p),
            Rotate(45, p=p),
            HorizontalFlip(p=p),
            VerticalFlip(p=p),
            Normalize(always_apply=True),
            ToTensorV2(always_apply=True),
        ]
    )


def get_val_transforms(p=0.5):
    return Compose(
        [
            Resize(64, 64),
            Normalize(always_apply=True),
            ToTensorV2(always_apply=True),
        ]
    )
