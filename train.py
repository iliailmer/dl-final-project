import os
from collections import defaultdict
from glob import glob

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from dataset import (
    TrainValData,
    TrainValDataSat,
    get_train_transforms,
    get_val_transforms,
)
from loss import IoULoss, MixedLoss, log_iou
from model import UNet11
from trainer import Config, Trainer
from utils import model_surgery, seed_everything

Optimizers: defaultdict = defaultdict(
    adam=torch.optim.Adam, adamw=torch.optim.AdamW
)
seed_everything(42)
PREFIX_train = "./train/"
PREFIX_train_masks = "./train_masks/"
PREFIX_csv = "./train_masks.csv"

train_pre = pd.read_csv(os.path.join(PREFIX_csv))
train_pre, valid_pre = train_test_split(train_pre, test_size=0.25)
valid_pre, test_pre = train_test_split(valid_pre, test_size=0.25)
print(
    (
        f"Data split:\n\ttrain: {len(train_pre)/(len(train_pre)+len(test_pre)+len(valid_pre))}"
        f"\n\tvalidation: {len(valid_pre)/(len(train_pre)+len(test_pre)+len(valid_pre))}"
        f"\n\ttest: {len(test_pre)/(len(train_pre)+len(test_pre)+len(valid_pre))}"
    )
)

pretrained_model = UNet11(pretrained=True, act="relu")

criterion_pre = MixedLoss(
    weight=0.75, loss_fn1=nn.BCELoss(), loss_fn2=IoULoss(loss=True)
)  # w*fn1+(1-w)*fn2
config_pre = Config(
    epochs=5,
    optim_config=defaultdict(float, lr=1e-3, weight_decay=1e-6),
    train_data_config=defaultdict(
        int,
        df=train_pre,
        batch_size=16,
        transforms=get_train_transforms(),
        drop_last=False,
    ),
    val_data_config=defaultdict(
        int,
        df=valid_pre,
        batch_size=16,
        transforms=get_val_transforms(),
        drop_last=False,
    ),
    test_data_config=defaultdict(
        int,
        df=test_pre,
        batch_size=1,
        transforms=get_val_transforms(),
        drop_last=False,
    ),
)

loaders_pre = dict(
    train=DataLoader(
        TrainValData(
            df=config_pre.train_data_config["df"],
            prefix=PREFIX_train,
            transforms=config_pre.train_data_config["transforms"],
        ),
        batch_size=config_pre.train_data_config["batch_size"],
        shuffle=True,
    ),
    valid=DataLoader(
        TrainValData(
            df=config_pre.val_data_config["df"],
            prefix=PREFIX_train,
            transforms=config_pre.val_data_config["transforms"],
        ),
        batch_size=config_pre.val_data_config["batch_size"],
        shuffle=False,
    ),
)
optimizer_pre = Optimizers["adam"](
    pretrained_model.parameters(),
    lr=config_pre.optim_config["lr"],
    weight_decay=config_pre.optim_config["weight_decay"],
)
scheduler_pre = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_pre,
    factor=0.5,
    patience=5,
)

im_dirs = glob("../input/inria180/training/training/images/*.png")
gt_dirs = glob("../input/inria180/training/training/gt/*.png")

inria = pd.DataFrame()
inria["img"] = im_dirs
inria["mask"] = gt_dirs

train, valid = train_test_split(inria, test_size=0.25)
valid, test = train_test_split(valid, test_size=0.25)
print(
    (
        f"Data split:\n\ttrain: {len(train)/(len(train)+len(test)+len(valid))}"
        f"\n\tvalidation: {len(valid)/(len(train)+len(test)+len(valid))}"
        f"\n\ttest: {len(test)/(len(train)+len(test)+len(valid))}"
    )
)

criterion = MixedLoss(weight=0.5, loss_fn1=log_iou, loss_fn2=nn.BCELoss())
config = Config(
    epochs=10,
    optim_config=defaultdict(float, lr=1e-3, weight_decay=1e-6),
    train_data_config=defaultdict(
        int,
        df=train,
        batch_size=16,
        transforms=get_train_transforms(),
        drop_last=False,
    ),
    val_data_config=defaultdict(
        int,
        df=valid,
        batch_size=16,
        transforms=get_val_transforms(),
        drop_last=False,
    ),
    test_data_config=None,
)

loaders = dict(
    train=DataLoader(
        TrainValDataSat(
            df=config.train_data_config["df"],
            transforms=config.train_data_config["transforms"],
        ),
        batch_size=config.train_data_config["batch_size"],
        shuffle=True,
    ),
    valid=DataLoader(
        TrainValDataSat(
            df=config.val_data_config["df"],
            transforms=config.val_data_config["transforms"],
        ),
        batch_size=config.val_data_config["batch_size"],
        shuffle=False,
    ),
)


if __name__ == "__main__":
    trainer = Trainer(
        model=pretrained_model,
        train_df=train_pre,
        test_df=None,
        val_df=valid_pre,
        config=config_pre,
        loaders=loaders_pre,
        optimizer=optimizer_pre,
        scheduler=scheduler_pre,
        loss=criterion,
        grad_acc=1,
        metrics=IoULoss(loss=False),
    )
    trainer.train("./pretrained_model")
    baseline_model = model_surgery(pretrained_model, "relu")
    swish_model = model_surgery(pretrained_model, "swish")
    resu_model = model_surgery(pretrained_model, "resu")
    optimizer = Optimizers["adamw"](
        baseline_model.parameters(),
        lr=config.optim_config["lr"],
        weight_decay=config.optim_config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=5,
    )

    trainer_baseline = Trainer(
        model=baseline_model,
        train_df=train,
        test_df=None,
        val_df=valid,
        config=config,
        loaders=loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=criterion,
        grad_acc=1,
        metrics=IoULoss(),
    )

    trainer_swish = Trainer(
        model=swish_model,
        train_df=train,
        test_df=None,
        val_df=valid,
        config=config,
        loaders=loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=criterion,
        grad_acc=8,
        metrics=IoULoss(),
    )

    trainer_resu = Trainer(
        model=resu_model,
        train_df=train,
        test_df=None,
        val_df=valid,
        config=config,
        loaders=loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=criterion,
        grad_acc=8,
        metrics=IoULoss(),
    )
    trainer_baseline.train("./baseline_model")
    trainer_swish.train("./swish_model")
    trainer_resu.train("./resu_model")
