import copy
import gc
from collections import defaultdict
from typing import DefaultDict, NamedTuple

import pandas as pd
import torch
from tqdm.auto import tqdm

from model import UNet11


class Config(NamedTuple):
    epochs: int
    optim_config: DefaultDict
    train_data_config: DefaultDict
    val_data_config: DefaultDict
    test_data_config: DefaultDict


class Trainer:
    def __init__(
        self,
        model: UNet11,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_df: pd.DataFrame,
        config: Config,
        loaders: dict,
        metrics: dict,
        optimizer,
        scheduler,
        loss,
        grad_acc: int = 1,
        train_loader=None,
        val_loader=None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epochs = config.epochs
        self.model = model.to(self.device)
        self.train_loader = loaders.get("train", None)
        self.val_loader = loaders.get("valid", None)
        self.test_loader = loaders.get("test", None)
        self.loss_fn = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_losses: list = []
        self.val_losses: list = []
        self.val_metrics: defaultdict = defaultdict(list)
        self.train_metrics: defaultdict = defaultdict(list)
        self.grad_accumulation_steps = grad_acc

    def train(self, PATH: str):
        for epoch in range(self.epochs):
            gc.collect()
            total_loss = 0.0
            total_iou = 0.0
            best_iou = 0.0
            n = 0
            self.model.train()
            pbar = tqdm(self.train_loader)
            pbar.set_description(f"Train: {epoch+1}/{self.epochs}")
            for idx, batch in enumerate(pbar):
                # set running trainig metrics
                running_iou = 0.0
                # model step
                out = self.model(batch["image"].to(self.device))
                loss = self.loss_fn(
                    input=out, target=batch["mask"].to(self.device).float()
                )
                total_loss += loss.item()

                loss.backward()
                # optimizer step
                if idx % self.grad_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                # get batch metrics
                running_iou = self.metrics(
                    target=batch["mask"].float(), input=out.detach().cpu()
                )
                # collect total metrics
                total_iou += running_iou
                n += 1
                pbar.set_postfix_str(
                    (
                        f"train_loss={loss.item():.4f},"
                        f"iou={running_iou.item():.4f}"  # noqa
                    )
                )
            total_loss /= n
            total_iou /= n
            self.train_losses.append(total_loss)
            self.train_metrics["iou"].append(total_iou)
            total_loss = 0.0
            total_iou = 0.0
            n = 0
            pbar = tqdm(self.val_loader)
            with torch.no_grad():
                self.model.eval()
                pbar.set_description(f"Valid: {epoch+1}/{self.epochs}")
                for idx, batch in enumerate(pbar):
                    # model step
                    out = self.model(batch["image"].to(self.device))
                    loss = self.loss_fn(
                        input=out, target=batch["mask"].to(self.device).float()
                    )
                    # batch metrics
                    running_iou = self.metrics(
                        target=batch["mask"].float(), input=out.detach().cpu()
                    )
                    # collect total metrics
                    total_loss += loss.item()
                    total_iou += running_iou
                    n += 1
                    pbar.set_postfix_str(
                        (
                            f"val_loss={loss.item():.2f},"
                            f"iou={running_iou:.4f}"
                        )
                    )
            total_loss /= n
            total_iou /= n

            if total_iou > best_iou:
                best_iou = total_iou
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": copy.deepcopy(self.model)
                        .cpu()
                        .state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dic": self.scheduler.state_dict(),
                        "loss": self.loss_fn,
                    },
                    PATH,
                )

            self.scheduler.step(total_loss)
            self.val_losses.append(total_loss)
            self.val_metrics["iou"].append(total_iou)
            total_loss = 0.0
            n = 0
            print((f"Training metrics:\tloss: {self.train_losses[-1]:.2f},"))
            print((f"Validation metrics:\tloss: {self.val_losses[-1]:.2f},"))
