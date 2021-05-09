import torch
from torch import nn


class IoULoss(nn.Module):
    def __init__(self, eps=1e-5, loss=False):
        super().__init__()
        self.eps = eps
        self.loss = loss

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        target = target.view(-1)
        input = input.view(-1)

        intersection = input * target  # .sum()
        union = input + target  # .sum()
        union -= (
            intersection  # we count intersection portion twice in the union
        )

        iou = (intersection + self.eps) / (union + self.eps)
        iou = iou.mean()
        if self.loss:
            return 1 - iou
        else:
            return iou


def log_iou(
    input: torch.Tensor, target: torch.Tensor, eps=1e-5
) -> torch.Tensor:
    target = target.view(-1)
    input = input.view(-1)

    intersection = input * target  # .sum()
    union = input + target  # .sum()
    union -= intersection  # we count intersection portion twice in the union

    iou = (intersection + eps) / (union + eps)

    return -torch.log(iou.mean() + eps)


class LogIoULoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return log_iou(input=input, target=target, eps=self.eps)


class MixedLoss(nn.Module):
    def __init__(self, weight: float, loss_fn1, loss_fn2):
        super().__init__()
        assert 0 < weight < 1, f"Weight must be between 0, 1,  got {weight}"

        self.f1 = loss_fn1
        self.f2 = loss_fn2
        self.w = weight

    def forward(self, input, target):
        f1 = self.f1(input=input, target=target)
        f2 = self.f2(input=input, target=target)
        return self.w * f1 + (1 - self.w) * f2
