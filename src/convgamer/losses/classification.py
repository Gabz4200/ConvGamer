from __future__ import annotations

import torch.nn as nn


class ClassificationLoss(nn.Module):
    """Loss wrapper selectable via config."""

    def __init__(self, loss_type: str = "cross_entropy"):
        super().__init__()
        if loss_type == "cross_entropy":
            self._loss = nn.CrossEntropyLoss()
        elif loss_type == "mse":
            self._loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss_type '{loss_type}'")

    def forward(self, pred, target):
        return self._loss(pred, target)
