"""Plain metric helpers used across the training loop."""

from __future__ import annotations

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == targets).float().mean().item())
