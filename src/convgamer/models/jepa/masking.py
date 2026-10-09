"""Shared token-mask utilities for the JEPA predictor and loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["resize_token_mask"]


def resize_token_mask(mask: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    """Nearest-resize a boolean ``(B, T, H, W)`` token mask to ``size``.

    Masks mark discrete tokens, not a smooth field, so nearest interpolation
    keeps binary boundaries intact. Returns a boolean tensor.
    """
    if mask.ndim != 4:
        raise ValueError(f"Expected 4D mask (B, T, H, W), got {mask.ndim}D")
    if mask.dtype != torch.bool:
        raise ValueError(f"Expected bool mask, got {mask.dtype}")
    return F.interpolate(mask.float().unsqueeze(1), size=size, mode="nearest").bool().squeeze(1)
