"""Shared utilities for multi-head (per-modality) JEPA prediction.

V-JEPA 2.1 (§2.3.2) attaches one prediction head per encoder level, each head
consuming a distinct modality stream. This module provides a single helper
that both :class:`~convgamer.models.jepa.predictor.VJEPAPredictor` and
:class:`~convgamer.models.jepa.loss.JEPALoss` share when they need to apply a
convolution to a batch × head tensor of feature maps::

    (B, H, F, T, H, W) -> conv across (B*H, ...)

Collapsing and restoring the batch/head axes here avoids every caller writing
the same ``reshape`` dance — and, more importantly, keeps the two modules
guaranteed consistent: any change to the folding convention lives in one place.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["apply_across_heads"]


def apply_across_heads(
    module: nn.Module,
    x: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Apply ``module`` to each head's feature maps independently.

    The feature tensor is interpreted as ``(B, H, F, T, H, W)`` where ``H`` is
    the head/modality axis. The head axis is folded into the batch axis so
    ``module`` sees a conventional ``(B*H, F, T, H, W)`` layout, then the
    head axis is restored, yielding ``(B, H, F', T, H, W)``.

    Args:
        module: A module accepting and returning 5D ``(N, C, T, H, W)`` input.
        x: Input tensor of shape ``(B, H, F, T, H, W)``.
        num_heads: Size of the head axis (``x.shape[1]``).

    Returns:
        Tensor of shape ``(B, H, ...)`` with the module applied per head.
    """
    if x.ndim != 6:
        raise ValueError(
            f"Expected (B, H, F, T, H, W) input (6D), got {x.ndim}D with shape {tuple(x.shape)}"
        )
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    b, h, *_ = x.shape
    if h != num_heads:
        raise ValueError(f"Head axis size {h} does not match num_heads={num_heads}")
    x_flat = x.reshape(b * h, *x.shape[2:])
    out_flat = module(x_flat)
    out = out_flat.reshape(b, h, *out_flat.shape[1:])
    return out
