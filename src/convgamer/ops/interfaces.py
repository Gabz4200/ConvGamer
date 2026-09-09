from __future__ import annotations

from typing import Protocol

import torch


class ForwardOp(Protocol):
    """ISP: a stateless forward-only op."""

    def __call__(self, *args, **kwargs) -> torch.Tensor: ...


class DifferentiableOp(Protocol):
    """ISP: an op that participates in autograd."""

    def forward(self, *args, **kwargs) -> torch.Tensor: ...

    def backward(self, *grad_outputs) -> tuple[torch.Tensor, ...]: ...


class ProfilerOp(Protocol):
    """ISP: an op that can report timing information."""

    def profile(self, *args, **kwargs) -> float: ...


__all__ = ["ForwardOp", "DifferentiableOp", "ProfilerOp"]
