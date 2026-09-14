"""Op boundary protocols for injected callables.

Structural types only: plain functions already conform without inheriting.
Current conformers: ``uniform_temporal_subsample`` and ``spatial_softmax``
in ``models.convgamer.blocks``, and the ``srgb_*``/``oklab_*`` converters
in ``data.oklab`` (the upcoming dataset color pipeline consumes them as
``ForwardOp`` callables).
"""

from __future__ import annotations

from typing import Protocol

import torch


class ForwardOp(Protocol):
    """ISP: a stateless forward-only op.

    Used as the boundary type for ops injected into models: any callable
    mapping tensors to a tensor satisfies it without explicit inheritance.
    """

    def __call__(self, *args, **kwargs) -> torch.Tensor: ...


class DifferentiableOp(Protocol):
    """ISP: an op that participates in autograd."""

    def forward(self, *args, **kwargs) -> torch.Tensor: ...

    def backward(self, *grad_outputs) -> tuple[torch.Tensor, ...]: ...


class ProfilerOp(Protocol):
    """ISP: an op that can report timing information."""

    def profile(self, *args, **kwargs) -> float: ...


__all__ = ["ForwardOp", "DifferentiableOp", "ProfilerOp"]
