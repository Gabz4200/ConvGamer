from __future__ import annotations

import torch
from torch import nn


class GeometryOp(nn.Module):
    """Base class for geometry ops (DIP — injected, not imported by models)."""

    def forward(self, position: torch.Tensor, velocity: torch.Tensor, dt: float) -> torch.Tensor:
        raise NotImplementedError


from .geometry_impls import build_geometry_op  # noqa: E402
from .references.geometry_reference import (  # noqa: E402
    reference_integrate_particles,
)


def integrate_particles(
    position: torch.Tensor,
    velocity: torch.Tensor,
    dt: float,
    op: GeometryOp | None = None,
) -> torch.Tensor:
    """Public entry point called by ``models/``.

    Dispatches to the injected ``op``.  If ``op`` is ``None`` uses the
    reference (pure PyTorch) implementation.
    """
    if op is None:
        return reference_integrate_particles(position, velocity, dt)
    return op(position, velocity, dt)


__all__ = ["GeometryOp", "integrate_particles", "build_geometry_op"]
