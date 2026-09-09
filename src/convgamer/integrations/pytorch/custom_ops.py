"""Skeleton — structure only, user adds fake-tensor metadata + autograd."""

from __future__ import annotations

import torch
from torch.library import custom_op


@custom_op("convgamer::integrate_particles", mutates_args=())
def integrate_particles_op(
    position: torch.Tensor,
    velocity: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    from convgamer.ops.geometry_impls import build_geometry_op

    op = build_geometry_op("taichi")
    return op(position, velocity, dt)
