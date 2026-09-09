from typing import Any

import taichi as ti
import torch

from .runtime import ensure_initialized


@ti.kernel
def _integrate_particles(
    pos: Any,
    vel: Any,
    out: Any,
    dt: float,
):
    for i in ti.grouped(pos):
        out[i] = pos[i] + vel[i] * dt


def taichi_integrate_particles(
    position: torch.Tensor,
    velocity: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    ensure_initialized()
    out = torch.empty_like(position)
    _integrate_particles(position, velocity, out, dt)
    return out
