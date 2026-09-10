from __future__ import annotations

import os
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
    if not position.is_contiguous() or not velocity.is_contiguous():
        raise ValueError("Taichi kernel requires contiguous tensors")
    if position.shape != velocity.shape:
        raise ValueError(
            f"position and velocity must have same shape, got {position.shape} vs {velocity.shape}"
        )
    arch = os.getenv("CONVGAMER_TAICHI_ARCH", "cpu")
    is_cpu_tensor = str(position.device) == "cpu" or position.device.type == "cpu"
    if arch == "cpu" and not is_cpu_tensor:
        raise RuntimeError(
            f"Taichi arch is 'cpu' but tensor on {position.device}; "
            "set CONVGAMER_TAICHI_ARCH accordingly"
        )
    out = torch.empty_like(position)
    _integrate_particles(position, velocity, out, dt)
    return out
