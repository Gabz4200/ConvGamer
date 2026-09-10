from __future__ import annotations

import torch

from ..kernels.taichi.runtime import ensure_initialized
from .geometry import GeometryOp
from .references.geometry_reference import reference_integrate_particles


class ReferenceGeometryOp(GeometryOp):
    """Pure-PyTorch geometry op used for parity tests."""

    def forward(self, position: torch.Tensor, velocity: torch.Tensor, dt: float) -> torch.Tensor:
        return reference_integrate_particles(position, velocity, dt)


class TaichiGeometryOp(GeometryOp):
    """Taichi-backed geometry op."""

    def __init__(self):
        ensure_initialized()

    def forward(self, position: torch.Tensor, velocity: torch.Tensor, dt: float) -> torch.Tensor:
        from ..kernels.taichi.geometry import taichi_integrate_particles

        return taichi_integrate_particles(position, velocity, dt)


def build_geometry_op(backend: str) -> GeometryOp:
    try:
        cls = {
            "reference": ReferenceGeometryOp,
            "taichi": TaichiGeometryOp,
        }[backend]
    except KeyError:
        raise ValueError(f"Unknown backend '{backend}'. Valid: reference, taichi") from None
    return cls()
