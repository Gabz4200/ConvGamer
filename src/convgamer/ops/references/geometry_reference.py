from __future__ import annotations

import torch


def reference_integrate_particles(
    position: torch.Tensor,
    velocity: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Plain PyTorch reference — used by tests and as the default backend."""
    return position + velocity * dt
