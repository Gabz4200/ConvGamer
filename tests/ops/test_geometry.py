"""Reference vs Taichi geometry op parity (Section 3.2, paper Table 1)."""

import torch

from convgamer.ops.geometry import integrate_particles
from convgamer.ops.references.geometry_reference import reference_integrate_particles


def test_reference_parity():
    position = torch.randn(4, 8, 32, 32)
    velocity = torch.randn_like(position)
    dt = 0.01

    ref = reference_integrate_particles(position, velocity, dt)
    op = integrate_particles(position, velocity, dt)
    torch.testing.assert_close(op, ref)
