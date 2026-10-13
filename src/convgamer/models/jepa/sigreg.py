"""Sketched Isotropic Gaussian Regularization (LeJEPA, arXiv:2511.08544).

Faithful port of the LeJEPA MINIMAL.md ``SIGReg`` (~15 lines): random 1D
slices of the embedding distribution are tested for Gaussianity with the
Epps-Pulley empirical-characteristic-function statistic. Minimizing it
pressures embeddings toward an isotropic Gaussian — the distribution LeJEPA
proves minimizes downstream prediction risk — which resists representation
collapse without any teacher, stop-gradient, or scheduler.

Vendored (not ``pip install lejepa``) so training gains zero extra
dependencies and the exact math is pinned: the original hardcodes CUDA and
256 slices, while here device/dtype follow the input and the slice count is
configurable. Recommended trade-off ``sigreg_lambda=0.02`` is the paper's
goto value (prediction loss unscaled, regularizer added).
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["SIGReg"]


class SIGReg(nn.Module):
    """Epps-Pulley sketched Gaussianity statistic over embedding tokens.

    Args:
        num_slices: Random 1D projection directions redrawn every call (256).
        knots: Quadrature points for the ECF integral on ``[0, t_max]`` (17).
        t_max: Upper integration bound (3.0).
    """

    t: torch.Tensor
    phi: torch.Tensor
    weights: torch.Tensor

    def __init__(self, num_slices: int = 256, knots: int = 17, t_max: float = 3.0) -> None:
        if num_slices < 1:
            raise ValueError(f"num_slices must be >= 1, got {num_slices}")
        super().__init__()
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        self.num_slices = num_slices

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Gaussianity statistic: lower means closer to isotropic Gaussian.

        Args:
            tokens: Embedding tokens ``(N, D)`` with gradients attached.

        Returns:
            Nonnegative scalar; collapse/degenerate input scores high.
        """
        if tokens.ndim != 2:
            raise ValueError(f"SIGReg expects (N, D) tokens, got {tuple(tokens.shape)}")
        # fp32 statistics under AMP: half-precision input would fail the
        # matmul against fp32 slice directions, so cast up explicitly.
        directions = torch.randn(
            tokens.shape[-1], self.num_slices, device=tokens.device, dtype=torch.float32
        )
        directions = directions.div_(directions.norm(p=2, dim=0).clamp_min(1e-12))
        x_t = (tokens.float() @ directions).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * tokens.shape[-2]
        return statistic.mean()
