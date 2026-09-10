"""Building blocks for InceptionNeXt.

Re-implements the "Inception depthwise convolution" (Algorithm 1 in
arXiv:2303.16900) and the MetaNeXt block that uses it as the token mixer.
"""

from __future__ import annotations

import torch
from torch import nn


class InceptionDWConv2d(nn.Module):
    """Decomposed depthwise convolution from InceptionNeXt.

    Splits the input channels into four groups:
      * 1/8  -> 3x3 square kernel
      * 1/8  -> 1 x band (horizontal band)
      * 1/8  -> band x 1 (vertical band)
      * 5/8  -> identity passthrough

    This mirrors the PyTorch code in the paper's Algorithm 1.  ``band_kernel_size``
    must be odd so that integer padding is symmetric.
    """

    def __init__(
        self,
        in_channels: int,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,
        branch_ratio: float = 1 / 8,
    ):
        super().__init__()
        if not 0 < branch_ratio < 1:
            raise ValueError(f"branch_ratio must be in (0,1), got {branch_ratio}")
        if band_kernel_size % 2 == 0:
            raise ValueError(f"band_kernel_size must be odd, got {band_kernel_size}")
        gc = int(in_channels * branch_ratio)  # channels per conv branch
        if gc <= 0:
            raise ValueError(
                f"in_channels={in_channels} with branch_ratio={branch_ratio} gives gc=0; "
                "need in_channels >= 8 for default 1/8 ratio"
            )
        if in_channels - 3 * gc <= 0:
            raise ValueError(f"in_channels={in_channels} too small for 3 branches of gc={gc}")
        self.split_indexes = (gc, gc, gc, in_channels - 3 * gc)

        self.dwconv_hw = nn.Conv2d(
            gc,
            gc,
            square_kernel_size,
            padding=square_kernel_size // 2,
            groups=gc,
        )
        self.dwconv_w = nn.Conv2d(
            gc,
            gc,
            kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2),
            groups=gc,
        )
        self.dwconv_h = nn.Conv2d(
            gc,
            gc,
            kernel_size=(band_kernel_size, 1),
            padding=(band_kernel_size // 2, 0),
            groups=gc,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hw, x_w, x_h, x_id = torch.split(x, list(self.split_indexes), dim=1)
        return torch.cat(
            (
                self.dwconv_hw(x_hw),
                self.dwconv_w(x_w),
                self.dwconv_h(x_h),
                x_id,
            ),
            dim=1,
        )


class InceptionNeXtBlock(nn.Module):
    """MetaNeXt block whose token mixer is InceptionDWConv2d.

    TokenMixer -> Norm -> MLP -> + shortcut  (Section 3.1, Eq. 3 of the paper).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        self.token_mixer = InceptionDWConv2d(in_channels)
        self.norm = nn.LayerNorm(in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_channels),
        )
        self.gamma = (
            nn.Parameter(layer_scale_init * torch.ones(in_channels))
            if layer_scale_init > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.token_mixer(x)
        b, c, h, w = x.shape
        # LayerNorm + MLP over channel dim: (B,C,H,W) -> (B,H*W,C) -> MLP -> back
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.mlp(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x
        return x + residual
