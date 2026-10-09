from __future__ import annotations

import torch
import torch.nn as nn

from .causal import CausalConv3d, CausalLayerNorm


class ConvGamerStem(nn.Module):
    """Multi-scale 3D convolution stem.

    Input: (B, C, T, H, W) -> Output: (B, 4*C, T, H, W) by default.

    Three parallel branches (1x1x1 local, 3x3x3 near, 7x7x7 grouped far)
    are concatenated, fused with a 1x1x1 conv, then concatenated with the
    residual input and normalized. Resolution is preserved; downsampling is
    expected to happen before this module.
    """

    def __init__(self, in_channels: int = 24, use_norm: bool = False) -> None:
        super().__init__()
        branch_channels = in_channels * 2
        fused_channels = in_channels * 3

        self.in_channels = in_channels
        self.out_channels = in_channels + fused_channels

        self.near_conv = CausalConv3d(
            in_channels=in_channels,
            out_channels=branch_channels,
            kernel_size=3,
        )
        self.local_conv = CausalConv3d(
            in_channels=in_channels,
            out_channels=branch_channels,
            kernel_size=1,
        )
        self.far_conv = CausalConv3d(
            in_channels=in_channels,
            out_channels=branch_channels,
            kernel_size=7,
            groups=in_channels,
        )
        self.act = nn.GELU()
        self.fuse_conv = CausalConv3d(
            in_channels=branch_channels * 3,
            out_channels=fused_channels,
            kernel_size=1,
        )
        self.norm = CausalLayerNorm(self.out_channels) if use_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        near = self.near_conv(x)
        local = self.local_conv(x)
        far = self.far_conv(x)
        new = torch.cat((near, local, far), dim=1)
        new = self.act(new)
        new = self.fuse_conv(new)
        x = torch.cat((x, new), dim=1)
        return self.norm(x)
