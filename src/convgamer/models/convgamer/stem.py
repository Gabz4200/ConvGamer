from __future__ import annotations

import torch
import torch.nn as nn

from .causal import CausalConv3d, CausalLayerNorm
from .downsampler import spatial_softmax


class ConvGamerStem(nn.Module):
    """Multi-scale 3D convolution stem.

    Input: (B, C, T, H, W) -> Output: (B, 4*C, T, H, W) by default.

    Three parallel branches (1x1x1 local, 3x3x3 near, 7x7x7 grouped far)
    are concatenated, fused with a 1x1x1 conv, then concatenated with the
    residual input and normalized. Resolution is preserved; downsampling is
    expected to happen before this module. ``use_softmax`` doubles the
    output (raw + spatial distribution) and is off by default: enable only
    as an explicit ablation.
    """

    def __init__(self, in_channels: int = 24, use_softmax: bool = False, use_norm=False) -> None:
        super().__init__()
        branch_channels = in_channels * 2
        fused_channels = in_channels * 3

        self.in_channels = in_channels
        self.use_softmax = use_softmax
        base_channels = in_channels + fused_channels
        self.out_channels = base_channels * 2 if use_softmax else base_channels

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

        # The image with spatial_softmax can be interpreted as a
        # distribution of the features across the spatial dimensions.
        # For example, if it was RGB image, then, for the Red channel,
        # we would get "how much of the red in the image is in this position?"
        if self.use_softmax:
            x = torch.cat((x, spatial_softmax(x)), dim=1)

        return self.norm(x)

    def reset_cache(
        self,
        batch_size: int,
        h: int,
        w: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Reset temporal cache for all conv layers in the stem."""
        for conv in [self.near_conv, self.far_conv, self.local_conv, self.fuse_conv]:
            conv.reset_cache(batch_size, h, w, device=device, dtype=dtype)

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        """Stream one frame ``(B, C, 1, H, W)`` through the stem."""
        near = self.near_conv.step(x_t)
        local = self.local_conv.step(x_t)
        far = self.far_conv.step(x_t)
        new = torch.cat((near, local, far), dim=1)
        new = self.act(new)
        new = self.fuse_conv.step(new)
        x = torch.cat((x_t, new), dim=1)
        if self.use_softmax:
            x = torch.cat((x, spatial_softmax(x)), dim=1)
        return self.norm(x)
