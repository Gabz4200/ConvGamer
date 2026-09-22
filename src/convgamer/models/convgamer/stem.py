from __future__ import annotations

import torch
import torch.nn as nn

from convgamer.models.io import StemState

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

    def init_state(
        self,
        batch_size: int,
        h: int,
        w: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> StemState:
        """Fresh streaming state: one past-frame window per branch conv.

        ``h``/``w`` are the spatial dims the stem sees (post-downsample).
        """
        return StemState(
            near=self.near_conv.init_state(batch_size, h, w, device=device, dtype=dtype),
            local=self.local_conv.init_state(batch_size, h, w, device=device, dtype=dtype),
            far=self.far_conv.init_state(batch_size, h, w, device=device, dtype=dtype),
            fuse=self.fuse_conv.init_state(batch_size, h, w, device=device, dtype=dtype),
        )

    def step(self, x_t: torch.Tensor, state: StemState) -> tuple[torch.Tensor, StemState]:
        """Stream one frame ``(B, C, 1, H, W)``; returns ``(out_t, next_state)``."""
        near, near_state = self.near_conv.step(x_t, state.near)
        local, local_state = self.local_conv.step(x_t, state.local)
        far, far_state = self.far_conv.step(x_t, state.far)
        new = torch.cat((near, local, far), dim=1)
        new = self.act(new)
        new, fuse_state = self.fuse_conv.step(new, state.fuse)
        x = torch.cat((x_t, new), dim=1)
        if self.use_softmax:
            x = torch.cat((x, spatial_softmax(x)), dim=1)
        return self.norm(x), StemState(
            near=near_state, local=local_state, far=far_state, fuse=fuse_state
        )
