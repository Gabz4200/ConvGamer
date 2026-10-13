from __future__ import annotations

import torch
import torch.nn as nn

from .causal import CausalConv3d, CausalLayerNorm


class LearnedSpatialTemporalDownsampler(nn.Module):
    """Patchify-first spatial downsampler with low-res residual correction.

    Pipeline per clip ``(B, C, T, H, W)``:

    1. ``patch_embed``: strided ``(1, P, P)`` conv (temporal extent 1, so it is
       stateless in time and trivially streaming-safe) downsamples space by
       ``patch_size`` while expanding channels ``in_channels -> in_channels *
       out_factor``. Aspect ratio is preserved — no forced square.
    2. ``correction_conv``: small causal ``(3, K, K)`` conv at low resolution
       (``K=7`` depthwise or ``K=4`` dense, see ``correction_kernel``).
    3. ``out_correction_conv``: ``1x1x1`` projection back to base width,
       added residually, then normalized.

    Temporal dim is always preserved; only space is downsampled.
    """

    def __init__(
        self,
        in_channels: int = 3,
        expansion: int = 12,
        out_factor: int = 2,
        patch_size: int = 8,
        correction_kernel: str = "dense4",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        if out_factor <= 0:
            raise ValueError(f"out_factor must be positive, got {out_factor}")
        if expansion <= 0:
            raise ValueError(f"expansion must be positive, got {expansion}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        base_channels = in_channels * out_factor
        self.out_channels = base_channels
        intermediate_channels = base_channels * expansion
        self.intermediate_channels = intermediate_channels

        self.patch_embed = CausalConv3d(
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )
        if correction_kernel == "dw7":
            self.correction_conv = CausalConv3d(
                in_channels=base_channels,
                out_channels=intermediate_channels,
                kernel_size=(3, 7, 7),
                groups=base_channels,
            )
        elif correction_kernel == "dense4":
            self.correction_conv = CausalConv3d(
                in_channels=base_channels,
                out_channels=intermediate_channels,
                kernel_size=(3, 4, 4),
            )
        else:
            raise ValueError(
                f"correction_kernel must be 'dw7' or 'dense4', got {correction_kernel!r}"
            )
        self.act = nn.GELU()

        self.out_correction_conv = CausalConv3d(
            in_channels=intermediate_channels,
            out_channels=base_channels,
            kernel_size=1,
        )

        self.norm = CausalLayerNorm(self.out_channels)

    def resolve_spatial_size(self, h: int, w: int) -> tuple[int, int]:
        """Output spatial dims of the strided patch embed for an ``h, w`` input."""
        p = self.patch_size
        return (max(1, (h + p - 1) // p), max(1, (w + p - 1) // p))

    @property
    def history_frames(self) -> int:
        """Past frames needed alongside the current one for streaming."""
        return self.correction_conv.temporal_extent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W) — temporal dim preserved, spatial downsampled only
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (B, C, T, H, W), got shape {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {x.shape[1]}")

        base = self.patch_embed(x)

        # Correction branch runs at low resolution on the patchified features.
        # Temporal context comes from the 3D conv's causal padding; T preserved.
        correct = self.correction_conv(base)
        correct = self.act(correct)
        correct = self.out_correction_conv(correct)

        out = base + correct
        out = self.norm(out)
        return out
