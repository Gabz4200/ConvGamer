"""Video encoder: causal downsampler stem, per-frame InceptionNeXt, causal head.

Input (B, C, T, H, W) -> logits (B, num_classes). Fully causal: no future
frame influences any past output. Output for frame ``t`` pools only frames
``<= t`` (or returns the full per-frame sequence when ``return_sequence``).
"""

from __future__ import annotations

import einops
import torch
from torch import nn

from convgamer.models.base import BaseModel
from convgamer.models.convgamer.blocks import (
    CausalTemporalMixer,
    ConvGamerStem,
    LearnedSpatialTemporalDownsampler,
)
from convgamer.models.inception_next.encoder import InceptionNeXtEncoder
from convgamer.models.registry import register_model


@register_model("ConvGamerEncoder")
class ConvGamerEncoder(BaseModel):
    """Video encoder: downsampler stem, per-frame maps, causal temporal mix.

    Per-frame path uses ``forward_feature_map`` (spatial maps, never the
    classification head): maps are stacked to (B, F, T, H, W), mixed
    causally across T at full resolution, then spatially pooled.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 96,
        num_layers: int | tuple[int, ...] = 3,
        num_classes: int = 1000,
        layer_scale_init: float = 1e-6,
        mlp_ratios: tuple[int, int, int, int] = (4, 4, 4, 3),
        out_factor: int = 2,
        use_softmax: bool = False,
        target_size: tuple[int, int] | int | None = None,
        temporal_dilations: tuple[int, ...] = (1, 2, 4),
    ):
        super().__init__()

        downsampler = LearnedSpatialTemporalDownsampler(
            in_channels=input_dim, out_factor=out_factor, target_size=target_size
        )
        stem = ConvGamerStem(
            in_channels=downsampler.out_channels, use_softmax=use_softmax, use_norm=False
        )
        self.stem = nn.Sequential(downsampler, stem)
        self.frame_encoder = InceptionNeXtEncoder(
            input_dim=stem.out_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=0,
            layer_scale_init=layer_scale_init,
            mlp_ratios=mlp_ratios,
        )
        self.temporal_mix = CausalTemporalMixer(
            channels=self.frame_encoder.feature_dim, dilations=temporal_dilations
        )
        self.norm = nn.LayerNorm(self.frame_encoder.feature_dim)
        self.head = (
            nn.Linear(self.frame_encoder.feature_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def _video_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        b, _c, t, _h, _w = x.shape
        frames = einops.rearrange(x, "b c t h w -> (b t) c h w")
        maps = self.frame_encoder.forward_feature_map(frames)
        _, _, h, w = maps.shape
        video = einops.rearrange(maps, "(b t) f h w -> b f t h w", b=b, t=t, h=h, w=w)
        mixed = self.temporal_mix(video)
        return mixed.mean(dim=[3, 4])

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """Encode video to logits, causally pooled over past frames only."""
        video = self._video_features(x)
        if return_sequence:
            b, _, t = video.shape
            flat = einops.rearrange(video, "b f t -> (b t) f")
            logits = self.head(self.norm(flat))
            return einops.rearrange(logits, "(b t) k -> b t k", b=b, t=t)
        pooled = torch.cumsum(video, dim=2) / torch.arange(
            1, video.shape[2] + 1, device=video.device, dtype=video.dtype
        )
        return self.head(self.norm(pooled[:, :, -1]))
