"""Video encoder: causal downsampler stem, per-frame InceptionNeXt, causal head.

Input (B, C, T, H, W) -> logits (B, num_classes). Fully causal: no future
frame influences any past output. Output for frame ``t`` pools only frames
``<= t`` (or returns the full per-frame sequence when ``return_sequence``).
"""

from __future__ import annotations

import warnings

import einops
import torch
from torch import nn

from convgamer.models.convgamer.causal import CausalLayerNorm, CausalTemporalMixer
from convgamer.models.convgamer.downsampler import LearnedSpatialTemporalDownsampler
from convgamer.models.convgamer.stem import ConvGamerStem
from convgamer.models.inception_next.encoder import InceptionNeXtEncoder
from convgamer.models.io import StepOutput, StreamingState
from convgamer.models.registry import register_model


@register_model("ConvGamerEncoder")
class ConvGamerEncoder(nn.Module):
    """Video encoder: downsampler stem, per-frame InceptionNeXt, causal head.

    Per-frame path uses ``forward_feature_map`` (spatial maps, never the
    classification head): maps are stacked to (B, F, T, H, W), mixed
    causally across T at full resolution, then spatially pooled.

    ``step`` mirrors that pipeline one frame at a time with all history carried
    in an explicit :class:`~convgamer.models.io.StreamingState`.

    As a foundation model for JEPA pre-training this encoder exposes
    ``forward_features`` as its public seam; the classification ``head``
    is retained only for downstream fine-tuning.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 96,
        num_layers: int | tuple[int, ...] = 3,
        num_classes: int | None = None,
        layer_scale_init: float = 1e-6,
        mlp_ratios: tuple[int, int, int, int] = (4, 4, 4, 3),
        out_factor: int = 2,
        use_softmax: bool = False,
        target_size: tuple[int, int] | int | None = None,
        temporal_dilations: tuple[int, ...] = (1, 2, 4),
        widths: tuple[int, int, int, int] | None = None,
    ):
        super().__init__()

        downsampler = LearnedSpatialTemporalDownsampler(
            in_channels=input_dim, out_factor=out_factor, target_size=target_size
        )
        stem = ConvGamerStem(
            in_channels=downsampler.out_channels, use_softmax=use_softmax, use_norm=False
        )
        self.downsampler = downsampler
        self.spatial_stem = stem
        self.stem = nn.Sequential(downsampler, stem)
        self.frame_encoder = InceptionNeXtEncoder(
            input_dim=stem.out_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=0,
            layer_scale_init=layer_scale_init,
            mlp_ratios=mlp_ratios,
            widths=widths,
        )
        self.temporal_mix = CausalTemporalMixer(
            channels=self.frame_encoder.feature_dim, dilations=temporal_dilations
        )
        self.feature_norm = CausalLayerNorm(self.frame_encoder.feature_dim)
        self.norm = nn.LayerNorm(self.frame_encoder.feature_dim)
        # Foundation-model seam: no classification head by default.
        # Callers may attach one later via ``add_classification_head``.
        if num_classes is not None and num_classes > 0:
            warnings.warn(
                "ConvGamerEncoder is now a foundation model without a classification "
                "head. Pass ``num_classes=None`` (or omit). The head will be added "
                "separately via ``add_classification_head`` for downstream tasks.",
                FutureWarning,
                stacklevel=2,
            )
            self.head = nn.Linear(self.frame_encoder.feature_dim, num_classes)
        else:
            self.head = nn.Identity()

    def add_classification_head(self, num_classes: int) -> None:
        """Attach a classification head after foundation pre-training."""
        self.head = nn.Linear(self.frame_encoder.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Per-frame video features ``(B, F, T)`` — no causal pooling, no head.

        Stem -> per-frame InceptionNeXt feature map -> causal temporal mix ->
        feature norm -> spatial mean. Output frame ``t`` is a function of
        input frames ``<= t`` (the temporal mix is causal). The
        cumulative-mean pooling that makes logits use only past context lives
        in ``forward``, keeping ``forward_features`` a clean backbone seam.
        """
        x = self.stem(x)
        b, _c, t, _h, _w = x.shape
        frames = einops.rearrange(x, "b c t h w -> (b t) c h w")
        maps = self.frame_encoder.forward_feature_map(frames)
        _, _, h, w = maps.shape
        video = einops.rearrange(maps, "(b t) f h w -> b f t h w", b=b, t=t, h=h, w=w)
        mixed = self.feature_norm(self.temporal_mix(video))
        return mixed.mean(dim=[3, 4])

    def forward_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial feature maps ``(B, F, T, H', W')`` for dense JEPA prediction.

        Same pipeline as ``forward_features`` but without the final spatial
        mean-pooling, preserving the 2D spatial structure needed by the
        predictor to produce dense predictions. Maps are channel-normalized
        with the pre-norm ``feature_norm`` before returning.
        """
        x = self.stem(x)
        b, _c, t, _h, _w = x.shape
        frames = einops.rearrange(x, "b c t h w -> (b t) c h w")
        maps = self.frame_encoder.forward_feature_map(frames)
        _, _, h, w = maps.shape
        video = einops.rearrange(maps, "(b t) f h w -> b f t h w", b=b, t=t, h=h, w=w)
        mixed = self.temporal_mix(video)
        return self.feature_norm(mixed)

    def init_state(
        self,
        batch_size: int = 1,
        height: int = 32,
        width: int = 32,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> StreamingState:
        """Fresh streaming state for every temporal component.

        ``height``/``width`` are the **input** spatial dims.  The downsampler
        conv runs at full resolution so it gets these directly; the stem and the
        temporal mixer see the downsampled spatial size, with the mixer one
        frame-encoder stride (4x) below that (matching
        ``InceptionNeXtEncoder``'s stride-4 stem conv).
        """
        ref = self.frame_encoder.stem.weight
        dev = ref.device if device is None else device
        dt = ref.dtype if dtype is None else dtype
        h_out, w_out = self.downsampler.resolve_spatial_size(height, width)
        fe_h = max(1, h_out // 4)
        fe_w = max(1, w_out // 4)
        return StreamingState(
            downsampler=self.downsampler.init_state(
                batch_size, height, width, device=dev, dtype=dt
            ),
            stem=self.spatial_stem.init_state(batch_size, h_out, w_out, device=dev, dtype=dt),
            mixer=self.temporal_mix.init_state(batch_size, fe_h, fe_w, device=dev, dtype=dt),
        )

    def step(self, x_t: torch.Tensor, state: StreamingState) -> StepOutput:
        """Stream one frame ``(B, C, H, W)``; returns a :class:`StepOutput`.

        ``features`` is ``(B, F, 1)`` (matching ``forward_features`` at that
        frame index) and ``logits`` is ``(B, K)`` or ``(B, 1, K)`` (matching
        ``forward`` pooled at that frame index).  When no classification head is
        attached (``head`` is ``nn.Identity``) ``logits`` is the pooled, normed
        feature directly.

        State is passed in and returned — the module keeps no history, so two
        streams can run concurrently and prefixes can be discarded at will.
        """
        b = x_t.shape[0]
        x_t = x_t.unsqueeze(2)  # (B, C, 1, H, W)
        x, downsampler_state = self.downsampler.step(x_t, state.downsampler)
        x, stem_state = self.spatial_stem.step(x, state.stem)
        # Per-frame feature map (spatial, no temporal mixing yet)
        frames = einops.rearrange(x, "b c t h w -> (b t) c h w")
        maps = self.frame_encoder.forward_feature_map(frames)
        _, _, h, w = maps.shape
        video = einops.rearrange(maps, "(b t) f h w -> b f t h w", b=b, t=1, h=h, w=w)
        mixed, mixer_state = self.temporal_mix.step(video, state.mixer)
        mixed = self.feature_norm(mixed)
        features = mixed.mean(dim=[3, 4])  # (B, F, 1)

        # Causal cumulative mean: frame t is pooled over frames <= t only.
        step_idx = state.step_idx + 1
        cumsum = (
            features.squeeze(2).clone()
            if state.cumsum is None
            else state.cumsum + features.squeeze(2)
        )
        logits = self.head(self.norm(cumsum / step_idx))

        next_state = StreamingState(
            downsampler=downsampler_state,
            stem=stem_state,
            mixer=mixer_state,
            cumsum=cumsum,
            step_idx=step_idx,
        )
        return StepOutput(
            features=features,
            logits=logits.unsqueeze(1) if logits.dim() == 2 else logits,
            new_state=next_state,
        )

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """Encode video to logits, causally pooled over past frames only.

        ``forward_features`` returns per-frame features; here we apply the
        causal cumulative-mean (frame ``t`` sees only ``<= t``) then the head.
        ``return_sequence=True`` skips pooling and emits per-frame logits.

        As a foundation model with no head, returns the pooled feature
        ``(B, F)`` instead of logits.
        """
        video = self.forward_features(x)
        if return_sequence:
            b, _, t = video.shape
            flat = einops.rearrange(video, "b f t -> (b t) f")
            logits = self.head(self.norm(flat))
            return einops.rearrange(logits, "(b t) k -> b t k", b=b, t=t)
        pooled = torch.cumsum(video, dim=2) / torch.arange(
            1, video.shape[2] + 1, device=video.device, dtype=video.dtype
        )
        return self.head(self.norm(pooled[:, :, -1]))
