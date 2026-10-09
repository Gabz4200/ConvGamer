"""Video encoder: causal downsampler stem, per-frame InceptionNeXt, causal head.

Input (B, C, T, H, W) -> logits (B, num_classes). Fully causal: no future
frame influences any past output. Output for frame ``t`` pools only frames
``<= t`` (or returns the full per-frame sequence when ``return_sequence``).

Multi-level design (V-JEPA 2.1 §2.3.2): the four InceptionNeXt stage outputs
are tapped as distinct feature levels, projected to a uniform level dim,
temporally mixed with shared weights, and fused for prediction. Each level is
a distinct modality — never a copy of the final map.

Streaming: all temporal layers are stateless functions of their input clip.
``step()`` keeps a fixed-size cache of projected level features (receptive
field minus one frames) and re-runs the mixer over the cache window, so
streaming matches ``forward()`` on the same prefix exactly while staying
O(1) in the sequence length. Training always uses the parallel ``forward``.
"""

from __future__ import annotations

import warnings

import einops
import torch
from torch import nn
from torch.nn.init import trunc_normal_

from convgamer.models.convgamer.causal import CausalLayerNorm, CausalTemporalMixer
from convgamer.models.convgamer.downsampler import LearnedSpatialTemporalDownsampler
from convgamer.models.convgamer.stem import ConvGamerStem
from convgamer.models.inception_next.encoder import InceptionNeXtEncoder
from convgamer.models.io import StepOutput, StreamCache
from convgamer.models.registry import register_model

#: Number of tapped encoder levels (the four InceptionNeXt stages).
NUM_LEVELS = 4


@register_model("ConvGamerEncoder")
class ConvGamerEncoder(nn.Module):
    """Video encoder: downsampler stem, per-frame InceptionNeXt, causal head.

    Per-frame path taps the four InceptionNeXt stage maps, projects each to
    ``level_dim``, mixes causally across T with shared weights, then fuses
    for prediction or pools spatially per level.

    ``step`` mirrors that pipeline one frame at a time with all history carried
    in an explicit :class:`~convgamer.models.io.StreamCache`.

    As a foundation model for JEPA pre-training this encoder exposes
    ``forward_feature_maps`` (level pyramid) as its public seam; the
    classification ``head`` is retained only for downstream fine-tuning.

    Args:
        input_dim: Input channel count (standardized RGB = 3).
        hidden_dim: Per-stage base channel count.
        num_layers: Per-stage block counts.
        num_classes: Deprecated; kept as ``None`` (foundation model). Ignored
            with a warning if a positive value is supplied.
        layer_scale_init: InceptionNeXt layer-scale init value.
        mlp_ratios: Per-stage MLP expansion ratios.
        out_factor: Channel multiplier applied by the downsampler stem.
        patch_size: Spatial patch size of the strided patchify stem.
        correction_kernel: Low-res correction conv — ``"dw7"`` (7x7 depthwise)
            or ``"dense4"`` (4x4 dense).
        expansion: Correction-branch width multiplier at low resolution.
        level_dim: Uniform per-level channel dim after projection.
        temporal_dilations: Dilations for the causal temporal mixer.
        widths: Optional explicit per-stage width overrides.
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
        patch_size: int = 8,
        correction_kernel: str = "dense4",
        expansion: int = 12,
        level_dim: int = 256,
        temporal_dilations: tuple[int, ...] = (1, 2, 4),
        widths: tuple[int, int, int, int] | None = None,
    ):
        super().__init__()
        self.level_dim = level_dim

        downsampler = LearnedSpatialTemporalDownsampler(
            in_channels=input_dim,
            out_factor=out_factor,
            patch_size=patch_size,
            correction_kernel=correction_kernel,
            expansion=expansion,
        )
        stem = ConvGamerStem(in_channels=downsampler.out_channels, use_norm=False)
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
        stage_dims = (
            tuple(widths)
            if widths is not None
            else (hidden_dim, hidden_dim * 2, hidden_dim * 4, hidden_dim * 8)
        )
        self.level_projs = nn.ModuleList(
            [nn.Conv3d(d, level_dim, kernel_size=1) for d in stage_dims]
        )
        self.temporal_mix = CausalTemporalMixer(channels=level_dim, dilations=temporal_dilations)
        self.level_norm = CausalLayerNorm(level_dim)
        self.level_fuse = nn.Sequential(
            nn.Conv3d(NUM_LEVELS * level_dim, level_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(level_dim, level_dim, kernel_size=1),
        )
        for m in list(self.level_projs) + list(self.level_fuse):
            if isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
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
            self.head = nn.Linear(level_dim, num_classes)
        else:
            self.head = nn.Identity()

    def add_classification_head(self, num_classes: int) -> None:
        """Attach a classification head after foundation pre-training."""
        self.head = nn.Linear(self.level_dim, num_classes)

    @property
    def history_frames(self) -> int:
        """Mixer cache window: receptive field minus one (fixed, O(1) in T)."""
        return self.temporal_mix.receptive_field - 1

    @property
    def _correction_history(self) -> int:
        """Downsampler correction window (temporal extent of its conv)."""
        return self.downsampler.correction_conv.temporal_extent

    @property
    def _stem_history(self) -> int:
        """Stem window (max temporal extent of the branch convs)."""
        stem = self.spatial_stem
        return max(
            stem.near_conv.temporal_extent,
            stem.local_conv.temporal_extent,
            stem.far_conv.temporal_extent,
            stem.fuse_conv.temporal_extent,
        )

    def _encode_projected(self, x: torch.Tensor) -> torch.Tensor:
        """Stem -> per-frame stage taps -> per-level projection.

        Returns stacked projected levels ``(B, 4, P, T, H', W')`` with no
        temporal mixing yet.
        """
        x = self.stem(x)
        b, t = x.shape[0], x.shape[2]
        frames = einops.rearrange(x, "b c t h w -> (b t) c h w")
        stage_maps = self.frame_encoder.forward_stage_maps(frames)
        if len(stage_maps) != NUM_LEVELS:
            raise ValueError(f"Expected {NUM_LEVELS} stage maps, got {len(stage_maps)}")
        levels = []
        for proj, maps in zip(self.level_projs, stage_maps, strict=True):
            video = einops.rearrange(maps, "(b t) c h w -> b c t h w", b=b, t=t)
            levels.append(proj(video))
        return torch.stack(levels, dim=1)

    def _mix_levels(self, projected: torch.Tensor) -> torch.Tensor:
        """Shared causal temporal mix + norm per level; stack unchanged."""
        if projected.ndim != 6 or projected.shape[1] != NUM_LEVELS:
            raise ValueError(
                f"Expected (B, {NUM_LEVELS}, P, T, H, W), got {tuple(projected.shape)}"
            )
        mixed = [self.level_norm(self.temporal_mix(projected[:, i])) for i in range(NUM_LEVELS)]
        return torch.stack(mixed, dim=1)

    def forward_pyramid(self, x: torch.Tensor) -> torch.Tensor:
        """Four temporally-mixed encoder levels, stacked.

        Returns ``(B, 4, P, T, H', W')``.
        """
        return self._mix_levels(self._encode_projected(x))

    def fuse_pyramid(self, pyramid: torch.Tensor) -> torch.Tensor:
        """Fuse stacked levels ``(B, 4, P, T, H, W)`` to ``(B, P, T, H, W)``."""
        if pyramid.ndim != 6 or pyramid.shape[1] != NUM_LEVELS:
            raise ValueError(
                f"Expected (B, {NUM_LEVELS}, P, T, H, W) pyramid, got {tuple(pyramid.shape)}"
            )
        b, _l, p, t, h, w = pyramid.shape
        flat = pyramid.reshape(b, NUM_LEVELS * p, t, h, w)
        return self.level_fuse(flat)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Per-frame fused video features — no causal pooling, no head.

        Returns ``(B, P, T)``. Output frame ``t`` is a function of input
        frames ``<= t`` (the temporal mix is causal). The cumulative-mean
        pooling for logits lives in ``forward``.
        """
        fused = self.fuse_pyramid(self.forward_pyramid(x))
        return fused.mean(dim=[-2, -1])

    def forward_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Level pyramid for dense JEPA prediction: ``(B, 4, P, T, H', W')``."""
        return self.forward_pyramid(x)

    def init_state(
        self,
        batch_size: int = 1,
        height: int = 32,
        width: int = 32,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> StreamCache:
        """Fresh empty streaming cache (one window per temporal stage).

        ``height``/``width`` are the **input** spatial dims. Windows start
        empty and grow to their bound, holding real frames only — each stage
        then supplies missing history through its own left padding, exactly
        like ``forward()`` on the same prefix.
        """
        ref = self.frame_encoder.stem.weight
        dev = ref.device if device is None else device
        dt = ref.dtype if dtype is None else dtype
        h_out, w_out = self.downsampler.resolve_spatial_size(height, width)
        fe_h = max(1, h_out // 4)
        fe_w = max(1, w_out // 4)
        base_ch = self.downsampler.out_channels
        # The down cache holds post-downsampler frames (same width as base).
        down_ch = base_ch
        return StreamCache(
            base=torch.zeros(batch_size, base_ch, 0, h_out, w_out, device=dev, dtype=dt),
            down=torch.zeros(batch_size, down_ch, 0, h_out, w_out, device=dev, dtype=dt),
            levels=torch.zeros(
                batch_size,
                NUM_LEVELS,
                self.level_dim,
                0,
                fe_h,
                fe_w,
                device=dev,
                dtype=dt,
            ),
        )

    def step(self, x_t: torch.Tensor, state: StreamCache) -> StepOutput:
        """Stream one frame ``(B, C, H, W)``; returns a :class:`StepOutput`.

        ``features`` is ``(B, P, 1)`` (matching ``forward_features`` at that
        frame index) and ``logits`` is ``(B, K)`` (matching ``forward`` pooled
        at that frame index). When no classification head is attached
        (``head`` is ``nn.Identity``) ``logits`` is the pooled feature
        directly.

        The cache is passed in and returned — the module keeps no history, so
        two streams can run concurrently and prefixes can be discarded at will.
        Each stage replays its stateless ``forward`` over its fixed cache
        window, so the output matches ``forward()`` on the same prefix.
        """
        ds = self.downsampler
        kb, ks, km = self._correction_history, self._stem_history, self.history_frames

        # Downsampler: stateless patchify, then the correction conv replayed
        # over its window; take the last frame to match forward().
        base_cur = ds.patch_embed(x_t.unsqueeze(2))
        base_win = torch.cat([state.base, base_cur], dim=2)[:, :, -kb - 1 :]
        assert base_win.shape[2] <= kb + 1
        correct = ds.act(ds.correction_conv(base_win))
        # Per-location norm: last-frame application matches forward() exactly.
        down_cur = ds.norm(base_cur + ds.out_correction_conv(correct)[:, :, -1:])

        # Stem: replayed over its window; take the last frame.
        down_win = torch.cat([state.down, down_cur], dim=2)[:, :, -ks - 1 :]
        assert down_win.shape[2] <= ks + 1
        stem_cur = self.spatial_stem(down_win)[:, :, -1:]

        # Per-frame stage taps + projection (no temporal mixing yet).
        frames = einops.rearrange(stem_cur, "b c t h w -> (b t) c h w")
        stage_maps = self.frame_encoder.forward_stage_maps(frames)
        b = stem_cur.shape[0]
        proj_cur = torch.stack(
            [
                proj(einops.rearrange(maps, "(b t) c h w -> b c t h w", b=b, t=1))
                for proj, maps in zip(self.level_projs, stage_maps, strict=True)
            ],
            dim=1,
        )
        lvl_win = torch.cat([state.levels, proj_cur], dim=3)[:, :, :, -km - 1 :]
        assert lvl_win.shape[3] <= km + 1
        mixed = self._mix_levels(lvl_win)[:, :, :, -1:]
        fused = self.fuse_pyramid(mixed)
        # Spatial mean over (H, W): (B, P, 1).
        features = fused.mean(dim=[-2, -1])

        # Causal cumulative mean: frame t is pooled over frames <= t only.
        # Detached: the cache carries history, not a gradient path (streaming
        # is inference-only; training always uses the parallel forward).
        count = state.count + 1
        cumsum = (
            features.detach().clone() if state.cumsum is None else state.cumsum + features.detach()
        )
        logits = self.head((cumsum / count).squeeze(2))

        return StepOutput(
            features=features,
            logits=logits,
            new_state=StreamCache(
                base=base_win[:, :, -kb:].detach(),
                down=down_win[:, :, -ks:].detach(),
                levels=lvl_win[:, :, :, -km:].detach() if km else lvl_win[:, :, :, :0].detach(),
                count=count,
                cumsum=cumsum,
            ),
        )

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """Encode video to logits, causally pooled over past frames only.

        ``forward_features`` returns per-frame features; here we apply the
        causal cumulative-mean (frame ``t`` sees only ``<= t``) then the head.
        ``return_sequence=True`` skips pooling and emits per-frame logits.

        As a foundation model with no head, returns the pooled feature
        ``(B, P)`` instead of logits.
        """
        video = self.forward_features(x)
        if return_sequence:
            b, _, t = video.shape
            flat = einops.rearrange(video, "b f t -> (b t) f")
            logits = self.head(flat)
            return einops.rearrange(logits, "(b t) k -> b t k", b=b, t=t)
        pooled = torch.cumsum(video, dim=2) / torch.arange(
            1, video.shape[2] + 1, device=video.device, dtype=video.dtype
        )
        return self.head(pooled[:, :, -1])
