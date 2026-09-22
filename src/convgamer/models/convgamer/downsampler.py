from __future__ import annotations

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2

from convgamer.models.io import DownsamplerState

from .causal import CausalConv3d, CausalLayerNorm


def uniform_temporal_subsample(
    x: torch.Tensor, num_samples: int, temporal_dim: int = -3
) -> torch.Tensor:
    """Strided causal temporal subsampling for the streaming contract.

    Takes every ``step``-th frame starting at index 0, where
    ``step = max(1, T // num_samples)``. Online-safe: output frame ``i``
    depends only on input frames ``<= i * step``, so prefixes can be
    emitted without seeing the full clip. The last output frame is the
    latest frame at or before ``(num_samples - 1) * step``, which may be
    earlier than ``T - 1`` when ``T`` is not a multiple of ``step``.
    """
    t = x.shape[temporal_dim]
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")
    if t <= 0:
        raise ValueError(f"temporal dim size must be > 0, got {t}")

    step = max(1, t // num_samples)
    indices = torch.arange(0, t, step, device=x.device)[:num_samples]
    return torch.index_select(x, temporal_dim, indices)


class LearnedSpatialTemporalDownsampler(nn.Module):
    """Learned spatial-temporal downsampler with residual correction.

    Streaming contract: temporal subsampling is strided from frame 0, so
    output frame ``i`` depends only on input frames ``<= i * step``.
    Prefixes can be emitted online without seeing the full clip.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channel_multiple: int = 85,
        out_factor: int = 2,
        kernel_size: tuple[int, int, int] | int = (7, 7, 3),
        target_size: tuple[int, int] | int | None = None,
        max_spatial_size: int = 64,
        interpolation_mode: v2.InterpolationMode = v2.InterpolationMode.BILINEAR,
        depthwise: bool = True,
        concat_original: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_factor = out_factor
        self.kernel_size = kernel_size
        self.depthwise = depthwise
        self.concat_original = concat_original
        self.channel_multiple = channel_multiple
        # Correction branch stays expressive and depthwise-compatible by
        # construction: intermediate channels are always a multiple of inputs.
        intermediate_channels = in_channels * channel_multiple
        self.intermediate_channels = intermediate_channels
        self.interpolation_mode = interpolation_mode
        self.max_spatial_size = max_spatial_size

        if out_factor <= 0:
            raise ValueError(f"out_factor must be positive, got {out_factor}")
        if max_spatial_size <= 0:
            raise ValueError(f"max_spatial_size must be positive, got {max_spatial_size}")

        if target_size is None:
            self.target_size: tuple[int, int] | None = None
        else:
            self.target_size = (
                (target_size, target_size) if isinstance(target_size, int) else target_size
            )

        # Compute output channels
        base_channels = in_channels * out_factor
        self.out_channels = base_channels + (in_channels if concat_original else 0)

        if channel_multiple <= 0:
            raise ValueError(f"channel_multiple must be positive, got {channel_multiple}")

        self.correction_conv = CausalConv3d(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=kernel_size,
            groups=in_channels if depthwise else 1,
        )
        self.act = nn.GELU()

        # antialias is only supported for BILINEAR and BICUBIC
        antialias = self.interpolation_mode in (
            v2.InterpolationMode.BILINEAR,
            v2.InterpolationMode.BICUBIC,
        )
        self._antialias = antialias

        self.out_correction_conv = CausalConv3d(
            in_channels=intermediate_channels,
            out_channels=base_channels,
            kernel_size=1,
        )

        self.norm = CausalLayerNorm(self.out_channels)

    def resolve_spatial_size(self, h: int, w: int) -> tuple[int, int]:
        """Public spatial-size contract used by the encoder streaming state."""
        return self._resolve_spatial_size(h, w)

    def _resolve_spatial_size(self, h: int, w: int) -> tuple[int, int]:
        if self.target_size is not None:
            return self.target_size
        scale = min(1.0, self.max_spatial_size / max(h, w))
        return (max(1, round(h * scale)), max(1, round(w * scale)))

    def _resize_spatial(self, x_down: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return v2.Resize(
            size=size,
            antialias=self._antialias,
            interpolation=self.interpolation_mode,
        )(x_down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W) — temporal dim preserved, spatial downsampled only
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (B, C, T, H, W), got shape {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {x.shape[1]}")

        T = x.shape[2]
        h_out, w_out = self._resolve_spatial_size(x.shape[3], x.shape[4])

        # Spatial downsampling (relative mode never upsamples; explicit target wins)
        if (h_out, w_out) == (x.shape[3], x.shape[4]):
            x_down = x
        else:
            x_down = einops.rearrange(x, "b c t h w -> b t c h w")
            x_down = self._resize_spatial(x_down, (h_out, w_out))
            x_down = einops.rearrange(x_down, "b t c h w -> b c t h w")

        # Block tile [c0,c1,.., c0,c1,..]: identity lives in first
        # in_channels slots; correction zero-pads the remainder.
        tiled_x_down = x_down.repeat(1, self.out_factor, 1, 1, 1)

        # Correction branch runs before downsampling so it can learn from
        # full-resolution spatial information. Temporal context is captured
        # via the 3D conv's causal padding; T is preserved.
        correct = self.correction_conv(x)
        correct = self.act(correct)
        correct = einops.rearrange(correct, "b c t h w -> (b t) c h w")
        correct = F.adaptive_avg_pool2d(correct, output_size=(h_out, w_out))
        correct = einops.rearrange(correct, "(b t) c h w -> b c t h w", b=x.shape[0], t=T)
        correct = self.out_correction_conv(correct)  # (B, C*out_factor, T, H, W)

        # Combine
        out = tiled_x_down + correct

        # Concat downsampled (spatial) original with corrected output
        if self.concat_original:
            out = torch.cat((x_down, out), dim=1)

        out = self.norm(out)
        return out

    def init_state(
        self,
        batch_size: int,
        h: int,
        w: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> DownsamplerState:
        """Fresh streaming state for both correction convs.

        ``h``/``w`` are the **input** frame's spatial dims: the correction conv
        runs at full resolution before pooling, so it is the one that needs
        history.  ``out_correction_conv`` is a 1x1x1 conv and keeps an empty
        window.
        """
        return DownsamplerState(
            correction=self.correction_conv.init_state(
                batch_size, h, w, device=device, dtype=dtype
            ),
            out_correction=self.out_correction_conv.init_state(
                batch_size, h, w, device=device, dtype=dtype
            ),
        )

    def step(
        self, x_t: torch.Tensor, state: DownsamplerState
    ) -> tuple[torch.Tensor, DownsamplerState]:
        """Stream one frame ``(B, C, 1, H, W)``; returns ``(out_t, next_state)``.

        Matches ``forward`` frame-by-frame: spatial resize, causal correction
        conv carrying explicit history, channel repeat, residual, norm.
        """
        b, c, _t, h, w = x_t.shape
        h_out, w_out = self._resolve_spatial_size(h, w)

        # Spatial downsample current frame
        if (h_out, w_out) == (h, w):
            x_down = x_t
        else:
            x_down = einops.rearrange(x_t, "b c t h w -> b t c h w")
            x_down = self._resize_spatial(x_down, (h_out, w_out))
            x_down = einops.rearrange(x_down, "b t c h w -> b c t h w")

        tiled_x_down = x_down.repeat(1, self.out_factor, 1, 1, 1)

        # Correction branch: step through the causal convs with explicit state
        correct, correction_state = self.correction_conv.step(x_t, state.correction)
        correct = self.act(correct)
        correct = einops.rearrange(correct, "b c t h w -> (b t) c h w")
        correct = F.adaptive_avg_pool2d(correct, output_size=(h_out, w_out))
        correct = einops.rearrange(correct, "(b t) c h w -> b c t h w", b=b, t=1)
        correct, out_correction_state = self.out_correction_conv.step(correct, state.out_correction)

        out = tiled_x_down + correct
        if self.concat_original:
            out = torch.cat((x_down, out), dim=1)
        out = self.norm(out)
        return out, DownsamplerState(
            correction=correction_state, out_correction=out_correction_state
        )


def spatial_softmax(x: torch.Tensor) -> torch.Tensor:
    """Applies softmax over (H, W) independently for every channel.

    Supports (B, C, H, W) and (B, C, T, H, W) tensors with any channel count.
    H and W are always the trailing dims; other leading dims never mix.
    """
    if x.ndim not in (4, 5):
        raise ValueError(f"Expected 4D (B,C,H,W) or 5D (B,C,T,H,W), got {x.ndim}D")

    # Flatten only the last two spatial dimensions (H, W) -> (..., H*W)
    # For 4D: (B, C, H, W) -> (B, C, H*W)
    # For 5D: (B, T, C, H, W) -> (B, T, C, H*W)
    flattened = x.flatten(start_dim=-2)

    # Apply softmax over the flattened spatial dimension
    probs = F.softmax(flattened.float(), dim=-1).to(dtype=flattened.dtype)

    # Restore original shape
    return probs.view_as(x)
