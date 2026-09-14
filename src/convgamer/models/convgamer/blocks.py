from __future__ import annotations

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2


class CausalConv3d(nn.Module):
    """3D convolution causal in T, symmetric in H/W.

    Temporal causality via left-only padding on depth (T) dimension.
    Spatial dims are padded symmetrically for odd kernels. Even spatial kernels
    use one extra pixel on the right/bottom to preserve the output shape.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int] | int,
        stride: tuple[int, int, int] | int = 1,
        dilation: tuple[int, int, int] | int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        def _triple(v: tuple[int, int, int] | int) -> tuple[int, int, int]:
            return (v, v, v) if isinstance(v, int) else v

        kt, kh, kw = _triple(kernel_size)
        dt, dh, dw = _triple(dilation)
        self._kernel_t = kt
        self._stride = _triple(stride)
        self._dilation = (dt, dh, dw)

        pt = dt * (kt - 1)
        ph = dh * (kh - 1)
        pw = dw * (kw - 1)

        # temporal left only, spatial symmetric
        self._pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2, pt, 0)

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kt, kh, kw),
            stride=self._stride,
            padding=0,
            dilation=(dt, dh, dw),
            groups=groups,
            bias=bias,
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.conv.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if any(p != 0 for p in self._pad):
            x = F.pad(x, self._pad)
        return self.conv(x)


class CausalLayerNorm(nn.LayerNorm):
    """Channel LayerNorm applied per location to preserve temporal causality."""

    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)
        self.num_channels = num_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if input.ndim == 5:
            b, _c, t, h, w = input.shape
            y = einops.rearrange(input, "b c t h w -> (b t h w) c")
            y = super().forward(y)
            return einops.rearrange(y, "(b t h w) c -> b c t h w", b=b, t=t, h=h, w=w)
        if input.ndim == 4:
            b, _c, h, w = input.shape
            y = einops.rearrange(input, "b c h w -> (b h w) c")
            y = super().forward(y)
            return einops.rearrange(y, "(b h w) c -> b c h w", b=b, h=h, w=w)
        raise ValueError(f"Expected 4D (B,C,H,W) or 5D (B,C,T,H,W), got {input.ndim}D")


class CausalTemporalMixer(nn.Module):
    """Dilated causal TCN over frame vectors (B, F, T).

    Stacks ``CausalConv3d(kernel=(3, 1, 1))`` layers with dilations
    ``(1, 2, 4)`` by default: receptive field 15 subsampled frames with
    3 layers instead of 3 frames for a single conv. Each layer is
    residual (``x + GELU(conv(x))``); all ops are pointwise or causal,
    so no future frame leaks into the past. Residuals are unnormalized:
    fine at the default depth, add a per-layer norm past ~6 layers.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__()
        self.channels = channels
        self.dilations = dilations
        self.receptive_field = 1 + sum(d * (kernel_size - 1) for d in dilations)
        self.layers = nn.ModuleList(
            [
                CausalConv3d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=(kernel_size, 1, 1),
                    dilation=(d, 1, 1),
                )
                for d in dilations
            ]
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.layers:
            x = x + self.act(conv(x))
        return x


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
        intermediate_channels: int | None = None,
        out_factor: int = 2,
        kernel_size: tuple[int, int, int] | int = (7, 7, 3),
        target_size: tuple[int, int] | int | None = None,
        max_spatial_size: int = 64,
        temporal_reduction_factor: int = 2,
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
        if intermediate_channels is None:
            # 85x oversampling keeps the depthwise correction branch expressive
            # while staying divisible by in_channels for any input_dim.
            intermediate_channels = in_channels * 85
        self.intermediate_channels = intermediate_channels
        self.temporal_reduction_factor = temporal_reduction_factor
        self.interpolation_mode = interpolation_mode
        self.max_spatial_size = max_spatial_size

        if temporal_reduction_factor <= 0:
            raise ValueError(
                f"temporal_reduction_factor must be positive, got {temporal_reduction_factor}"
            )
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

        if depthwise and intermediate_channels % in_channels != 0:
            raise ValueError(
                f"intermediate_channels={intermediate_channels} must be divisible "
                f"by in_channels={in_channels} for depthwise convolution"
            )

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
        # x: (B, C, T, H, W)
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (B, C, T, H, W), got shape {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {x.shape[1]}")

        T = x.shape[2]
        # Clamp target_t to be at least 1 and at most T (if temporal_reduction_factor < 1)
        target_t = max(1, min(T, T // self.temporal_reduction_factor))
        h_out, w_out = self._resolve_spatial_size(x.shape[3], x.shape[4])

        # Spatial downsampling (relative mode never upsamples; explicit target wins)
        if (h_out, w_out) == (x.shape[3], x.shape[4]):
            x_down = x
        else:
            x_down = einops.rearrange(x, "b c t h w -> b t c h w")
            x_down = self._resize_spatial(x_down, (h_out, w_out))
            x_down = einops.rearrange(x_down, "b t c h w -> b c t h w")

        # Temporal downsampling
        x_down = uniform_temporal_subsample(
            x_down, num_samples=target_t, temporal_dim=-3
        )  # (B, C, target_t, H, W)

        # Repeat channels, repeat_interleave produces [C0, C0, C0, C1, C1, C1, ..., Cn, Cn, Cn].
        tiled_x_down = x_down.repeat_interleave(self.out_factor, dim=1)

        # Correction branch runs before downsampling so it can learn from
        # full-resolution spatial and temporal information.
        correct = self.correction_conv(x)
        correct = self.act(correct)
        correct = einops.rearrange(correct, "b c t h w -> (b t) c h w")
        correct = F.adaptive_avg_pool2d(correct, output_size=(h_out, w_out))
        correct = einops.rearrange(correct, "(b t) c h w -> b c t h w", b=x.shape[0], t=T)
        correct = uniform_temporal_subsample(correct, num_samples=target_t, temporal_dim=-3)
        correct = self.out_correction_conv(correct)  # (B, C*out_factor, target_t, H, W)

        # Combine
        out = tiled_x_down + correct

        # Concat downsampled (spatial+temporal) original with corrected output
        if self.concat_original:
            out = torch.cat((x_down, out), dim=1)

        out = self.norm(out)
        return out


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
