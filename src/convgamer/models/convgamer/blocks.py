from __future__ import annotations

import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2


class CausalConv3d(nn.Module):
    """3D convolution causal in T, symmetric in H/W.

    Temporal causality via left-only padding on depth (T) dimension.
    Spatial dims padded symmetrically to preserve H/W with stride 1.
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


class CausalGroupNorm(nn.GroupNorm):
    """GroupNorm applied per-frame to preserve temporal causality."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if input.ndim == 5:
            b, _c, t, _h, _w = input.shape
            x = einops.rearrange(input, "b c t h w -> (b t) c h w")
            x = super().forward(x)
            return einops.rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
        return super().forward(input)


def uniform_temporal_subsample(
    x: torch.Tensor, num_samples: int, temporal_dim: int = -3
) -> torch.Tensor:
    """Equispaced nearest-neighbour temporal subsampling."""

    t = x.shape[temporal_dim]
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")
    if t <= 0:
        raise ValueError(f"temporal dim size must be > 0, got {t}")

    # Use .round() for true nearest-neighbor.
    indices = torch.linspace(0, t - 1, num_samples, device=x.device, dtype=torch.float32)
    indices = indices.round().long()

    return torch.index_select(x, temporal_dim, indices)


class LearnedSpatialTemporalDownsampler(nn.Module):
    """Learned spatial-temporal downsampler with residual correction."""

    def __init__(
        self,
        in_channels: int = 3,
        intermediate_channels: int = 255,  # 255=85*3 divisible by in_channels=3
        out_factor: int = 8,
        kernel_size: tuple[int, int, int] | int = (7, 7, 2),
        target_size: tuple[int, int] | int = (64, 64),
        temporal_reduction_factor: int = 2,
        interpolation_mode: v2.InterpolationMode = v2.InterpolationMode.BILINEAR,
        depthwise: bool = True,
        concat_original: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_factor = out_factor
        self.kernel_size = kernel_size
        self.depthwise = depthwise
        self.concat_original = concat_original
        self.intermediate_channels = intermediate_channels
        self.temporal_reduction_factor = temporal_reduction_factor
        self.interpolation_mode = interpolation_mode

        if temporal_reduction_factor <= 0:
            raise ValueError(
                f"temporal_reduction_factor must be positive, got {temporal_reduction_factor}"
            )

        # Normalize target_size to a tuple of ints for spatial dims
        self.target_size: tuple[int, int] = (
            (target_size, target_size) if isinstance(target_size, int) else target_size
        )

        # Compute output channels
        base_channels = in_channels * out_factor
        norm_channels = base_channels + (in_channels if concat_original else 0)
        self.out_channels = base_channels

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
        self.downsample_x_down = v2.Resize(
            size=self.target_size, antialias=antialias, interpolation=self.interpolation_mode
        )

        self.out_correction_conv = CausalConv3d(
            in_channels=intermediate_channels,
            out_channels=self.out_channels,
            kernel_size=1,
        )

        self.norm = CausalGroupNorm(
            num_groups=norm_channels // in_channels, num_channels=norm_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (B, C, T, H, W), got shape {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {x.shape[1]}")

        T = x.shape[2]
        # Clamp target_t to be at least 1 and at most T (if temporal_reduction_factor < 1)
        target_t = max(1, min(T, T // self.temporal_reduction_factor))
        h_out, w_out = self.target_size

        # Spatial downsampling
        x_down = x
        x_down = einops.rearrange(x_down, "b c t h w -> b t c h w")  # (B, T, C, H, W)
        x_down = self.downsample_x_down(x_down)
        x_down = einops.rearrange(x_down, "b t c h w -> b c t h w")  # (B, C, T, H, W)

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
        correct = F.adaptive_avg_pool3d(
            correct,
            output_size=(target_t, h_out, w_out),
        )
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

    Supports (B, C, H, W) and (B, T, C, H, W) tensors with any channel count.
    Cross-frame channels do not communicate; each (b, t, c) slice gets its own spatial softmax.
    """
    if x.ndim not in (4, 5):
        raise ValueError(f"Expected 4D (B,C,H,W) or 5D (B,T,C,H,W), got {x.ndim}D")

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

    Input: (B, C, T, H, W) -> Output: (B, 8*C, T, H, W) by default.

    Three parallel branches (1x1x1 local, 3x3x3 near, 7x7x7 grouped far)
    are concatenated, fused with a 1x1x1 conv, then concatenated with the
    residual input and normalized. Resolution is preserved; downsampling is
    expected to happen before this module.
    """

    def __init__(self, in_channels: int = 24, use_softmax: bool = True, use_norm=False) -> None:
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
        self.norm = (
            CausalGroupNorm(num_groups=4, num_channels=self.out_channels)
            if use_norm
            else nn.Identity()
        )

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


class SmoothPWAct(nn.Module):
    """Smooth piecewise activation via Gaussian kernel.

    For each value in the input tensor:
    1. Compute squared distance to each x_cord.
    2. Convert to weights with ``softmax(-sq_dist / temperature)``

    3. Weighted sum of y_cords.
    4. Return tensor with same shape as input. Works for arbitrary input shapes.
    """

    def __init__(self, num_anchors: int = 16, temperature: float = 0.1) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.temperature = temperature

        if num_anchors <= 1:
            raise ValueError("num_anchors must be > 1.")

        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.y_cords = nn.Parameter(torch.linspace(-2, 2, num_anchors))
        self.x_cords = nn.Parameter(torch.linspace(-2, 2, num_anchors))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape contract: (B, C), which is a 1d vector but allows Batching.
        # Use squared distance for C1 smoothness; broadcasting handles arbitrary x.shape.
        x_cords = self.x_cords.to(dtype=x.dtype)
        y_cords = self.y_cords.to(dtype=x.dtype)
        sq_distances = (x.unsqueeze(-1) - x_cords).pow(2)
        weights = F.softmax(-sq_distances / self.temperature, dim=-1)
        out = torch.sum(weights * y_cords, dim=-1)
        return out.reshape(x.shape)


class StrictLearnableGrid(nn.Module):
    def __init__(
        self,
        n_points: int,
        low: float,
        high: float,
        min_spacing: float,
    ) -> None:
        super().__init__()

        if n_points < 2:
            raise ValueError("n_points must be at least 2")

        if not (math.isfinite(low) and math.isfinite(high) and math.isfinite(min_spacing)):
            raise ValueError("low, high, and min_spacing must be finite")

        if high <= low:
            raise ValueError("high must be greater than low")

        if min_spacing < 0:
            raise ValueError("min_spacing must be non-negative")

        intervals = n_points - 1
        domain_length = high - low
        available = domain_length - intervals * min_spacing

        if available < 0:
            raise ValueError("min_spacing is too large for this interval")

        self.n_points = n_points
        self.intervals = intervals
        self.low = low
        self.high = high
        self.min_spacing = min_spacing
        self.available = available

        self.raw = nn.Parameter(torch.zeros(intervals))

    def forward(self) -> torch.Tensor:
        weights = torch.softmax(self.raw, dim=0)
        gaps = self.min_spacing + self.available * weights

        origin = torch.zeros_like(gaps[:1])
        positions = torch.cat([origin, torch.cumsum(gaps, dim=0)])

        return positions + self.low


class PWInterpolationAct(nn.Module):
    """Learnable piecewise-linear interpolation activation.

    The module defines a function through learnable monotonic knots:

        (x_cords[0], y_cords[0]), ..., (x_cords[N - 1], y_cords[N - 1])

    Inputs of any shape are interpolated elementwise. Inputs outside
    [min_range, max_range] are clamped to endpoint values.
    """

    def __init__(
        self,
        num_anchors: int = 16,
        min_range: float = -2.0,
        max_range: float = 2.0,
        min_spacing: float = 0.1,
        monotonic: bool = True,
    ) -> None:
        super().__init__()

        if num_anchors < 2:
            raise ValueError("num_anchors must be at least 2")

        if max_range <= min_range:
            raise ValueError("max_range must be greater than min_range")

        if min_spacing < 0:
            raise ValueError("min_spacing must be non-negative")

        if (num_anchors - 1) * min_spacing > max_range - min_range:
            raise ValueError("min_spacing is too large for the selected range")

        self.num_anchors = num_anchors
        self.min_range = min_range
        self.max_range = max_range
        self.monotonic = monotonic

        self.x_cords_gen = StrictLearnableGrid(
            n_points=num_anchors,
            low=min_range,
            high=max_range,
            min_spacing=min_spacing,
        )

        self.y_cords_gen: StrictLearnableGrid | None
        self.y_cords: nn.Parameter | None
        if monotonic:
            self.y_cords_gen = StrictLearnableGrid(
                n_points=num_anchors,
                low=min_range,
                high=max_range,
                min_spacing=min_spacing,
            )
            self.y_cords = None
        else:
            self.y_cords_gen = None
            self.y_cords = nn.Parameter(torch.linspace(min_range, max_range, num_anchors))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_query = x.reshape(-1)

        x_cords = self.x_cords_gen().to(dtype=x.dtype)
        if self.monotonic:
            y_cords_gen = self.y_cords_gen
            # It is impossible for it to be None at this point,
            # but pyrefly yells at me if I dont check for it
            if y_cords_gen is None:
                raise RuntimeError("monotonic mode requires y_cords_gen")
            y_cords = y_cords_gen().to(dtype=x.dtype)
        else:
            y_cords = self.y_cords
            # It is impossible for it to be None at this point,
            # but pyrefly yells at me if I dont check for it
            if y_cords is None:
                raise RuntimeError("non-monotonic mode requires y_cords")
            y_cords = y_cords.to(dtype=x.dtype)

        # Outside-domain behavior: constant endpoint extrapolation.
        x_query = x_query.clamp(
            min=x_cords[0],
            max=x_cords[-1],
        )

        # index satisfies:
        # x_cords[index - 1] <= x_query < x_cords[index]
        right = torch.searchsorted(
            x_cords,
            x_query,
            right=True,
        )

        # Convert insertion position into the left interval index.
        left = (right - 1).clamp(
            min=0,
            max=self.num_anchors - 2,
        )
        right = left + 1

        x_left = x_cords[left]
        x_right = x_cords[right]

        y_left = y_cords[left]
        y_right = y_cords[right]

        denominator = x_right - x_left
        weight = (x_query - x_left) / denominator

        return torch.lerp(y_left, y_right, weight).reshape(original_shape)
