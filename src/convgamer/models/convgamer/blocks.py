import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2


def uniform_temporal_subsample(
    x: torch.Tensor, num_samples: int, temporal_dim: int = -3
) -> torch.Tensor:
    """Equispaced nearest-neighbour temporal subsampling."""

    t = x.shape[temporal_dim]
    assert num_samples > 0 and t > 0

    # Use .round() for true nearest-neighbor.
    indices = torch.linspace(0, t - 1, num_samples, device=x.device, dtype=torch.float32)
    indices = indices.round().long()

    return torch.index_select(x, temporal_dim, indices)


class LearnedSpatialTemporalDownsampler(nn.Module):
    """Learned spatial-temporal downsampler with residual correction."""

    def __init__(
        self,
        in_channels: int = 3,
        intermediate_channels: int = 255,
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

        self.correction_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=kernel_size,
            groups=in_channels if depthwise else 1,
            padding="same",
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

        self.out_correction_conv = nn.Conv3d(
            in_channels=intermediate_channels,
            out_channels=self.out_channels,
            kernel_size=1,
        )

        self.norm = nn.GroupNorm(
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

        # Correction branch
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


class ConvGamerStem(nn.Module):
    """Multi-scale 3D convolution stem.

    Input: (B, C, T, H, W) -> Output: (B, 4*C, T, H, W).

    Three parallel branches (1x1x1 local, 3x3x3 near, 7x7x7 grouped far)
    are concatenated, fused with a 1x1x1 conv, then concatenated with the
    residual input and normalized. Resolution is preserved; downsampling is
    expected to happen before this module.
    """

    def __init__(self, in_channels: int = 24):
        super().__init__()
        branch_channels = in_channels * 2
        fused_channels = in_channels * 3

        self.in_channels = in_channels
        self.out_channels = in_channels + fused_channels

        self.near_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=branch_channels,
            kernel_size=3,
            padding=1,
        )
        self.local_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=branch_channels,
            kernel_size=1,
        )
        self.far_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=branch_channels,
            kernel_size=7,
            padding=3,
            groups=in_channels,
        )
        self.act = nn.GELU()
        self.fuse_conv = nn.Conv3d(
            in_channels=branch_channels * 3,
            out_channels=fused_channels,
            kernel_size=1,
        )
        self.norm = nn.GroupNorm(num_groups=4, num_channels=self.out_channels)

    def __getattr__(self, name: str) -> nn.Module:
        if name == "new_f_conv":
            modules = self.__dict__.get("_modules")
            if modules is not None and "fuse_conv" in modules:
                return modules["fuse_conv"]
            return object.__getattribute__(self, "fuse_conv")
        return super().__getattr__(name)  # type: ignore[misc]

    def __setattr__(self, name: str, value: object) -> None:
        if name == "new_f_conv":
            super().__setattr__("fuse_conv", value)
        else:
            super().__setattr__(name, value)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):  # type: ignore[no-untyped-def]
        if prefix + "new_f_conv.weight" in state_dict:
            state_dict[prefix + "fuse_conv.weight"] = state_dict.pop(prefix + "new_f_conv.weight")
        if prefix + "new_f_conv.bias" in state_dict:
            state_dict[prefix + "fuse_conv.bias"] = state_dict.pop(prefix + "new_f_conv.bias")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

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
        # Use squared distance for C1 smoothness; broadcasting handles arbitrary x.shape.
        sq_distances = (x.unsqueeze(-1) - self.x_cords).pow(2)
        weights = F.softmax(-sq_distances / self.temperature, dim=-1)
        out = torch.sum(weights * self.y_cords, dim=-1)
        return out.reshape(x.shape)
