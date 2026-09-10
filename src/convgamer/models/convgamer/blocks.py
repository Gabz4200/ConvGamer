import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2


def uniform_temporal_subsample(
    x: torch.Tensor, num_samples: int, temporal_dim: int = -3
) -> torch.Tensor:
    """Equispaced nearest-neighbour temporal subsampling.

    Replaces ``pytorchvideo.transforms.functional.uniform_temporal_subsample``
    with a dependency-free implementation: ``linspace`` over the temporal
    indices clamped to ``[0, t-1]`` then ``index_select``.
    """
    t = x.shape[temporal_dim]
    assert num_samples > 0 and t > 0
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return torch.index_select(x, temporal_dim, indices)


class LearnedSpatialTemporalDownsampler(nn.Module):
    """Learned spatial-temporal downsampler with residual correction.

    Tiling path resizes the input spatially then subsamples temporally and
    tiles channels by ``out_factor``. A depthwise correction branch learns a
    residual that is added before an optional concatenation with the
    downsampled original and GroupNorm.
    """

    def __init__(
        self,
        in_channels: int = 3,
        intermediate_channels: int = 255,
        out_factor: int = 4,
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

        self.downsample_x_down = v2.Resize(
            size=self.target_size, antialias=True, interpolation=self.interpolation_mode
        )

        self.out_correction_conv = nn.Conv3d(
            in_channels=intermediate_channels,
            out_channels=self.out_channels,
            kernel_size=1,
        )

        self.norm = nn.GroupNorm(num_groups=1, num_channels=norm_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (B, C, T, H, W), got shape {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {x.shape[1]}")

        T = x.shape[2]
        target_t = max(1, T // self.temporal_reduction_factor)
        h_out, w_out = self.target_size

        # Spatial downsampling
        x_down = x
        # v2.Resize expects (B, T, C, H, W) or (B, C, H, W)
        x_down = einops.rearrange(x_down, "b c t h w -> b t c h w")  # (B, T, C, H, W)
        x_down = self.downsample_x_down(x_down)
        x_down = einops.rearrange(x_down, "b t c h w -> b c t h w")  # (B, C, T, H, W)

        # Temporal downsampling with PyTorchVideo
        x_down = uniform_temporal_subsample(
            x_down, num_samples=target_t, temporal_dim=-3
        )  # (B, C, target_t, H, W)

        # Tile channels
        tiled_x_down = x_down.repeat_interleave(
            self.out_factor, dim=1
        )  # (B, C*out_factor, target_t, H, W)

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
