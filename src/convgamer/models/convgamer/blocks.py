import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision.transforms import v2


class LearnedDownsampler(nn.Module):
    """
    Uses a learned convolution to get a correction for an existing downsampling
    to make it more ML friendly.

    - After processing, both `x` and `correction` have shape:
        [B, out_channels, H_t, W_t], where (H_t, W_t) = target_size.
    - `x` channels are tiled copies of the resized input:
        [c0, c1, c2, c0, c1, c2, ...] up to out_channels.
    - `correction` channels are:
        [0, 0, ..., 0 (in_channels times), d0, d1, ..., d_{out_channels-in_channels-1}]
    - Output = x + correction.

    """

    def __init__(
        self,
        target_size: tuple[int, int] | int,
        kernel_size: int = 3,
        in_channels: int = 3,
        intermediate_channels: int = 192,  # Multiple of 3 and 8
        out_channels: int = 48,  # Multiple of 3 and 8
        groups: int = 1,
        interpolation_mode: v2.InterpolationMode = v2.InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()

        if out_channels <= in_channels:
            raise ValueError("out_channels must be > in_channels for this design.")

        if groups > 1:
            if in_channels % groups != 0:
                raise ValueError("in_channels must be divisible by groups")
            if intermediate_channels % groups != 0:
                raise ValueError("intermediate_channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.intermediate_channels = intermediate_channels
        self.kernel_size = kernel_size

        self.target_size: tuple[int, int] = (
            target_size if isinstance(target_size, tuple) else (target_size, target_size)
        )

        # Strided Conv to produce an intermediate representation for correction.
        self.input_correction_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=2,
            groups=groups,
        )

        self.pool_correction = nn.AdaptiveAvgPool2d(self.target_size)

        self.pool_image = v2.Resize(
            size=self.target_size,
            interpolation=interpolation_mode,
            antialias=True,
        )

        self.out_correction_conv = nn.Conv2d(
            in_channels=intermediate_channels,
            out_channels=out_channels - in_channels,
            kernel_size=1,
            groups=1,  # Always 1, this is intentional
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H_in, W_in = x.shape

        if self.in_channels != C:
            raise ValueError(f"Expected {self.in_channels} input channels, got {C}")

        # Learned correction path
        correction = self.input_correction_conv(x)  # [B, intermediate_channels, H//2, W//2]
        correction = self.pool_correction(correction)  # [B, intermediate_channels, H_t, W_t]
        correction = self.out_correction_conv(
            correction
        )  # [B, out_channels - in_channels, H_t, W_t]

        # Resize input to target spatial size
        x_resized = self.pool_image(x)  # [B, in_channels, H_t, W_t]

        # Channel tiling with einops:
        # Goal: turn [B, in_channels, H_t, W_t] into [B, out_channels, H_t, W_t]
        # with pattern: [c0, c1, c2, c0, c1, c2, ...] (tiled) up to out_channels.

        repeats = self.out_channels // self.in_channels
        remainder = self.out_channels % self.in_channels

        # Repeat full blocks of in_channels -> [B, in_channels * repeats, H_t, W_t]
        x_full_blocks = repeat(
            x_resized,
            "b c h w -> b (c repeat) h w",
            repeat=repeats,
        )

        if remainder == 0:
            # Exact multiple: just take the first out_channels (should already match)
            x_tiled = rearrange(
                x_full_blocks,
                "b (c repeat) h w -> b (repeat c) h w",
                c=self.in_channels,
                repeat=repeats,
            )
            # Ensure exact channel count (safety slice)
            x_tiled = x_tiled[:, : self.out_channels, :, :]
        else:
            # We have extra channels to fill from a prefix of x_resized.
            # Take the first `remainder` channels explicitly using rearrange:
            x_prefix = rearrange(
                x_resized,
                "b c h w -> b c h w",
            )[:, :remainder, :, :]  # [B, remainder, H_t, W_t]

            # Now concatenate full blocks + prefix along channel dimension.
            # We keep this cat explicit for clarity; einops doesn't add much here.
            x_tiled = torch.cat([x_full_blocks, x_prefix], dim=1)  # [B, out_channels, H_t, W_t]

        # At this point:
        #   x_tiled: [B, out_channels, H_t, W_t]
        #   channel pattern: [c0..c_{in-1}, c0..c_{in-1}, ..., c0..c_{remainder-1}]

        # Build correction tensor with zeros in the first in_channels,
        # then learned features in the remaining out_channels - in_channels.

        zeros = torch.zeros(
            B,
            self.in_channels,
            *correction.shape[2:],
            device=correction.device,
            dtype=correction.dtype,
        )

        # Prepend zero channels to correction:
        #   zeros:          [B, in_channels, H_t, W_t]
        #   correction:     [B, out_channels - in_channels, H_t, W_t]
        #   -> correction_padded: [B, out_channels, H_t, W_t]
        correction_padded = torch.cat([zeros, correction], dim=1)

        # Now:
        #   x_tiled:           [B, out_channels, H_t, W_t], tiled input channels
        #   correction_padded: [B, out_channels, H_t, W_t], [0..0, learned...]
        # This matches:
        #   x = [a,b,c,a,b,c,a,b,c,...]
        #   correction = [0,0,0,d,e,f,g,h,i,j,k,l,...]

        out: torch.Tensor = x_tiled + correction_padded
        return out
