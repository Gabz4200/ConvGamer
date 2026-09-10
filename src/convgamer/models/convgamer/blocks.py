import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from torchvision.transforms import v2


class LearnedDownsampler(nn.Module):
    r"""Learned spatial downsampler with channel expansion via a residual correction path.

    Resizes the input to ``target_size`` using differentiable interpolation, then
    tiles (repeats) the resized channels to reach ``out_channels``. A parallel
    learned convolution branch produces a correction tensor whose first
    ``in_channels`` slots are zero (so the tiling path stays untouched) and
    whose remaining slots hold learned residual features. The two are summed,
    yielding an output where every channel is a distinct learned quantity —
    avoiding the duplication redundancy a naive tile would impose.

    Args:
        target_size (tuple[int, int] or int): Desired output spatial
            ``(H_t, W_t)``. If an ``int``, both dimensions are set to it.
        kernel_size (int): Kernel size of the strided correction convolution.
            Default: ``3``
        in_channels (int): Number of channels in the input tensor.
            Default: ``3``
        intermediate_channels (int): Width of the hidden correction features
            produced by the strided convolution. Must be divisible by
            ``groups``. Default: ``192``
        out_channels (int): Number of output channels. Must be greater than
            ``in_channels`` so that the correction branch has room to learn
            new feature channels. Default: ``48``
        groups (int): Number of groups for the first correction convolution.
            When greater than 1, both ``in_channels`` and
            ``intermediate_channels`` must be divisible by ``groups``.
            Default: ``1``
        interpolation_mode (torchvision.transforms.v2.InterpolationMode):
            Interpolation mode used when resizing the input to
            ``target_size``. Default: ``InterpolationMode.BILINEAR``

    Shape:
        - Input: :math:`(B, \text{in\_channels}, H_{\text{in}}, W_{\text{in}})`
        - Output: :math:`(B, \text{out\_channels}, H_t, W_t)`

    .. note::
        The first ``in_channels`` output channels are an identity path: the
        resized input passes through unchanged with zero correction applied, so
        channels :math:`[0, \text{in\_channels})` equal the interpolated input.
        Channels :math:`[\text{in\_channels}, \text{out\_channels})` carry
        learned residual features and are distinct learned quantities.

        Tiling repeats channels in blocks of ``in_channels`` —
        ``[c0, c1, ..., c_{in-1}, c0, c1, ..., ...]`` — so that the first
        ``in_channels`` entries of the tiling align exactly with the zero-padded
        correction slots, preserving the identity path.

    Example::

        >>> import torch
        >>> op = LearnedDownsampler(
        ...     target_size=(16, 16),
        ...     in_channels=3,
        ...     out_channels=24,
        ... )
        >>> x = torch.randn(2, 3, 32, 32)
        >>> out = op(x)
        >>> out.shape
        torch.Size([2, 24, 16, 16])
    """

    def __init__(
        self,
        target_size: tuple[int, int] | int,
        kernel_size: int = 3,
        in_channels: int = 3,
        intermediate_channels: int = 192,  # Divisible by groups for grouped conv
        out_channels: int = 48,  # Divisible by in_channels for clean tiling
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
        # Pattern: (repeat c) gives [c0,c1,c2, c0,c1,c2, ...] so the first
        # in_channels of x_tiled align with the zero-padded correction path.
        x_full_blocks = repeat(
            x_resized,
            "b c h w -> b (repeat c) h w",
            repeat=repeats,
        )

        if remainder == 0:
            # Exact multiple: x_full_blocks already has out_channels in the
            # correct (repeat c) order — no rearrange needed.
            x_tiled = x_full_blocks
        else:
            # Extra channels from a prefix of x_resized. These match the
            # start of the block pattern [c0,c1,...] since x_resized[:remainder]
            # are channels [c0..c_{remainder-1}], consistent with (repeat c).
            x_prefix = x_resized[:, :remainder, :, :]  # [B, remainder, H_t, W_t]

            # Concatenate full blocks + prefix along the channel dimension.
            x_tiled = torch.cat([x_full_blocks, x_prefix], dim=1)  # [B, out_channels, H_t, W_t]

        # At this point:
        #   x_tiled: [B, out_channels, H_t, W_t]
        #   channel pattern: [c0,c1,c2, c0,c1,c2, ..., c0..c_{remainder-1}]

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
        #   x_tiled = [a,b,c, a,b,c, a,b,c, ...]
        #   correction = [0,0,0, d,e,f, g,h,i, ...]

        out: torch.Tensor = x_tiled + correction_padded
        return out
