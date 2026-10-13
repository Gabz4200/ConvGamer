from __future__ import annotations

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv3d(nn.Module):
    """3D convolution causal in T, symmetric in H/W.

    Temporal causality via left-only padding on depth (T) dimension.
    Spatial dims are padded symmetrically for odd kernels. Even spatial kernels
    use one extra pixel on the right/bottom to preserve the output shape.

    Stateless: ``forward`` maps a whole clip ``(B, C_in, T, H, W)``.
    Frame-by-frame streaming is handled one level up by the encoder, which
    re-runs this stateless mapping over its fixed-size frame cache.
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
        stride = _triple(stride)
        if stride[0] != 1:
            raise ValueError(
                f"CausalConv3d requires temporal stride 1 for streaming parity, got {stride}"
            )

        pt = dt * (kt - 1)
        ph = dh * (kh - 1)
        pw = dw * (kw - 1)

        # temporal left only, spatial symmetric
        self._pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2, pt, 0)
        self._pt = pt

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kt, kh, kw),
            stride=stride,
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

    @property
    def temporal_extent(self) -> int:
        """Past frames needed alongside the current one (left pad depth)."""
        return self._pt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if any(p != 0 for p in self._pad):
            x = F.pad(x, self._pad)
        return self.conv(x)


class CausalLayerNorm(nn.LayerNorm):
    """Channel LayerNorm applied per location to preserve temporal causality."""

    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)

    @property
    def num_channels(self) -> int:
        return self.normalized_shape[0]

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
    3 layers instead of 3 frames for a single conv. Each layer is a pre-norm
    residual (``x + GELU(conv(norm(x)))``); all ops are pointwise or causal,
    so no future frame leaks into the past.
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
        self.norms = nn.ModuleList([CausalLayerNorm(channels) for _ in dilations])
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.layers, self.norms, strict=True):
            x = x + self.act(conv(norm(x)))
        return x
