from __future__ import annotations

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from .minconv import MinConvExpLSTM


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
        use_caching: bool = True,
    ) -> None:
        super().__init__()

        def _triple(v: tuple[int, int, int] | int) -> tuple[int, int, int]:
            return (v, v, v) if isinstance(v, int) else v

        kt, kh, kw = _triple(kernel_size)
        dt, dh, dw = _triple(dilation)
        self._kernel_t = kt
        self._stride = _triple(stride)
        self._dilation = (dt, dh, dw)
        self._use_caching = use_caching

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
            stride=self._stride,
            padding=0,
            dilation=(dt, dh, dw),
            groups=groups,
            bias=bias,
        )

        if use_caching:
            # Cache: sliding window of past frames (B, C_in, pt, H, W)
            # Registered as buffer so it moves with .to(device)/.to(dtype)
            self.register_buffer(
                "_cache",
                torch.empty(0, in_channels, pt, 1, 1),
                persistent=False,
            )
        else:
            # Non-caching: still need a placeholder so step() can detect it
            self.register_buffer("_cache", torch.empty(0), persistent=False)

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.conv.bias

    def reset_cache(
        self,
        batch_size: int,
        spatial_h: int,
        spatial_w: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Allocate the cache buffer for streaming step.

        ``spatial_h``/``spatial_w`` are the H/W of the **input** frame
        (i.e., before any spatial padding).  The cache stores raw frames;
        spatial padding is applied per-frame at ``step`` time so it matches
        ``forward`` semantics exactly.
        """
        if not self._use_caching:
            raise RuntimeError(
                f"CausalConv3d({self.conv.kernel_size}) requires use_caching=True "
                "to use streaming step()"
            )
        c_in = self.conv.in_channels
        ref = self.conv.weight
        dev = ref.device if device is None else device
        dt = ref.dtype if dtype is None else dtype
        self._cache = torch.zeros(
            batch_size,
            c_in,
            self._pt,
            spatial_h,
            spatial_w,
            device=dev,
            dtype=dt,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if any(p != 0 for p in self._pad):
            x = F.pad(x, self._pad)
        return self.conv(x)

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        """Process one frame ``(B, C_in, 1, H, W)`` using cached history.

        The cache is a sliding window of ``pt`` past **raw** frames.  At
        each step the current frame is spatially padded (symmetric), prepended
        to the cache, and the convolution runs on the full padded window —
        mirroring ``forward`` exactly.  The cache slides by one (drop oldest).
        Returns ``(B, C_out, 1, H_out, W_out)``.
        """
        if self._pt == 0 and all(p == 0 for p in self._pad[:4]):
            # Kernel 1x1x1 with no padding — frame independent
            return self.conv(x_t)

        if not self._use_caching:
            raise RuntimeError(
                f"CausalConv3d({self.conv.kernel_size}) requires use_caching=True "
                "to use streaming step()"
            )

        b, c_in, _, h, w = x_t.shape
        cache = self._cache

        # Apply spatial padding to current frame only (symmetric on H, W).
        # Temporal left padding is supplied by the cache.
        sp_pad = (self._pad[0], self._pad[1], self._pad[2], self._pad[3])
        x_t_padded = F.pad(x_t, sp_pad + (0, 0)) if any(p != 0 for p in sp_pad) else x_t

        if cache.shape[2] != self._pt or cache.shape[3] != h or cache.shape[4] != w:
            raise RuntimeError(
                f"Cache mismatch: expected (B, {c_in}, {self._pt}, {h}, {w}), "
                f"got {tuple(cache.shape)}"
            )

        # Pad cache spatially too (so cached frames see the same border behavior)
        cache_padded = F.pad(cache, sp_pad + (0, 0)) if any(p != 0 for p in sp_pad) else cache

        # Full window: [pt padded past frames, 1 padded current frame]
        # Spatial pad gives hp/h_out and wp/w_out matching forward()
        window = torch.cat([cache_padded, x_t_padded], dim=2)  # (B, C, pt+1, hp, wp)
        # Conv with padding=0 (no temporal pad — cache handles it)
        out = self.conv(window)
        out_t = out[:, :, -1:]  # (B, C_out, 1, H_out, W_out)

        # Slide window: drop oldest frame from cache.  Store raw frame (pre-spatial-pad)
        # so the cache keeps original H/W.
        self._cache = torch.cat([cache[:, :, 1:], x_t], dim=2).detach()
        return out_t


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
        self.aggregator = MinConvExpLSTM(channels, channels, 3, bias=True)
        self.register_buffer(
            "_agg_state",
            torch.empty(0, channels, 1, 1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.layers, self.norms, strict=True):
            x = x + self.act(conv(norm(x)))
        return self.aggregator(x)

    def reset_cache(
        self,
        batch_size: int,
        spatial_h: int = 1,
        spatial_w: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Reset cache for all temporal conv layers and the LSTM aggregator."""
        for layer in self.layers:
            if not isinstance(layer, CausalConv3d):
                raise TypeError(f"Expected CausalConv3d, got {type(layer).__name__}")
            layer.reset_cache(batch_size, spatial_h, spatial_w, device=device, dtype=dtype)
        self._agg_state = self.aggregator.init_state(
            batch_size, spatial_h, spatial_w, device=device, dtype=dtype
        )

    def step(self, x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one frame ``(B, F, 1, H, W)`` through the stacked dilated convs.

        Returns ``(output_t, next_state)`` where ``output_t`` is ``(B, F, 1, H, W)``
        and ``next_state`` is the updated aggregator state ``(B, Hid, H, W)``.
        The aggregator state is held in ``self._agg_state`` and updated in place,
        so callers must use the returned ``next_state`` for chaining.
        """
        for conv, norm in zip(self.layers, self.norms, strict=True):
            if not isinstance(conv, CausalConv3d):
                raise TypeError(f"Expected CausalConv3d, got {type(conv).__name__}")
            out = conv.step(norm(x_t))
            x_t = x_t + self.act(out)
        x_t_4d = x_t.squeeze(2)
        out_t, self._agg_state = self.aggregator.step(x_t_4d, self._agg_state)
        out_t = out_t.unsqueeze(2)
        return out_t, self._agg_state
