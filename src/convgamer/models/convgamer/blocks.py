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
            assert isinstance(layer, CausalConv3d)
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
            assert isinstance(conv, CausalConv3d)
            out = conv.step(norm(x_t))
            x_t = x_t + self.act(out)
        x_t_4d = x_t.squeeze(2)
        out_t, self._agg_state = self.aggregator.step(x_t_4d, self._agg_state)
        out_t = out_t.unsqueeze(2)
        return out_t, self._agg_state


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

        # Repeat channels, repeat_interleave produces [C0, C0, C0, C1, C1, C1, ..., Cn, Cn, Cn].
        tiled_x_down = x_down.repeat_interleave(self.out_factor, dim=1)

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

    def reset_cache(
        self,
        batch_size: int,
        h: int,
        w: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Reset temporal cache for the correction conv layers."""
        self.correction_conv.reset_cache(batch_size, h, w, device=device, dtype=dtype)
        self.out_correction_conv.reset_cache(batch_size, h, w, device=device, dtype=dtype)

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        """Stream one frame ``(B, C, 1, H, W)`` through the downsampler.

        Matches ``forward`` frame-by-frame: spatial resize, causal correction
        conv with cached history, channel repeat, residual, norm.
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

        tiled_x_down = x_down.repeat_interleave(self.out_factor, dim=1)

        # Correction branch: step through cached convs
        correct = self.correction_conv.step(x_t)
        correct = self.act(correct)
        correct = einops.rearrange(correct, "b c t h w -> (b t) c h w")
        correct = F.adaptive_avg_pool2d(correct, output_size=(h_out, w_out))
        correct = einops.rearrange(correct, "(b t) c h w -> b c t h w", b=b, t=1)
        correct = self.out_correction_conv.step(correct)

        out = tiled_x_down + correct
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

    def reset_cache(
        self,
        batch_size: int,
        h: int,
        w: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Reset temporal cache for all conv layers in the stem."""
        for conv in [self.near_conv, self.far_conv, self.local_conv, self.fuse_conv]:
            conv.reset_cache(batch_size, h, w, device=device, dtype=dtype)

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        """Stream one frame ``(B, C, 1, H, W)`` through the stem."""
        near = self.near_conv.step(x_t)
        local = self.local_conv.step(x_t)
        far = self.far_conv.step(x_t)
        new = torch.cat((near, local, far), dim=1)
        new = self.act(new)
        new = self.fuse_conv.step(new)
        x = torch.cat((x_t, new), dim=1)
        if self.use_softmax:
            x = torch.cat((x, spatial_softmax(x)), dim=1)
        return self.norm(x)


def _linear_prefix_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inclusive scan of ``v_t = a_t * v_{t-1} + b_t`` with ``v_{-1} = 0``.

    Hillis-Steele doubling: ``log2(T)`` vectorized steps over the time axis,
    no Python loop over frames. Inputs are ``(B, C, T, H, W)``; returns the
    full prefix sequence with the same shape.
    """
    if a.ndim != 5 or b.ndim != 5:
        raise ValueError(f"Expected 5D (B, C, T, H, W) scan inputs, got {a.ndim}D/{b.ndim}D")
    if a.shape != b.shape:
        raise ValueError(f"Scan inputs must share shape, got {a.shape} vs {b.shape}")
    t = a.shape[2]
    if t <= 0:
        raise ValueError(f"Scan sequence length must be > 0, got {t}")

    a_cur, b_cur = a, b
    stride = 1
    while stride < t:
        a_tail = a_cur[:, :, stride:] * a_cur[:, :, : t - stride]
        b_tail = b_cur[:, :, stride:] + a_cur[:, :, stride:] * b_cur[:, :, : t - stride]
        a_cur = torch.cat([a_cur[:, :, :stride], a_tail], dim=2)
        b_cur = torch.cat([b_cur[:, :, :stride], b_tail], dim=2)
        stride *= 2
    return b_cur


def _normalized_sigmoid_gates(
    forget_logits: torch.Tensor, input_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable ``f / (f + i)`` normalization via log-domain subtraction."""
    log_f = -F.softplus(-forget_logits)
    log_i = -F.softplus(-input_logits)
    denom = torch.logaddexp(log_f, log_i)
    return torch.exp(log_f - denom), torch.exp(log_i - denom)


class MinConvLSTM(nn.Module):
    """Minimal convolutional LSTM with parallel and streaming forms.

    Recurrence per location: ``h^t = f_hat^t * h^{t-1} + i_hat^t * h_tilde^t``
    with ``f_hat, i_hat = f / (f + i), i / (f + i)`` from
    ``f = sigmoid(Conv_f[x^t])``, ``i = sigmoid(Conv_i[x^t])``,
    ``h_tilde = Conv_h[x^t]``. Gates see only the current frame, so the
    recurrence is linear in ``h`` and parallelizes as a prefix scan.
    ``forward`` maps ``(B, C, T, H, W)`` to ``(B, Hid, T, H, W)``;
    ``init_state``/``step`` roll one ``(B, C, H, W)`` frame at a time.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None = None,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")
        hidden = in_channels if hidden_channels is None else hidden_channels
        if hidden <= 0:
            raise ValueError(f"hidden_channels must be > 0, got {hidden}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd int, got {kernel_size}")
        self.in_channels = in_channels
        self.hidden_channels = hidden
        self.kernel_size = kernel_size
        pad = kernel_size // 2
        self.conv_f = nn.Conv2d(in_channels, hidden, kernel_size, padding=pad, bias=bias)
        self.conv_i = nn.Conv2d(in_channels, hidden, kernel_size, padding=pad, bias=bias)
        self.conv_h = nn.Conv2d(in_channels, hidden, kernel_size, padding=pad, bias=bias)
        self._init_gates()

    def _init_gates(self) -> None:
        """Forget gate starts open (bias=1), input gate starts low (bias=0),
        candidate gate zeroed — standard LSTM init for long-sequence stability."""
        if self.conv_f.bias is not None:
            nn.init.constant_(self.conv_f.bias, 1.0)
        if self.conv_i.bias is not None:
            nn.init.zeros_(self.conv_i.bias)

    def gates(self, x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized forget/input gates for one frame."""
        if x_t.ndim != 4:
            raise ValueError(f"Expected 4D frame (B, C, H, W), got {x_t.ndim}D")
        if x_t.shape[1] != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {x_t.shape[1]}")
        return _normalized_sigmoid_gates(self.conv_f(x_t), self.conv_i(x_t))

    def init_state(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Zero hidden state ``(B, Hid, H, W)`` for streaming rollout."""
        if batch_size <= 0 or height <= 0 or width <= 0:
            raise ValueError(f"batch, height, width must be > 0, got {(batch_size, height, width)}")
        ref = self.conv_h.weight
        return torch.zeros(
            batch_size,
            self.hidden_channels,
            height,
            width,
            device=ref.device if device is None else device,
            dtype=ref.dtype if dtype is None else dtype,
        )

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one frame: returns ``(output_t, next_state)`` (identical)."""
        if x_t.ndim != 4:
            raise ValueError(f"Expected 4D frame (B, C, H, W), got {x_t.ndim}D")
        if x_t.shape[1] != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {x_t.shape[1]}")
        if state.ndim != 4 or state.shape[1] != self.hidden_channels:
            raise ValueError(
                f"Expected state (B, {self.hidden_channels}, H, W), got {tuple(state.shape)}"
            )
        if state.shape[0] != x_t.shape[0] or state.shape[2:] != x_t.shape[2:]:
            raise ValueError(
                f"State {tuple(state.shape)} must match frame batch/spatial {tuple(x_t.shape)}"
            )
        f_hat, i_hat = self.gates(x_t)
        h_tilde = self.conv_h(x_t)
        next_state = f_hat * state + i_hat * h_tilde
        return next_state, next_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel prefix-scan mapping over the full clip."""
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (B, C, T, H, W), got shape {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {x.shape[1]}")
        if x.shape[2] <= 0:
            raise ValueError(f"Temporal dim must be > 0, got {x.shape[2]}")
        b, _c, t, h, w = x.shape
        frames = einops.rearrange(x, "b c t h w -> (b t) c h w")
        f_hat, i_hat = _normalized_sigmoid_gates(self.conv_f(frames), self.conv_i(frames))
        h_tilde = self.conv_h(frames)
        f_hat = einops.rearrange(f_hat, "(b t) c h w -> b c t h w", b=b, t=t)
        i_hat = einops.rearrange(i_hat, "(b t) c h w -> b c t h w", b=b, t=t)
        h_tilde = einops.rearrange(h_tilde, "(b t) c h w -> b c t h w", b=b, t=t)
        return _linear_prefix_scan(f_hat, i_hat * h_tilde)


class MinConvExpLSTM(nn.Module):
    """MinConvLSTM with single-sigmoid exponential gating.

    Recurrence per location: ``h^t = f_hat^t * h^{t-1} + i_hat^t * h_tilde^t``
    with ``f_hat = sigmoid(Conv_f[x^t] - Conv_i[x^t])`` and
    ``i_hat = 1 - f_hat``. Same parallel ``forward`` and streaming
    ``init_state``/``step`` seams as ``MinConvLSTM``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None = None,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")
        hidden = in_channels if hidden_channels is None else hidden_channels
        if hidden <= 0:
            raise ValueError(f"hidden_channels must be > 0, got {hidden}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd int, got {kernel_size}")
        self.in_channels = in_channels
        self.hidden_channels = hidden
        self.kernel_size = kernel_size
        pad = kernel_size // 2
        self.conv_f = nn.Conv2d(in_channels, hidden, kernel_size, padding=pad, bias=bias)
        self.conv_i = nn.Conv2d(in_channels, hidden, kernel_size, padding=pad, bias=bias)
        self.conv_h = nn.Conv2d(in_channels, hidden, kernel_size, padding=pad, bias=bias)
        self._init_gates()

    def _init_gates(self) -> None:
        """Bias conv_f positive, conv_i zero so the forget gate starts open,
        keeping the cell memory available for long sequences."""
        if self.conv_f.bias is not None:
            nn.init.constant_(self.conv_f.bias, 1.0)
        if self.conv_i.bias is not None:
            nn.init.zeros_(self.conv_i.bias)

    def gates(self, x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forget gate and its complement for one frame."""
        if x_t.ndim != 4:
            raise ValueError(f"Expected 4D frame (B, C, H, W), got {x_t.ndim}D")
        if x_t.shape[1] != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {x_t.shape[1]}")
        f_hat = torch.sigmoid(self.conv_f(x_t) - self.conv_i(x_t))
        return f_hat, 1.0 - f_hat

    def init_state(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Zero hidden state ``(B, Hid, H, W)`` for streaming rollout."""
        if batch_size <= 0 or height <= 0 or width <= 0:
            raise ValueError(f"batch, height, width must be > 0, got {(batch_size, height, width)}")
        ref = self.conv_h.weight
        return torch.zeros(
            batch_size,
            self.hidden_channels,
            height,
            width,
            device=ref.device if device is None else device,
            dtype=ref.dtype if dtype is None else dtype,
        )

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one frame: returns ``(output_t, next_state)`` (identical)."""
        if x_t.ndim != 4:
            raise ValueError(f"Expected 4D frame (B, C, H, W), got {x_t.ndim}D")
        if x_t.shape[1] != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {x_t.shape[1]}")
        if state.ndim != 4 or state.shape[1] != self.hidden_channels:
            raise ValueError(
                f"Expected state (B, {self.hidden_channels}, H, W), got {tuple(state.shape)}"
            )
        if state.shape[0] != x_t.shape[0] or state.shape[2:] != x_t.shape[2:]:
            raise ValueError(
                f"State {tuple(state.shape)} must match frame batch/spatial {tuple(x_t.shape)}"
            )
        f_hat, i_hat = self.gates(x_t)
        h_tilde = self.conv_h(x_t)
        next_state = f_hat * state + i_hat * h_tilde
        return next_state, next_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel prefix-scan mapping over the full clip."""
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (B, C, T, H, W), got shape {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {x.shape[1]}")
        if x.shape[2] <= 0:
            raise ValueError(f"Temporal dim must be > 0, got {x.shape[2]}")
        b, _c, t, h, w = x.shape
        frames = einops.rearrange(x, "b c t h w -> (b t) c h w")
        f_hat = torch.sigmoid(self.conv_f(frames) - self.conv_i(frames))
        i_hat = 1.0 - f_hat
        h_tilde = self.conv_h(frames)
        f_hat = einops.rearrange(f_hat, "(b t) c h w -> b c t h w", b=b, t=t)
        i_hat = einops.rearrange(i_hat, "(b t) c h w -> b c t h w", b=b, t=t)
        h_tilde = einops.rearrange(h_tilde, "(b t) c h w -> b c t h w", b=b, t=t)
        return _linear_prefix_scan(f_hat, i_hat * h_tilde)
