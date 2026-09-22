from __future__ import annotations

from typing import cast

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from convgamer.models.io import MixerState

from .minconv import MinConvExpLSTM


class CausalConv3d(nn.Module):
    """3D convolution causal in T, symmetric in H/W.

    Temporal causality via left-only padding on depth (T) dimension.
    Spatial dims are padded symmetrically for odd kernels. Even spatial kernels
    use one extra pixel on the right/bottom to preserve the output shape.

    ``forward`` maps a whole clip ``(B, C_in, T, H, W)``.  ``step`` streams one
    frame, taking its past-frame window as an explicit ``state`` argument and
    returning the updated window — the module keeps no history of its own.
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

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.conv.bias

    def init_state(
        self,
        batch_size: int,
        spatial_h: int,
        spatial_w: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Fresh streaming state: ``pt`` zero frames ``(B, C_in, pt, H, W)``.

        ``spatial_h``/``spatial_w`` are the H/W of the **input** frame (before
        any spatial padding).  The state stores raw frames; spatial padding is
        applied per-frame in :meth:`step` so streaming matches :meth:`forward`
        exactly.
        """
        ref = self.conv.weight
        return torch.zeros(
            batch_size,
            self.conv.in_channels,
            self._pt,
            spatial_h,
            spatial_w,
            device=ref.device if device is None else device,
            dtype=ref.dtype if dtype is None else dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if any(p != 0 for p in self._pad):
            x = F.pad(x, self._pad)
        return self.conv(x)

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one frame ``(B, C_in, 1, H, W)``; returns ``(out_t, next_state)``.

        ``state`` is the sliding window of ``pt`` past **raw** frames
        (``init_state`` output).  The current frame is spatially padded, appended
        to the window, and the convolution runs over the whole padded window —
        mirroring :meth:`forward` exactly.  ``next_state`` drops the oldest
        frame and appends the current one (detached: cache history is not a
        gradient path).
        """
        if self._pt == 0:
            # Temporal extent 1: the output at t depends on frame t alone, so
            # the window is empty and travels through untouched.
            return self.conv(x_t), state

        b, c_in, _, h, w = x_t.shape

        # Apply spatial padding to current frame only (symmetric on H, W).
        # Temporal left padding is supplied by the state window.
        sp_pad = (self._pad[0], self._pad[1], self._pad[2], self._pad[3])
        x_t_padded = F.pad(x_t, sp_pad + (0, 0)) if any(p != 0 for p in sp_pad) else x_t

        if state.shape != (b, c_in, self._pt, h, w):
            raise RuntimeError(
                f"Cache mismatch: expected (B, {c_in}, {self._pt}, {h}, {w}), "
                f"got {tuple(state.shape)}"
            )

        # Pad the window spatially too (so cached frames see the same border behavior)
        state_padded = F.pad(state, sp_pad + (0, 0)) if any(p != 0 for p in sp_pad) else state

        # Full window: [pt padded past frames, 1 padded current frame]
        window = torch.cat([state_padded, x_t_padded], dim=2)  # (B, C, pt+1, hp, wp)
        # Conv with padding=0 (no temporal pad — the window supplies history)
        out = self.conv(window)
        out_t = out[:, :, -1:]  # (B, C_out, 1, H_out, W_out)

        # Slide the window: drop the oldest frame, store the raw (unpadded) frame.
        next_state = torch.cat([state[:, :, 1:], x_t], dim=2).detach()
        return out_t, next_state


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.layers, self.norms, strict=True):
            x = x + self.act(conv(norm(x)))
        return self.aggregator(x)

    def init_state(
        self,
        batch_size: int,
        spatial_h: int = 1,
        spatial_w: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> MixerState:
        """Fresh streaming state: one window per dilated conv plus the LSTM hidden."""
        return MixerState(
            conv=tuple(
                cast(CausalConv3d, conv).init_state(
                    batch_size, spatial_h, spatial_w, device=device, dtype=dtype
                )
                for conv in self.layers
            ),
            aggregator=self.aggregator.init_state(
                batch_size, spatial_h, spatial_w, device=device, dtype=dtype
            ),
        )

    def step(self, x_t: torch.Tensor, state: MixerState) -> tuple[torch.Tensor, MixerState]:
        """Stream one frame ``(B, F, 1, H, W)``; returns ``(output_t, next_state)``.

        Both the dilated-conv windows and the aggregator hidden are carried in
        ``state``; nothing is written back onto the module.
        """
        windows: list[torch.Tensor] = []
        for conv, norm in zip(self.layers, self.norms, strict=True):
            conv_c = cast(CausalConv3d, conv)
            out, window = conv_c.step(norm(x_t), state.conv[len(windows)])
            x_t = x_t + self.act(out)
            windows.append(window)
        out_t, hidden = self.aggregator.step(x_t.squeeze(2), state.aggregator)
        return out_t.unsqueeze(2), MixerState(conv=tuple(windows), aggregator=hidden)
