from __future__ import annotations

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


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
