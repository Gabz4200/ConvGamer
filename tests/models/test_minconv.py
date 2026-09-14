"""Behavior tests for MinConvLSTM and MinConvExpLSTM.

Seams under test:

  - parallel ``forward(x: B,C,T,H,W) -> B,Hid,T,H,W`` (prefix-scan mapping,
    each output frame mixes the current and all previous input frames);
  - streaming ``init_state`` + ``step(x_t: B,C,H,W, h) -> (y_t, h')`` for
    inference-only O(1)-memory rollout.

Contracts: parallel/streaming parity, temporal causality, hand-worked gate
literals from the paper equations, Exp gate identity ``i_hat = 1 - f_hat``,
finite outputs, gradient flow, input validation.
"""

import pytest
import torch

from convgamer.models.convgamer.blocks import MinConvExpLSTM, MinConvLSTM


def _fix_gates_lstm(
    op: MinConvLSTM, forget_bias: float, input_bias: float, cand_bias: float
) -> None:
    fb, ib, hb = op.conv_f.bias, op.conv_i.bias, op.conv_h.bias
    assert fb is not None and ib is not None and hb is not None
    with torch.no_grad():
        for conv in (op.conv_f, op.conv_i, op.conv_h):
            torch.nn.init.zeros_(conv.weight)
            assert conv.bias is not None
            torch.nn.init.zeros_(conv.bias)
        torch.nn.init.constant_(fb, forget_bias)
        torch.nn.init.constant_(ib, input_bias)
        torch.nn.init.constant_(hb, cand_bias)


def _fix_gates_exp(
    op: MinConvExpLSTM, forget_bias: float, input_bias: float, cand_bias: float
) -> None:
    fb, ib, hb = op.conv_f.bias, op.conv_i.bias, op.conv_h.bias
    assert fb is not None and ib is not None and hb is not None
    with torch.no_grad():
        for conv in (op.conv_f, op.conv_i, op.conv_h):
            torch.nn.init.zeros_(conv.weight)
            assert conv.bias is not None
            torch.nn.init.zeros_(conv.bias)
        torch.nn.init.constant_(fb, forget_bias)
        torch.nn.init.constant_(ib, input_bias)
        torch.nn.init.constant_(hb, cand_bias)


# ── MinConvLSTM ─────────────────────────────────────────────────────────────


def test_when_valid_clip_then_parallel_output_shape() -> None:
    op = MinConvLSTM(in_channels=3)
    out = op(torch.randn(2, 3, 5, 6, 7))
    assert out.shape == (2, 3, 5, 6, 7)


def test_when_custom_hidden_then_output_channels_follow() -> None:
    op = MinConvLSTM(in_channels=3, hidden_channels=8)
    out = op(torch.randn(2, 3, 4, 5, 6))
    assert out.shape == (2, 8, 4, 5, 6)


def test_when_zero_bias_gates_then_hand_worked_prefix_values() -> None:
    """f = i = sigmoid(0) = 0.5, normalized to (0.5, 0.5), h_tilde = 2.

    h_0 = 0.5 * 2 = 1.0, h_1 = 0.5 * 1.0 + 0.5 * 2 = 1.5.
    """
    op = MinConvLSTM(in_channels=2).double()
    _fix_gates_lstm(op, forget_bias=0.0, input_bias=0.0, cand_bias=2.0)
    out = op(torch.zeros(1, 2, 2, 3, 3, dtype=torch.float64))
    torch.testing.assert_close(
        out[0, :, :, 0, 0],
        torch.tensor([[1.0, 1.5], [1.0, 1.5]], dtype=torch.float64),
    )


def test_when_parallel_then_streaming_parity_lstm() -> None:
    torch.manual_seed(0)
    op = MinConvLSTM(in_channels=3, hidden_channels=4).double()
    x = torch.randn(2, 3, 6, 5, 5, dtype=torch.float64)
    parallel = op(x)
    h = op.init_state(2, 5, 5, dtype=torch.float64)
    ys: list[torch.Tensor] = []
    for t in range(x.shape[2]):
        y, h = op.step(x[:, :, t], h)
        ys.append(y)
    streamed = torch.stack(ys, dim=2)
    torch.testing.assert_close(streamed, parallel)
    torch.testing.assert_close(h, parallel[:, :, -1])


def test_when_future_perturbed_then_past_unchanged_lstm() -> None:
    torch.manual_seed(0)
    op = MinConvLSTM(in_channels=2)
    x1 = torch.randn(1, 2, 6, 4, 4)
    x2 = x1.clone()
    x2[:, :, 4:] += 5.0
    y1, y2 = op(x1), op(x2)
    torch.testing.assert_close(y1[:, :, :4], y2[:, :, :4])


def test_when_lstm_forward_then_finite_and_grad_flows() -> None:
    torch.manual_seed(0)
    op = MinConvLSTM(in_channels=2, hidden_channels=3)
    x = torch.randn(1, 2, 4, 4, 4, requires_grad=True)
    out = op(x)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert op.conv_f.weight.grad is not None
    assert torch.isfinite(op.conv_f.weight.grad).all()


def test_when_bad_input_then_lstm_raises() -> None:
    op = MinConvLSTM(in_channels=2)
    with pytest.raises(ValueError, match="5D"):
        op(torch.randn(1, 2, 4, 4))
    with pytest.raises(ValueError, match="in_channels"):
        op(torch.randn(1, 3, 4, 4, 4))
    with pytest.raises(ValueError, match="4D"):
        op.step(torch.randn(1, 2, 4, 4, 4), op.init_state(1, 4, 4))


# ── MinConvExpLSTM ──────────────────────────────────────────────────────────


def test_when_valid_clip_then_exp_output_shape() -> None:
    op = MinConvExpLSTM(in_channels=3)
    out = op(torch.randn(2, 3, 5, 6, 7))
    assert out.shape == (2, 3, 5, 6, 7)


def test_when_zero_bias_gates_then_exp_hand_worked_prefix_values() -> None:
    """d = 0 - 0 = 0, f_hat = sigmoid(0) = 0.5, i_hat = 0.5, h_tilde = 2."""
    op = MinConvExpLSTM(in_channels=2).double()
    _fix_gates_exp(op, forget_bias=0.0, input_bias=0.0, cand_bias=2.0)
    out = op(torch.zeros(1, 2, 2, 3, 3, dtype=torch.float64))
    torch.testing.assert_close(
        out[0, :, :, 0, 0],
        torch.tensor([[1.0, 1.5], [1.0, 1.5]], dtype=torch.float64),
    )


def test_when_exp_gates_then_input_is_one_minus_forget() -> None:
    torch.manual_seed(1)
    op = MinConvExpLSTM(in_channels=2, hidden_channels=3)
    x = torch.randn(1, 2, 4, 4, 4)
    f_hat, i_hat = op.gates(x[:, :, 0])
    torch.testing.assert_close(i_hat, 1.0 - f_hat)
    torch.testing.assert_close(f_hat + i_hat, torch.ones_like(f_hat))


def test_when_parallel_then_streaming_parity_exp() -> None:
    torch.manual_seed(2)
    op = MinConvExpLSTM(in_channels=3, hidden_channels=4).double()
    x = torch.randn(2, 3, 6, 5, 5, dtype=torch.float64)
    parallel = op(x)
    h = op.init_state(2, 5, 5, dtype=torch.float64)
    ys: list[torch.Tensor] = []
    for t in range(x.shape[2]):
        y, h = op.step(x[:, :, t], h)
        ys.append(y)
    streamed = torch.stack(ys, dim=2)
    torch.testing.assert_close(streamed, parallel)


def test_when_future_perturbed_then_past_unchanged_exp() -> None:
    torch.manual_seed(3)
    op = MinConvExpLSTM(in_channels=2)
    x1 = torch.randn(1, 2, 6, 4, 4)
    x2 = x1.clone()
    x2[:, :, 4:] += 5.0
    torch.testing.assert_close(op(x1)[:, :, :4], op(x2)[:, :, :4])


def test_when_exp_forward_then_finite_and_grad_flows() -> None:
    torch.manual_seed(4)
    op = MinConvExpLSTM(in_channels=2, hidden_channels=3)
    x = torch.randn(1, 2, 4, 4, 4, requires_grad=True)
    out = op(x)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert op.conv_f.weight.grad is not None
    assert torch.isfinite(op.conv_f.weight.grad).all()
