"""Behavior tests for multi-level V-JEPA 2.1 prediction.

Seams tested:
- ``VJEPAPredictor`` with ``num_levels>1`` returns ``(B, L, dim, T, H, W)``
  with cross-level FFN interaction.
- ``num_levels=1`` keeps the legacy ``(B, dim, T, H, W)`` layout.
- ``num_heads`` controls attention heads inside the Transformer (default 4).
- ``JEPALoss`` accepts multi-level predictions, stop-grading the target.
- ``MultiHeadCausalTemporalMixer`` applies mixers per level and streams with
  parity against the parallel ``forward``.
"""

from __future__ import annotations

import torch

from convgamer.models.convgamer.causal import (
    CausalTemporalMixer,
    MultiHeadCausalTemporalMixer,
)
from convgamer.models.jepa import JEPALoss, VJEPAPredictor


def test_predictor_single_level_layout() -> None:
    """num_levels=1 must keep the legacy (B, dim, T, H, W) output layout."""
    B, F, T, H, W = 2, 32, 8, 8, 8
    feat = torch.randn(B, F, T, H, W)
    mask = torch.zeros(B, T, H, W, dtype=torch.bool)
    mask[0, :, : T // 2] = True

    predictor = VJEPAPredictor(
        feature_dim=F,
        predictor_dim=64,
        num_layers=2,
        num_levels=1,
        num_heads=4,
    )
    out = predictor(feat, mask)
    assert out.shape == (B, F, T, H, W)


def test_predictor_multi_level_shape_and_cross_level_ffn() -> None:
    """num_levels>1 gains a level axis at position 1 with cross-level FFN."""
    B, F, T, H, W = 2, 32, 8, 8, 8
    feat = torch.randn(B, 3, F, T, H, W)  # 6D: (B, num_levels, F, T, H, W)
    mask = torch.zeros(B, T, H, W, dtype=torch.bool)
    mask[0, :, : T // 2] = True

    predictor = VJEPAPredictor(
        feature_dim=F,
        predictor_dim=64,
        num_layers=2,
        num_levels=3,
        num_heads=4,
    )
    out = predictor(feat, mask)
    assert out.shape == (B, 3, F, T, H, W)

    # Cross-level FFN exists and has parameters
    assert hasattr(predictor, "level_ffn")
    assert list(predictor.level_ffn.parameters())


def test_predictor_multi_level_each_level_runs() -> None:
    """Every level must produce finite output for its slice."""
    B, F, T, H, W = 1, 16, 4, 4, 4
    feat = torch.randn(B, 4, F, T, H, W)  # (B, num_levels, F, T, H, W)
    mask = torch.ones(B, T, H, W, dtype=torch.bool)
    mask[0, : T // 2] = False

    predictor = VJEPAPredictor(
        feature_dim=F,
        predictor_dim=16,
        num_layers=2,
        num_levels=4,
        num_heads=4,
    )
    out = predictor(feat, mask)
    assert out.shape == (B, 4, F, T, H, W)
    assert torch.isfinite(out).all()


def test_predictor_attention_heads_default() -> None:
    """num_heads defaults to 4 (reasonable for Transformer)."""
    predictor = VJEPAPredictor(feature_dim=32, predictor_dim=64, num_layers=2)
    assert predictor.num_heads == 4
    assert predictor.num_levels == 1


def test_predictor_rejects_bad_num_levels() -> None:
    """num_levels<1 is a programming error."""
    try:
        VJEPAPredictor(num_levels=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for num_levels=0")


def test_predictor_rejects_bad_attention_heads() -> None:
    """num_heads<1 is a programming error."""
    try:
        VJEPAPredictor(num_heads=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for num_heads=0")


def test_predictor_rejects_non_divisible_dim() -> None:
    """predictor_dim must be divisible by num_heads."""
    try:
        VJEPAPredictor(predictor_dim=65, num_heads=4)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-divisible dim")


def test_loss_accepts_multi_level_predictions() -> None:
    """Multi-level predictions yield a scalar loss; target stop-graded."""
    B, F, T, H, W = 1, 16, 4, 4, 4
    pred = torch.randn(B, 3, F, T, H, W, requires_grad=True)
    target = torch.randn(B, F, T, H, W, requires_grad=True)
    mask = torch.zeros(B, T, H, W, dtype=torch.bool)
    mask[0, :, : T // 2] = True

    loss_fn = JEPALoss(feature_dim=F, num_levels=3)
    loss = loss_fn(pred, target, mask)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()

    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert target.grad is None


def test_loss_multi_level_matches_single_level_average() -> None:
    """Multi-level loss averaged over levels equals mean of single-level losses."""
    B, F, T, H, W = 1, 8, 4, 4, 4
    mask = torch.zeros(B, T, H, W, dtype=torch.bool)
    mask[0, :, : T // 2] = True
    torch.manual_seed(0)
    target = torch.randn(B, F, T, H, W)

    preds = [torch.randn(B, F, T, H, W, requires_grad=True) for _ in range(3)]

    loss_fn = JEPALoss(feature_dim=F, num_levels=3)
    stacked = torch.stack(preds, dim=1)
    multi_loss = loss_fn(stacked, target, mask)

    per_head = torch.stack([loss_fn(p, target, mask) for p in preds]).mean()
    assert torch.allclose(multi_loss, per_head, atol=1e-6, rtol=1e-6)


def test_loss_rejects_level_mismatch() -> None:
    """Multi-level pred with wrong level axis size must fail loudly."""
    B, F, T, H, W = 1, 8, 4, 4, 4
    pred = torch.randn(B, 2, F, T, H, W)  # wrong level count
    target = torch.randn(B, F, T, H, W)
    mask = torch.zeros(B, T, H, W, dtype=torch.bool)

    loss_fn = JEPALoss(feature_dim=F, num_levels=3)
    try:
        loss_fn(pred, target, mask)
    except ValueError:
        return
    raise AssertionError("expected ValueError for level axis mismatch")


def test_multilevel_mixer_forward_shape() -> None:
    """Multi-head mixer preserves the (B, H, C, T, H, W) layout."""
    B, H, C, T, S = 2, 3, 8, 6, 5
    x = torch.randn(B, H, C, T, S, S)
    mixer = MultiHeadCausalTemporalMixer(C, num_heads=H)
    out = mixer(x)
    assert out.shape == x.shape


def test_multilevel_mixer_streaming_parity() -> None:
    """step() one frame at a time must match forward() over the full sequence."""
    torch.manual_seed(0)
    s = MultiHeadCausalTemporalMixer(8, num_heads=2)
    inp = torch.randn(1, 2, 8, 7, 5, 5)
    par = s(inp)

    st = s.init_state(1, 5, 5)
    outs: list[torch.Tensor] = []
    for i in range(7):
        frame = inp[:, :, :, i : i + 1]
        o, st = s.step(frame, st)
        outs.append(o)
    stepy = torch.cat(outs, dim=3)
    torch.testing.assert_close(par, stepy, atol=1e-4, rtol=1e-4)


def test_multilevel_mixer_state_is_explicit() -> None:
    """step() carries history in its argument: module buffers state externally."""
    s = MultiHeadCausalTemporalMixer(4, num_heads=2)
    st = s.init_state(1, 4, 4)
    x1 = torch.randn(1, 2, 4, 1, 4, 4)
    x2 = torch.randn(1, 2, 4, 1, 4, 4)
    _, ns1 = s.step(x1, st)
    _, ns2 = s.step(x2, st)
    assert ns1 is not ns2
    assert not any("state" in k.lower() for k in dict(s.named_buffers()))
    single = CausalTemporalMixer(4)
    inp = torch.randn(1, 4, 6, 4, 4)
    par = single(inp)
    st = single.init_state(1, 4, 4)
    outs = []
    for i in range(6):
        o, st = single.step(inp[:, :, i : i + 1], st)
        outs.append(o)
    torch.testing.assert_close(par, torch.cat(outs, dim=2), atol=1e-4, rtol=1e-4)
