"""Behavior tests for the slim multi-level V-JEPA predictor and loss.

Seams tested:
- ``VJEPAPredictor.forward``: fused (B, F, T, H, W) in, per-level
  (B, L, F, T, H, W) out.
- ``JEPALoss``: computes L_predict + L_ctx with stop-gradient and
  distance-weighted lambda, averaged over levels.
- ``EMAEncoder``: shadow weights track source encoder.
"""

from __future__ import annotations

import pytest
import torch

from convgamer.models.jepa import JEPALoss, VJEPAPredictor
from convgamer.modules.ema import EMAEncoder


def test_predictor_outputs_mask_and_context_tokens() -> None:
    """Predictor must output predictions for every token position, not just masked."""
    B, F, T, H, W = 2, 32, 8, 8, 8
    feat = torch.randn(B, F, T, H, W)
    mask = torch.ones(B, T, H, W, dtype=torch.bool)  # all masked
    mask[:, : T // 2] = False  # half visible, half masked

    predictor = VJEPAPredictor(
        feature_dim=F,
        predictor_dim=64,
        num_layers=2,
    )
    out = predictor(feat, mask)
    # One prediction per level: (B, L, F, T, H, W).
    assert out.shape == (B, 4, F, T, H, W)
    assert torch.isfinite(out).all()


def test_predictor_rejects_non_5d_input() -> None:
    predictor = VJEPAPredictor(feature_dim=16, predictor_dim=16, num_layers=1)
    with pytest.raises(ValueError, match="5D"):
        predictor(torch.randn(2, 16, 4, 4), torch.zeros(2, 4, 4, 4, dtype=torch.bool))


def test_predictor_rejects_bad_num_outputs() -> None:
    with pytest.raises(ValueError, match="num_outputs"):
        VJEPAPredictor(num_outputs=0)


def test_loss_stop_grad_on_target() -> None:
    """Target encoder output must not receive gradients through the loss."""
    B, F, T, H, W = 1, 16, 4, 4, 4
    feat = torch.randn(B, F, T, H, W, requires_grad=True)
    target = torch.randn(B, F, T, H, W, requires_grad=True)

    loss_fn = JEPALoss()
    loss = loss_fn(feat, target, mask=torch.zeros(B, T, H, W, dtype=torch.bool))
    loss.backward()

    # feat should get gradients (predictor side)
    assert feat.grad is not None
    # target should NOT get gradients (stop-gradient)
    assert target.grad is None


def test_loss_zero_when_all_visible() -> None:
    """With no masked tokens, only L_ctx applies on all positions."""
    B, F, T, H, W = 1, 8, 2, 4, 4
    feat = torch.randn(B, F, T, H, W)
    target = feat.clone()  # perfect prediction -> loss should be ~0
    mask = torch.zeros(B, T, H, W, dtype=torch.bool)

    loss_fn = JEPALoss(lambda_base=0.5)
    loss = loss_fn(feat, target, mask=mask)
    assert loss.item() < 1e-5


def test_loss_level_average_matches_per_level_mean() -> None:
    """6D loss equals the mean of per-level 5D losses."""
    B, L, F, T, H, W = 1, 4, 8, 4, 4, 4
    torch.manual_seed(0)
    pred = torch.randn(B, L, F, T, H, W, requires_grad=True)
    target = torch.randn(B, L, F, T, H, W)
    mask = torch.zeros(B, T, H, W, dtype=torch.bool)
    mask[0, :, :2, :2] = True

    loss_fn = JEPALoss()
    multi = loss_fn(pred, target, mask)
    per_level = torch.stack([loss_fn(pred[:, i], target[:, i], mask) for i in range(L)]).mean()
    torch.testing.assert_close(multi, per_level, atol=1e-6, rtol=1e-6)


def test_loss_rejects_shape_mismatch() -> None:
    loss_fn = JEPALoss()
    pred = torch.randn(1, 4, 8, 4, 4, 4)
    target = torch.randn(1, 4, 8, 4, 4, 4)
    mask = torch.zeros(1, 4, 4, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="must match target"):
        loss_fn(pred, target[:, :, :, :2], mask)


def test_predictor_mask_tokens_change_output() -> None:
    """Masked positions must not echo the input: mask tokens take effect."""
    B, F, T, H, W = 1, 16, 4, 4, 4
    feat = torch.randn(B, F, T, H, W)
    predictor = VJEPAPredictor(feature_dim=F, predictor_dim=16, num_layers=1)
    visible = torch.zeros(B, T, H, W, dtype=torch.bool)
    masked = torch.ones(B, T, H, W, dtype=torch.bool)
    out_visible = predictor(feat, visible)
    out_masked = predictor(feat, masked)
    assert not torch.allclose(out_visible, out_masked)


def test_loss_all_masked_stays_finite() -> None:
    """Fully masked batch exercises the n_ctx clamp without NaN."""
    B, F, T, H, W = 1, 8, 2, 4, 4
    pred = torch.randn(B, F, T, H, W)
    target = torch.randn(B, F, T, H, W)
    mask = torch.ones(B, T, H, W, dtype=torch.bool)
    loss = JEPALoss()(pred, target, mask)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_ema_encoder_shadows_source() -> None:
    """EMA encoder weights must be copies that diverge as source trains."""
    from convgamer.models.convgamer.encoder import ConvGamerEncoder

    encoder = ConvGamerEncoder(input_dim=3, hidden_dim=8, num_layers=1, level_dim=8)
    ema = EMAEncoder(module=encoder, decay=0.999)

    # Initially identical
    with torch.no_grad():
        for p_src, p_tgt in zip(encoder.parameters(), ema.shadow_params, strict=True):
            torch.testing.assert_close(p_tgt, p_src)

    # Source changes
    first = next(encoder.parameters())
    original = first.data.clone()
    with torch.no_grad():
        first.fill_(99.0)
    ema.update()

    # EMA should be close to original (high decay = slow tracking)
    moved = ema.shadow_params[0]
    assert not torch.allclose(moved, first.data)
    # With decay=0.999, shadow = 0.999*old + 0.001*99, so moves toward 99 but stays near old
    assert torch.allclose(moved, original * 0.999 + 0.001 * 99.0, atol=1e-5)
