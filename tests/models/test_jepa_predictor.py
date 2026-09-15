"""Behavior tests for V-JEPA 2.1 predictor and loss components.

Seams tested:
- ``VJEPAPredictor.forward``: produces dense mask predictions + per-level
  predictions with correct shape.
- ``JEPALoss``: computes L_predict + L_ctx with stop-gradient and
  distance-weighted lambda.
- ``EMAEncoder``: shadow weights track source encoder.
"""

from __future__ import annotations

import torch
from torch import nn

from convgamer.models.jepa import JEPALoss, VJEPAPredictor
from convgamer.models.jepa.ema import EMAEncoder


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
        num_levels=1,
    )
    out = predictor(feat, mask)
    # Dense prediction preserves batch/time/space: (B, F, T, H, W).
    assert out.shape == (B, F, T, H, W)


def test_predictor_context_loss_weighting_decays_with_distance() -> None:
    """Distance-weighted lambda: closer to mask = higher weight."""
    mask = torch.zeros(1, 4, 4, 4, dtype=torch.bool)
    mask[0, 1, :, :] = True  # mask middle frame

    loss_fn = JEPALoss(feature_dim=16)
    lambdas = loss_fn.compute_context_lambdas(mask, lambda_base=0.5)

    # Masked frame (index 1) has lambda 0 (no ctx loss on masked)
    assert lambdas[0, 1, :, :].max() == 0.0
    # Nearest context to mask should have higher weight than farthest
    near_val = lambdas[0, 0, :, :].max()  # frame 0 (1 from frame 1)
    far_val = lambdas[0, 3, :, :].max()  # frame 3 (2 from frame 1)
    assert near_val > far_val


def test_loss_stop_grad_on_target() -> None:
    """Target encoder output must not receive gradients through the loss."""
    B, F, T, H, W = 1, 16, 4, 4, 4
    feat = torch.randn(B, F, T, H, W, requires_grad=True)
    target = torch.randn(B, F, T, H, W, requires_grad=True)

    loss_fn = JEPALoss(feature_dim=F)
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

    loss_fn = JEPALoss(feature_dim=F, lambda_base=0.5)
    loss = loss_fn(feat, target, mask=mask)
    assert loss.item() < 1e-5


def test_ema_encoder_shadows_source() -> None:
    """EMA encoder weights must be copies that diverge as source trains."""
    encoder = nn.Linear(4, 4)
    ema = EMAEncoder(module=encoder, decay=0.999)

    # Initially identical
    with torch.no_grad():
        for p_src, p_tgt in zip(encoder.parameters(), ema.shadow_params, strict=True):
            torch.testing.assert_close(p_tgt, p_src)

    # Source changes
    original = encoder.weight.data.clone()
    encoder.weight.data.fill_(99.0)
    ema.update()

    # EMA should be close to original (high decay = slow tracking)
    moved = ema.shadow_params[0]
    assert not torch.allclose(moved, encoder.weight.data)
    # With decay=0.999, shadow = 0.999*old + 0.001*99, so moves toward 99 but stays near old
    assert torch.allclose(moved, original * 0.999 + 0.001 * 99.0, atol=1e-5)
