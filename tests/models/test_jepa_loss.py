"""Behavior tests for V-JEPA 2.1 dense loss extensions.

Seams tested:
- ``JEPALoss``: per-modality lambda (0.5 video / 0.7 image) and the
  linear warmup schedule from Appendix A.
- ``compute_context_lambdas``: distance-weighted weighting still decays
  with distance to the nearest mask token.
"""

from __future__ import annotations

import torch

from convgamer.models.jepa import JEPALoss, compute_context_lambdas


def test_lambda_image_is_higher_than_video() -> None:
    """Static-image (T=1) context weight must exceed video (T>1)."""
    loss = JEPALoss(feature_dim=8, lambda_base=0.5, lambda_image=0.7)
    video_mask = torch.zeros(1, 4, 4, 4, dtype=torch.bool)
    image_mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    assert loss._effective_lambda(image_mask) == 0.7
    assert loss._effective_lambda(video_mask) == 0.5
    assert loss._effective_lambda(image_mask) > loss._effective_lambda(video_mask)


def test_lambda_warmup_ramps_from_zero() -> None:
    """With warmup enabled, the context-loss component starts at 0 and ramps."""
    loss = JEPALoss(feature_dim=8, lambda_base=0.5, lambda_image=0.7, lambda_warmup_steps=10)
    pred = torch.zeros(1, 8, 4, 4, 4)
    target = torch.ones(1, 8, 4, 4, 4)
    mask = torch.zeros(1, 4, 4, 4, dtype=torch.bool)
    mask[0, 1:, :, :] = True  # context on frame 0, masked elsewhere

    def _ctx_component() -> float:
        # Isolate the L_ctx half through the public forward: total minus L_predict.
        total = loss(pred, target, mask)
        l1 = (pred - target.detach()).abs()
        token_loss = l1.mean(dim=1)
        pred_loss = token_loss.masked_fill(~mask, 0.0)
        n_ctx = (~mask).sum().clamp(min=1)
        l_predict = pred_loss.sum() / mask.sum().clamp(min=1)
        assert n_ctx > 0  # guard: context positions must exist for the ramp
        return float((total - l_predict).item())

    loss.set_step(0)
    early = _ctx_component()
    loss.set_step(5)
    mid = _ctx_component()
    loss.set_step(10)
    late = _ctx_component()
    loss.set_step(10)
    full = _ctx_component()

    assert early == 0.0
    assert 0.0 < mid < full
    assert late == full


def test_context_lambdas_zero_at_masked_positions() -> None:
    """Masked tokens must contribute zero to L_ctx (Eq. 3)."""
    mask = torch.zeros(1, 4, 4, 4, dtype=torch.bool)
    mask[0, 0, 1, 1] = True
    lambdas = compute_context_lambdas(mask, lambda_base=0.5)
    assert lambdas.shape == (1, 4, 4, 4)
    assert lambdas[0, 0, 1, 1].item() == 0.0
    assert (lambdas[0, mask[0]] == 0).all()
    assert (lambdas[0, ~mask[0]] > 0).all()


def test_context_lambdas_decays_with_distance() -> None:
    """Closer to a mask token -> higher lambda weight."""
    mask = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    mask[0, 0, 4, 4] = True
    lambdas = compute_context_lambdas(mask, lambda_base=0.5)
    near = lambdas[0, 0, 4, 3].item()
    far = lambdas[0, 0, 0, 0].item()
    assert near > far > 0.0
