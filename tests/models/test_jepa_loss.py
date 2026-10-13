"""Behavior tests for V-JEPA 2.1 dense loss extensions.

Seams tested:
- ``JEPALoss``: per-modality lambda (0.5 video / 0.7 image) and the
  linear warmup schedule from Appendix A.
- ``compute_context_lambdas``: distance-weighted weighting still decays
  with distance to the nearest mask token.
- ``SIGReg`` (LeJEPA, arXiv:2511.08544): sketched Gaussianity statistic is
  finite, nonnegative, differentiable, and scores collapsed input above
  isotropic input; ``JEPALoss`` adds it only when enabled.
"""

from __future__ import annotations

import pytest
import torch

from convgamer.models.jepa import JEPALoss, SIGReg, compute_context_lambdas


def test_lambda_image_is_higher_than_video() -> None:
    """Static-image (T=1) context weight must exceed video (T>1)."""
    loss = JEPALoss(lambda_base=0.5, lambda_image=0.7)
    video_mask = torch.zeros(1, 4, 4, 4, dtype=torch.bool)
    image_mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    assert loss._effective_lambda(image_mask) == 0.7
    assert loss._effective_lambda(video_mask) == 0.5
    assert loss._effective_lambda(image_mask) > loss._effective_lambda(video_mask)


def test_lambda_warmup_ramps_from_zero() -> None:
    """With warmup enabled, the context-loss component starts at 0 and ramps."""
    loss = JEPALoss(lambda_base=0.5, lambda_image=0.7, lambda_warmup_steps=10)
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


def test_sigreg_scores_collapsed_above_isotropic() -> None:
    """Degenerate embeddings score higher than Gaussian ones (fixed slices)."""
    torch.manual_seed(0)
    collapsed = SIGReg()(torch.zeros(512, 16))
    torch.manual_seed(0)
    isotropic = SIGReg()(torch.randn(512, 16))
    assert torch.isfinite(collapsed) and torch.isfinite(isotropic)
    assert collapsed.item() >= 0.0 and isotropic.item() >= 0.0
    assert collapsed.item() > isotropic.item()


def test_sigreg_flows_gradients_and_validates_shape() -> None:
    """Statistic is a differentiable scalar; non-2D input fails loudly."""
    tokens = torch.randn(256, 16, requires_grad=True)
    out = SIGReg(num_slices=32)(tokens)
    assert out.dim() == 0
    out.backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    with pytest.raises(ValueError, match=r"\(N, D\)"):
        SIGReg()(torch.randn(4, 4, 4))


def test_jepa_loss_adds_sigreg_only_when_enabled() -> None:
    """sigreg_lambda=0 keeps pure JEPA; >0 adds exactly lambda * component."""
    pred = torch.zeros(1, 8, 2, 4, 4)
    target = torch.ones(1, 8, 2, 4, 4)
    mask = torch.zeros(1, 2, 4, 4, dtype=torch.bool)
    mask[0, 1:, :, :] = True

    off = JEPALoss(lambda_base=0.5)
    base = off(pred, target, mask)
    assert torch.isfinite(base)
    assert off.last_sigreg is None

    on = JEPALoss(lambda_base=0.5, sigreg_lambda=0.5)
    total = on(pred, target, mask)
    assert on.last_sigreg is not None and on.last_sigreg > 0
    torch.testing.assert_close(total, base + 0.5 * on.last_sigreg)


def test_lambda_warmup_survives_state_dict_round_trip() -> None:
    """The warmup counter persists via state_dict (resume keeps the ramp)."""
    loss = JEPALoss(lambda_base=0.5, lambda_image=0.7, lambda_warmup_steps=10)
    loss.set_step(7)
    restored = JEPALoss(lambda_base=0.5, lambda_image=0.7, lambda_warmup_steps=10)
    restored.load_state_dict(loss.state_dict())
    mask = torch.zeros(1, 4, 4, 4, dtype=torch.bool)
    assert restored._effective_lambda(mask) == loss._effective_lambda(mask)


def test_sigreg_accepts_half_precision() -> None:
    """AMP bf16/fp16 tokens must not crash the statistic."""
    tokens = torch.randn(64, 16, dtype=torch.float16)
    out = SIGReg(num_slices=16)(tokens)
    assert torch.isfinite(out)


def test_sigreg_rejects_zero_slices() -> None:
    """num_slices=0 would NaN; fail loudly at construction."""
    with pytest.raises(ValueError, match="num_slices"):
        SIGReg(num_slices=0)


def test_jepa_loss_validates_batch_and_device() -> None:
    """Batch mismatch fails loudly instead of broadcasting silently."""
    loss = JEPALoss(lambda_base=0.5)
    pred = torch.zeros(2, 8, 2, 4, 4)
    target = torch.zeros(2, 8, 2, 4, 4)
    mask = torch.zeros(1, 2, 4, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="batch"):
        loss(pred, target, mask)


def test_jepa_loss_skips_sigreg_in_eval() -> None:
    """Validation loss is deterministic: no random slices in eval mode."""
    loss = JEPALoss(lambda_base=0.5, sigreg_lambda=0.5)
    loss.eval()
    pred = torch.zeros(1, 8, 2, 4, 4)
    target = torch.ones(1, 8, 2, 4, 4)
    mask = torch.zeros(1, 2, 4, 4, dtype=torch.bool)
    mask[0, 1:, :, :] = True
    assert torch.isfinite(loss(pred, target, mask))
    assert loss.last_sigreg is None
