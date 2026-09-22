"""Behavior tests for the V-JEPA 2.1 ConvGamer LightningModule.

Seams tested:
- ``ConvGamerVJEPAModel.training_step``: produces scalar loss, no crash.
- ``ConvGamerVJEPAModel``: x-encoder, y-encoder (EMA), predictor wired.
- ``configure_optimizers``: returns optimizer + scheduler.
- EMA shadow: tracks the online encoder, round-trips through checkpoints.
"""

from __future__ import annotations

import pytest
import torch

from convgamer.callbacks.ema_update import EMAUpdateCallback
from convgamer.models.convgamer.encoder import ConvGamerEncoder
from convgamer.models.jepa import JEPALoss, VJEPAPredictor
from convgamer.modules.jepa_module import ConvGamerVJEPAModel


def _tiny_jepa_model(
    ema_decay: float = 0.99,
    lr: float = 5.25e-4,
    warmup_steps: int = 12_000,
    total_steps: int = 135_000,
) -> ConvGamerVJEPAModel:
    return ConvGamerVJEPAModel(
        encoder=ConvGamerEncoder(
            input_dim=3,
            hidden_dim=32,
            num_layers=1,
            target_size=(8, 8),
            temporal_dilations=(1,),
        ),
        predictor=VJEPAPredictor(feature_dim=256, predictor_dim=256, num_layers=1),
        loss=JEPALoss(feature_dim=256, lambda_base=0.5),
        ema_decay=ema_decay,
        lr=lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )


def test_jepa_model_training_step_runs() -> None:
    """One training step must return a scalar loss."""
    model = _tiny_jepa_model(ema_decay=0.99)

    x = torch.randn(1, 3, 4, 16, 16)
    y = torch.randn(1, 3, 4, 16, 16)
    mask = torch.zeros(1, 4, 16, 16, dtype=torch.bool)
    mask[0, :, :8, :8] = True

    loss = model.training_step((x, y, mask), batch_idx=0)
    assert loss.dim() == 0  # scalar
    assert torch.isfinite(loss)
    assert loss.item() > 0
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.encoder.parameters()
    )


def test_jepa_model_ema_encoder_exists() -> None:
    """Model must have an EMA target encoder."""
    model = _tiny_jepa_model(ema_decay=0.999)
    assert hasattr(model, "ema_encoder")
    assert model.ema_decay == 0.999


def test_jepa_model_configure_optimizers() -> None:
    """configure_optimizers must step the warmup scheduler every update."""
    model = _tiny_jepa_model(lr=1e-4, warmup_steps=10, total_steps=100)
    out = model.configure_optimizers()
    assert isinstance(out, dict)
    assert isinstance(out["optimizer"], torch.optim.Optimizer)
    scheduler_cfg = out["lr_scheduler"]
    assert isinstance(scheduler_cfg, dict)
    assert scheduler_cfg["interval"] == "step"
    assert scheduler_cfg["frequency"] == 1
    assert callable(getattr(scheduler_cfg["scheduler"], "get_last_lr", None))


def test_jepa_ema_callback_updates_shadow_weights() -> None:
    """EMAUpdateCallback moves the shadow toward the online encoder."""
    model = _tiny_jepa_model(ema_decay=0.0)
    online = next(model.encoder.parameters())
    with torch.no_grad():
        online.fill_(1.0)
        for p in model.ema_encoder.state_dict().values():
            p.fill_(0.0)
    EMAUpdateCallback().on_train_batch_end(
        trainer=None,
        pl_module=model,
        outputs=None,
        batch=None,
        batch_idx=0,  # type: ignore[arg-type]
    )
    shadow_first = next(iter(model.ema_encoder.state_dict().values()))
    assert torch.allclose(shadow_first, online)


def test_jepa_ema_callback_requires_ema_encoder() -> None:
    """A module without an EMA target fails loudly, not silently."""
    model = _tiny_jepa_model()
    del model.ema_encoder
    with pytest.raises(AttributeError, match="ema_encoder"):
        EMAUpdateCallback().on_train_batch_end(
            trainer=None,
            pl_module=model,
            outputs=None,
            batch=None,
            batch_idx=0,  # type: ignore[arg-type]
        )


def test_jepa_ema_checkpoint_round_trip() -> None:
    """EMA shadow survives save/load instead of being rebuilt from the encoder."""
    model = _tiny_jepa_model()
    with torch.no_grad():
        for p in model.ema_encoder.state_dict().values():
            p.fill_(3.0)
    checkpoint: dict = {}
    model.on_save_checkpoint(checkpoint)
    restored = _tiny_jepa_model()
    restored.on_load_checkpoint(checkpoint)
    restored_shadow = next(iter(restored.ema_encoder.state_dict().values()))
    torch.testing.assert_close(restored_shadow, torch.full_like(restored_shadow, 3.0))


def test_jepa_training_step_output_is_finite() -> None:
    """Full JEPA forward + backward: loss and gradients must be finite."""
    model = _tiny_jepa_model(ema_decay=0.99)

    x = torch.randn(1, 3, 4, 16, 16)
    y = torch.randn(1, 3, 4, 16, 16)
    mask = torch.zeros(1, 4, 16, 16, dtype=torch.bool)
    mask[0, :, :8, :8] = True

    loss = model.training_step((x, y, mask), batch_idx=0)
    assert torch.isfinite(loss), "JEPA loss is NaN or Inf"
    loss.backward()
    # All encoder and predictor gradients must be finite
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "Non-finite gradient in parameter"


def test_jepa_multi_step_training_stability() -> None:
    """Multi-step JEPA training: gradients must not explode or vanish.

    Runs 5 training steps and checks gradient norms stay healthy.
    """
    model = _tiny_jepa_model(ema_decay=0.99)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    x = torch.randn(1, 3, 4, 16, 16)
    y = torch.randn(1, 3, 4, 16, 16)
    mask = torch.zeros(1, 4, 16, 16, dtype=torch.bool)
    mask[0, :, :8, :8] = True

    grad_norms: list[float] = []
    for _ in range(5):
        opt.zero_grad()
        loss = model.training_step((x, y, mask), batch_idx=0)
        assert torch.isfinite(loss), "Loss became non-finite"
        loss.backward()
        # EMA maintenance is a trainer-level observer, not a module hook.
        EMAUpdateCallback().on_train_batch_end(
            trainer=None,
            pl_module=model,
            outputs=None,
            batch=(x, y, mask),
            batch_idx=0,  # type: ignore[arg-type]
        )

        total_norm = sum(
            p.grad.abs().sum().item() ** 2 for p in model.parameters() if p.grad is not None
        )
        grad_norms.append(total_norm**0.5)
        opt.step()

    max_norm = max(grad_norms)
    min_norm = min(grad_norms)
    assert min_norm > 1e-6, f"Gradient vanishing: min norm = {min_norm}"
    assert max_norm < 1e4, f"Gradient explosion: max norm = {max_norm}"
