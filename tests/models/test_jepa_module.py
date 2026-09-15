"""Behavior tests for the V-JEPA 2.1 ConvGamer LightningModule.

Seams tested:
- ``ConvGamerVJEPAModel.training_step``: produces scalar loss, no crash.
- ``ConvGamerVJEPAModel``: x-encoder, y-encoder (EMA), predictor wired.
- ``configure_optimizers``: returns optimizer + scheduler.
- EMA shadow: tracks the online encoder, round-trips through checkpoints.
"""

from __future__ import annotations

import torch

from convgamer.models.convgamer.encoder import ConvGamerEncoder
from convgamer.models.jepa import JEPALoss, VJEPAPredictor
from convgamer.modules.jepa_module import ConvGamerVJEPAModel


def test_jepa_model_training_step_runs() -> None:
    """One training step must return a scalar loss."""
    model = ConvGamerVJEPAModel(
        encoder=ConvGamerEncoder(
            input_dim=3,
            hidden_dim=32,
            num_layers=1,
            target_size=(8, 8),
            temporal_dilations=(1,),
        ),
        predictor=VJEPAPredictor(feature_dim=256, predictor_dim=256, num_layers=1, num_levels=1),
        loss=JEPALoss(feature_dim=256, lambda_base=0.5),
        ema_decay=0.99,
    )

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
    model = ConvGamerVJEPAModel(
        encoder=ConvGamerEncoder(
            input_dim=3,
            hidden_dim=32,
            num_layers=1,
            target_size=(8, 8),
            temporal_dilations=(1,),
        ),
        predictor=VJEPAPredictor(feature_dim=256, predictor_dim=256, num_layers=1, num_levels=1),
        loss=JEPALoss(feature_dim=256, lambda_base=0.5),
        ema_decay=0.999,
    )
    assert hasattr(model, "ema_encoder")
    assert model.ema_decay == 0.999


def test_jepa_model_configure_optimizers() -> None:
    """configure_optimizers must return optimizer and scheduler."""
    model = ConvGamerVJEPAModel(
        encoder=ConvGamerEncoder(
            input_dim=3,
            hidden_dim=32,
            num_layers=1,
            target_size=(8, 8),
            temporal_dilations=(1,),
        ),
        predictor=VJEPAPredictor(feature_dim=256, predictor_dim=256, num_layers=1, num_levels=1),
        loss=JEPALoss(feature_dim=256, lambda_base=0.5),
        ema_decay=0.999,
        lr=1e-4,
        warmup_steps=10,
        total_steps=100,
    )
    out = model.configure_optimizers()
    assert isinstance(out, tuple)
    opt, sched = out
    # Lightning returns lists
    if isinstance(opt, list):
        opt = opt[0]
    if isinstance(sched, list):
        sched = sched[0]
    assert isinstance(opt, torch.optim.Optimizer)
    assert isinstance(sched, list) or callable(getattr(sched, "get_last_lr", None))


def _tiny_jepa_model(ema_decay: float = 0.99) -> ConvGamerVJEPAModel:
    return ConvGamerVJEPAModel(
        encoder=ConvGamerEncoder(
            input_dim=3,
            hidden_dim=32,
            num_layers=1,
            target_size=(8, 8),
            temporal_dilations=(1,),
        ),
        predictor=VJEPAPredictor(feature_dim=256, predictor_dim=256, num_layers=1, num_levels=1),
        loss=JEPALoss(feature_dim=256, lambda_base=0.5),
        ema_decay=ema_decay,
    )


def test_jepa_ema_updates_shadow_weights() -> None:
    """on_train_batch_end must move the shadow toward the online encoder."""
    model = _tiny_jepa_model(ema_decay=0.0)
    online = next(model.encoder.parameters())
    shadow = next(model.ema_encoder.shadow_module.parameters())
    with torch.no_grad():
        online.fill_(1.0)
        shadow.fill_(0.0)
    model.on_train_batch_end(None, None, 0)
    assert torch.allclose(next(model.ema_encoder.shadow_module.parameters()), online)


def test_jepa_ema_checkpoint_round_trip() -> None:
    """EMA shadow survives save/load instead of being rebuilt from the encoder."""
    model = _tiny_jepa_model()
    shadow = next(model.ema_encoder.shadow_module.parameters())
    with torch.no_grad():
        shadow.fill_(3.0)
    checkpoint: dict = {}
    model.on_save_checkpoint(checkpoint)
    restored = _tiny_jepa_model()
    restored.on_load_checkpoint(checkpoint)
    restored_shadow = next(restored.ema_encoder.shadow_module.parameters())
    torch.testing.assert_close(restored_shadow, torch.full_like(restored_shadow, 3.0))
