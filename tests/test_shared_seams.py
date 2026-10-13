"""Behavior tests for shared seams: registry, callbacks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Timer

from convgamer.callbacks.collapse_guard import CollapseGuardCallback
from convgamer.callbacks.ema_update import EMAUpdateCallback
from convgamer.models.registry import MODEL_REGISTRY, get_model, register_model
from convgamer.training.engine import _build_callback


def test_when_unknown_model_then_key_error() -> None:
    """Unknown registry names raise KeyError with registered names listed."""
    with pytest.raises(KeyError, match="Unknown model"):
        get_model("no-such-model-xyz")


def test_when_duplicate_registration_then_raises() -> None:
    """Registering the same name twice is rejected."""

    import torch.nn as nn

    @register_model("__test_dup_probe__")
    class _Probe(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    with pytest.raises(ValueError, match="already registered"):
        register_model("__test_dup_probe__")(_Probe)
    MODEL_REGISTRY.pop("__test_dup_probe__", None)


def test_when_known_callback_then_builds_expected_type() -> None:
    """Trainer callback names map to the configured PL callback types."""
    assert isinstance(_build_callback("model_checkpoint"), ModelCheckpoint)
    assert isinstance(_build_callback("ema_update"), EMAUpdateCallback)
    assert isinstance(_build_callback("lr_monitor"), LearningRateMonitor)
    assert isinstance(_build_callback("timer"), Timer)


def test_when_unknown_callback_then_raises() -> None:
    """Unknown trainer callback names fail with the offending name."""
    with pytest.raises(ValueError, match="no-such-callback"):
        _build_callback("no-such-callback")


def test_when_collapse_guard_healthy_then_never_stops() -> None:
    """Healthy feature std resets the counter; missing signal is ignored."""
    guard = CollapseGuardCallback(min_std=1e-4, patience=3)
    trainer = SimpleNamespace(
        callback_metrics={"val/pred_std": torch.tensor(0.5)}, should_stop=False
    )
    for _ in range(5):
        guard.on_validation_epoch_end(trainer, None)  # type: ignore[arg-type]
    assert trainer.should_stop is False
    guard.on_validation_epoch_end(SimpleNamespace(callback_metrics={}, should_stop=False), None)  # type: ignore[arg-type]
    assert guard._bad_checks == 0


def test_when_collapse_guard_collapsed_then_stops_after_patience() -> None:
    """Sustained ~zero (or NaN) std trips should_stop only at patience."""
    guard = CollapseGuardCallback(min_std=1e-4, patience=3)
    trainer = SimpleNamespace(
        callback_metrics={"val/pred_std": torch.tensor(0.0)}, should_stop=False
    )
    guard.on_validation_epoch_end(trainer, None)  # type: ignore[arg-type]
    guard.on_validation_epoch_end(trainer, None)  # type: ignore[arg-type]
    assert trainer.should_stop is False
    guard.on_validation_epoch_end(trainer, None)  # type: ignore[arg-type]
    assert trainer.should_stop is True

    nan_guard = CollapseGuardCallback(min_std=1e-4, patience=1)
    nan_trainer = SimpleNamespace(
        callback_metrics={"val/pred_std": torch.tensor(float("nan"))}, should_stop=False
    )
    nan_guard.on_validation_epoch_end(nan_trainer, None)  # type: ignore[arg-type]
    assert nan_trainer.should_stop is True


def test_when_collapse_guard_below_min_steps_then_ignores_signal() -> None:
    """Early checks on near-init weights never stop, however bad the signal."""
    guard = CollapseGuardCallback(min_std=1e-4, patience=1)
    trainer = SimpleNamespace(
        callback_metrics={"val/pred_std": torch.tensor(0.0)},
        should_stop=False,
        global_step=100,
        min_steps=2000,
    )
    guard.on_validation_epoch_end(trainer, None)  # type: ignore[arg-type]
    assert trainer.should_stop is False
    assert guard._bad_checks == 0
