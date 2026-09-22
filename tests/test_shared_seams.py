"""Behavior tests for shared seams: registry, callbacks."""

from __future__ import annotations

import pytest
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Timer

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
