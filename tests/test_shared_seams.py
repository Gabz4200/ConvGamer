"""Behavior tests for shared seams: registry, loss wrapper, metrics, callbacks."""

from __future__ import annotations

import pytest
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    Timer,
)

from convgamer.callbacks.ema_update import EMAUpdateCallback
from convgamer.callbacks.logging import ConvGamerLogger
from convgamer.models.base import BaseModel
from convgamer.models.registry import MODEL_REGISTRY, get_model, register_model
from convgamer.training.engine import _build_callback


class ClassificationLoss(torch.nn.Module):
    """Test-only loss wrapper (no production module consumes it)."""

    def __init__(self, loss_type: str = "cross_entropy"):
        super().__init__()
        if loss_type == "cross_entropy":
            self._loss = torch.nn.CrossEntropyLoss()
        elif loss_type == "mse":
            self._loss = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss_type '{loss_type}'")

    def forward(self, pred, target):
        return self._loss(pred, target)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Test-only mean correctness fraction over argmax predictions."""
    preds = logits.argmax(dim=-1)
    return float((preds == targets).float().mean().item())


def test_when_unknown_model_then_key_error() -> None:
    """Unknown registry names raise KeyError with registered names listed."""
    with pytest.raises(KeyError, match="Unknown model"):
        get_model("no-such-model-xyz")


def test_when_duplicate_registration_then_raises() -> None:
    """Registering the same name twice is rejected."""

    @register_model("__test_dup_probe__")
    class _Probe(BaseModel):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    with pytest.raises(ValueError, match="already registered"):
        register_model("__test_dup_probe__")(_Probe)
    MODEL_REGISTRY.pop("__test_dup_probe__", None)


def test_when_cross_entropy_then_matches_nll() -> None:
    """Default loss wrapper matches CrossEntropyLoss on logits."""
    loss_fn = ClassificationLoss()
    logits = torch.randn(4, 3)
    targets = torch.randint(0, 3, (4,))
    torch.testing.assert_close(
        loss_fn(logits, targets),
        torch.nn.functional.cross_entropy(logits, targets),
    )


def test_when_mse_then_matches_mse() -> None:
    """MSE variant matches MSELoss elementwise."""
    loss_fn = ClassificationLoss(loss_type="mse")
    pred = torch.randn(2, 4)
    target = torch.randn(2, 4)
    torch.testing.assert_close(loss_fn(pred, target), torch.nn.functional.mse_loss(pred, target))


def test_when_unknown_loss_type_then_raises() -> None:
    """Unknown loss names fail at construction, not at forward."""
    with pytest.raises(ValueError, match="Unknown loss_type"):
        ClassificationLoss(loss_type="hinge")


def test_when_perfect_logits_then_accuracy_one() -> None:
    """Accuracy helper returns 1.0 for argmax-correct logits."""
    logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    assert accuracy(logits, torch.tensor([0, 1])) == 1.0


def test_when_half_correct_then_accuracy_half() -> None:
    """Accuracy helper returns the mean correctness fraction."""
    logits = torch.tensor([[5.0, 0.0], [5.0, 0.0]])
    assert accuracy(logits, torch.tensor([0, 1])) == 0.5


def test_when_known_callback_then_builds_expected_type() -> None:
    """Trainer callback names map to the configured PL callback types."""
    assert isinstance(_build_callback("model_checkpoint"), ModelCheckpoint)
    assert isinstance(_build_callback("early_stopping"), EarlyStopping)
    assert isinstance(_build_callback("logger"), ConvGamerLogger)
    assert isinstance(_build_callback("ema_update"), EMAUpdateCallback)
    assert isinstance(_build_callback("lr_monitor"), LearningRateMonitor)
    assert isinstance(_build_callback("timer"), Timer)


def test_when_unknown_callback_then_raises() -> None:
    """Unknown trainer callback names fail with the offending name."""
    with pytest.raises(ValueError, match="no-such-callback"):
        _build_callback("no-such-callback")
