"""Tests for the opt-in Lightning LR finder hook (training/engine)."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from convgamer.data.jepa_datamodule import VJEPAGamingDataModule
from convgamer.models.convgamer.encoder import ConvGamerEncoder
from convgamer.models.jepa import JEPALoss, VJEPAPredictor
from convgamer.modules.jepa_module import ConvGamerVJEPAModel
from convgamer.training.engine import create_trainer, maybe_run_lr_finder


def _tiny_setup(lr_finder: dict | None = None):
    model = ConvGamerVJEPAModel(
        encoder=ConvGamerEncoder(input_dim=3, hidden_dim=16, num_layers=1),
        predictor=VJEPAPredictor(feature_dim=256, predictor_dim=64, num_layers=1),
        loss=JEPALoss(lambda_base=0.5),
        lr=1e-4,
    )
    datamodule = VJEPAGamingDataModule(
        batch_size=1, num_workers=0, num_frames=2, height=64, width=64
    )
    trainer_cfg: dict = {
        "accelerator": "cpu",
        "devices": 1,
        "precision": "32-true",
        "log_every_n_steps": 1,
        "callbacks": [],
        "max_steps": 2,
        "limit_val_batches": 0,
    }
    if lr_finder is not None:
        trainer_cfg["lr_finder"] = lr_finder
    cfg = OmegaConf.create({"trainer": trainer_cfg, "fast_dev_run": False})
    trainer = create_trainer(cfg)
    return trainer, model, datamodule, cfg


def test_lr_finder_disabled_returns_none_and_keeps_lr() -> None:
    trainer, model, datamodule, cfg = _tiny_setup()
    assert maybe_run_lr_finder(trainer, model, datamodule, cfg) is None
    assert model.lr == pytest.approx(1e-4)


def test_lr_finder_sets_lr_and_restores_weights() -> None:
    trainer, model, datamodule, cfg = _tiny_setup(
        {"enabled": True, "min_lr": 1e-6, "max_lr": 1e-3, "num_training": 8}
    )
    before = {k: v.clone() for k, v in model.state_dict().items()}
    suggestion = maybe_run_lr_finder(trainer, model, datamodule, cfg)
    assert suggestion is not None and torch.isfinite(torch.tensor(suggestion))
    assert model.lr == pytest.approx(suggestion)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k])


def test_lr_finder_requires_lr_attribute() -> None:
    trainer, _, datamodule, cfg = _tiny_setup({"enabled": True})
    with pytest.raises(TypeError, match="'lr'"):
        maybe_run_lr_finder(trainer, object(), datamodule, cfg)
