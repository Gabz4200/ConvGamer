from __future__ import annotations

import pytorch_lightning as pl
from omegaconf import DictConfig

from convgamer.callbacks.logging import _build_callback  # noqa: F401


def create_trainer(cfg: DictConfig) -> pl.Trainer:
    callback_names = cfg.trainer.callbacks
    callbacks = [_build_callback(name) for name in callback_names]
    return pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        fast_dev_run=getattr(cfg, "fast_dev_run", False),
        callbacks=callbacks,
    )
