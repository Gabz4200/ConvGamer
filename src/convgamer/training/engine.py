from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Timer

from convgamer.callbacks.ema_update import EMAUpdateCallback

_BUILDER_FOR_LAZY: dict[str, Callable[[], Callback]] = {
    "model_checkpoint": lambda: ModelCheckpoint(
        monitor="val/loss", save_last=True, save_top_k=1, mode="min"
    ),
    "ema_update": lambda: EMAUpdateCallback(),
    "lr_monitor": lambda: LearningRateMonitor(logging_interval="step"),
    "timer": lambda: Timer(),
}


def _build_callback(name: str) -> Callback:
    """Factory matching callback names in trainer configs."""
    try:
        return _BUILDER_FOR_LAZY[name]()
    except KeyError:
        raise ValueError(f"Unknown callback '{name}'") from None


def create_trainer(cfg: DictConfig) -> pl.Trainer:
    callback_names = getattr(cfg.trainer, "callbacks", []) or []
    callbacks = [_build_callback(name) for name in callback_names]
    ckpt_cfg = getattr(cfg.trainer, "checkpoint_callback", None)
    if ckpt_cfg:
        from pytorch_lightning.callbacks import ModelCheckpoint

        container = OmegaConf.to_container(ckpt_cfg, resolve=True)
        if not isinstance(container, dict):
            raise TypeError(f"checkpoint config must be a dict, got {type(container).__name__}")
        kwargs_ckpt = {str(k): v for k, v in container.items()}
        callbacks.append(ModelCheckpoint(**kwargs_ckpt))
    kwargs: dict[str, Any] = {
        "accelerator": cfg.trainer.accelerator,
        "devices": cfg.trainer.devices,
        "precision": cfg.trainer.precision,
        "log_every_n_steps": cfg.trainer.log_every_n_steps,
        "fast_dev_run": getattr(cfg, "fast_dev_run", False),
        "callbacks": callbacks,
    }
    for key in (
        "max_epochs",
        "max_steps",
        "strategy",
        "accumulate_grad_batches",
        "gradient_clip_val",
        "val_check_interval",
        "limit_val_batches",
    ):
        value = getattr(cfg.trainer, key, None)
        if value is not None:
            kwargs[key] = value
    return pl.Trainer(**kwargs)
