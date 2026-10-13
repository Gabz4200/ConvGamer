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


def maybe_run_lr_finder(
    trainer: pl.Trainer,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    cfg: DictConfig,
) -> float | None:
    """Run Lightning's LR finder when ``trainer.lr_finder.enabled`` is true.

    Restores the pre-finder weights (the finder trains for real) and writes
    the suggestion back to ``model.lr``, which both training systems read in
    ``configure_optimizers``. Returns the suggestion, or ``None`` when disabled.
    """
    finder_cfg = getattr(cfg.trainer, "lr_finder", None)
    if not finder_cfg or not finder_cfg.get("enabled", False):
        return None
    if not hasattr(model, "lr"):
        raise TypeError(
            f"LR finder needs a model exposing an 'lr' attribute, got {type(model).__name__}"
        )
    import copy

    from pytorch_lightning.tuner import Tuner

    previous_lr = float(model.lr)  # type: ignore[attr-defined]
    snapshot = copy.deepcopy(model.state_dict())
    try:
        lr_finder = Tuner(trainer).lr_find(
            model,
            datamodule=datamodule,
            min_lr=float(finder_cfg.get("min_lr", 1e-6)),
            max_lr=float(finder_cfg.get("max_lr", 1e-2)),
            num_training=int(finder_cfg.get("num_training", 100)),
            mode=str(finder_cfg.get("mode", "exponential")),
            update_attr=False,
        )
    finally:
        model.load_state_dict(snapshot)
    if lr_finder is None:
        raise RuntimeError("LR finder returned no result")
    suggestion = lr_finder.suggestion()
    if suggestion is None:
        # Short sweeps hold fewer points than the default skip window.
        n_points = len(lr_finder.results["lr"])
        suggestion = lr_finder.suggestion(skip_begin=max(1, n_points // 4))
    if suggestion is None:
        raise RuntimeError("LR finder produced no usable suggestion")
    suggestion = float(suggestion)
    model.lr = suggestion  # type: ignore[attr-defined]
    print(f"LR finder: {previous_lr:.3e} -> {suggestion:.3e}")
    return suggestion


def create_trainer(cfg: DictConfig) -> pl.Trainer:
    callback_names = getattr(cfg.trainer, "callbacks", []) or []
    callbacks = [_build_callback(name) for name in callback_names]
    ckpt_cfg = getattr(cfg.trainer, "checkpoint_callback", None)
    if ckpt_cfg:
        container = OmegaConf.to_container(ckpt_cfg, resolve=True)
        if not isinstance(container, dict):
            raise TypeError(f"checkpoint config must be a dict, got {type(container).__name__}")
        kwargs_ckpt = {str(k): v for k, v in container.items()}
        # Drop the lazy val/loss checkpoint when an explicit one is configured.
        callbacks = [c for c in callbacks if not isinstance(c, ModelCheckpoint)]
        callbacks.append(ModelCheckpoint(**kwargs_ckpt))
    else:
        # No validation (fast_dev_run / limit_val_batches=0) breaks the
        # default val/loss monitor — fall back to last-checkpoint mode.
        limit_val = getattr(cfg.trainer, "limit_val_batches", None)
        if getattr(cfg, "fast_dev_run", False) or limit_val in (0, 0.0):
            callbacks = [
                c
                for c in callbacks
                if not (isinstance(c, ModelCheckpoint) and c.monitor == "val/loss")
            ]
            if not any(isinstance(c, ModelCheckpoint) for c in callbacks):
                callbacks.append(ModelCheckpoint(save_last=True))
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
        "benchmark",
    ):
        value = getattr(cfg.trainer, key, None)
        if value is not None:
            kwargs[key] = value
    return pl.Trainer(**kwargs)
