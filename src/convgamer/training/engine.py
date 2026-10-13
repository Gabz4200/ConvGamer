from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    Timer,
)

from convgamer.callbacks.collapse_guard import CollapseGuardCallback
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


def _section_kwargs(cfg: DictConfig, name: str) -> dict[str, Any] | None:
    """``cfg.trainer.<name>`` as a plain dict when ``enabled``, else ``None``.

    Absent or disabled sections (smoke configs, no-val runs) yield ``None``
    so the trainer builds without the callback. The collapse guard skips a
    missing sentinel silently; EarlyStopping defaults to ``strict=False`` so
    val-free runs are safe.
    """
    section = getattr(cfg.trainer, name, None)
    if not section or not bool(section.get("enabled", False)):
        return None
    container = OmegaConf.to_container(section, resolve=True)
    if not isinstance(container, dict):
        raise TypeError(f"{name} config must be a dict, got {type(container).__name__}")
    return {str(k): v for k, v in container.items() if str(k) != "enabled"}


def _build_early_stopping(cfg: DictConfig) -> EarlyStopping | None:
    """Build EarlyStopping from ``cfg.trainer.early_stopping`` when enabled."""
    kwargs = _section_kwargs(cfg, "early_stopping")
    if kwargs is None:
        return None
    kwargs.setdefault("strict", False)
    return EarlyStopping(**kwargs)


def _build_collapse_guard(cfg: DictConfig) -> CollapseGuardCallback | None:
    """Build the collapse guard from ``cfg.trainer.collapse_guard`` when enabled."""
    kwargs = _section_kwargs(cfg, "collapse_guard")
    return CollapseGuardCallback(**kwargs) if kwargs is not None else None


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
    from pytorch_lightning.tuner import Tuner

    previous_lr = float(model.lr)  # type: ignore[attr-defined]
    snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ema = getattr(model, "ema_encoder", None)
    ema_snapshot = None
    if ema is not None and hasattr(ema, "state_dict"):
        ema_snapshot = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
    loss_fn = getattr(model, "loss_fn", None)
    loss_step = loss_fn._current_step() if hasattr(loss_fn, "_current_step") else None
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
        if ema is not None and ema_snapshot is not None:
            ema.load_state_dict(ema_snapshot)
        if loss_fn is not None and loss_step is not None:
            loss_fn.set_step(int(loss_step))
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
    logging.getLogger(__name__).info("LR finder: %.3e -> %.3e", previous_lr, suggestion)
    return suggestion


def create_trainer(cfg: DictConfig) -> pl.Trainer:
    callback_names = getattr(cfg.trainer, "callbacks", []) or []
    # PL auto-adds a Timer when max_time is set; an explicit one drops max_time.
    callback_names = [n for n in callback_names if n != "timer"]
    callbacks = [_build_callback(name) for name in callback_names]
    no_val = bool(getattr(cfg, "fast_dev_run", False)) or getattr(
        cfg.trainer, "limit_val_batches", None
    ) in (0, 0.0)
    early_stopping = None if no_val else _build_early_stopping(cfg)
    if early_stopping is not None:
        callbacks.append(early_stopping)
    collapse_guard = _build_collapse_guard(cfg)
    if collapse_guard is not None:
        callbacks.append(collapse_guard)
    ckpt_cfg = getattr(cfg.trainer, "checkpoint_callback", None)
    if ckpt_cfg:
        container = OmegaConf.to_container(ckpt_cfg, resolve=True)
        if not isinstance(container, dict):
            raise TypeError(f"checkpoint config must be a dict, got {type(container).__name__}")
        kwargs_ckpt = {str(k): v for k, v in container.items()}
        if no_val and str(kwargs_ckpt.get("monitor", "")).startswith("val/"):
            # Val-free runs never log val/*: keep last-checkpoint mode only.
            kwargs_ckpt = {k: v for k, v in kwargs_ckpt.items() if k in ("dirpath", "save_last")}
            kwargs_ckpt.setdefault("save_last", True)
        # Drop the lazy val/loss checkpoint when an explicit one is configured.
        callbacks = [c for c in callbacks if not isinstance(c, ModelCheckpoint)]
        callbacks.append(ModelCheckpoint(**kwargs_ckpt))
    else:
        # No validation (fast_dev_run / limit_val_batches=0) breaks the
        # default val/loss monitor — fall back to last-checkpoint mode.
        if no_val:
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
        "min_steps",
        "max_time",
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
