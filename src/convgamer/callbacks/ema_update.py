"""V-JEPA EMA target update as a ``pl.Callback`` (observer).

Runs after the optimizer step on every training batch (V-JEPA 2.1 §2.1):
the online encoder's weights have already been updated, so the shadow
averages the *new* online weights — identical semantics to the module hook
this replaces, but now the training objective lives in ``modules/`` and the
lifecycle side-effect lives here.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

__all__ = ["EMAUpdateCallback"]


class EMAUpdateCallback(pl.Callback):
    """Update the JEPA target encoder after each optimizer step."""

    def __init__(self) -> None:
        super().__init__()
        self._last_step: int = -1

    def on_before_zero_grad(
        self,
        trainer: pl.Trainer | None,
        pl_module: pl.LightningModule,
        optimizer,
    ) -> None:
        step = int(getattr(trainer, "global_step", -1)) if trainer is not None else -1
        if step == self._last_step:
            return
        self._last_step = step
        ema_encoder = getattr(pl_module, "ema_encoder", None)
        if ema_encoder is None:
            raise AttributeError(
                f"{type(pl_module).__name__} has no 'ema_encoder'; "
                "EMAUpdateCallback requires a module exposing ema_encoder.update()"
            )
        ema_encoder.update()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer | None,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # Back-compat for trainers that never call on_before_zero_grad
        # (e.g. manual optimization): fall back only if no optimizer step ran.
        if trainer is not None and getattr(trainer, "global_step", -1) == self._last_step:
            return
        # If gradient accumulation is active, batch-end fires per micro-batch;
        # the on_before_zero_grad hook above already handles the true step.
        if trainer is not None and getattr(trainer, "accumulate_grad_batches", 1) != 1:
            return
        ema_encoder = getattr(pl_module, "ema_encoder", None)
        if ema_encoder is None:
            raise AttributeError(
                f"{type(pl_module).__name__} has no 'ema_encoder'; "
                "EMAUpdateCallback requires a module exposing ema_encoder.update()"
            )
        ema_encoder.update()
