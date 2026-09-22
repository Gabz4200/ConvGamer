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

    def on_train_batch_end(
        self,
        trainer: pl.Trainer | None,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        ema_encoder = getattr(pl_module, "ema_encoder", None)
        if ema_encoder is None:
            raise AttributeError(
                f"{type(pl_module).__name__} has no 'ema_encoder'; "
                "EMAUpdateCallback requires a module exposing ema_encoder.update()"
            )
        ema_encoder.update()

