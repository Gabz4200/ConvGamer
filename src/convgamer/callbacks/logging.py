from __future__ import annotations

import logging
from collections.abc import Callable

from pytorch_lightning import Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)

logger = logging.getLogger(__name__)


class ConvGamerLogger(Callback):
    """Generic metric logger callback."""

    def on_validation_epoch_end(self, trainer, pl_module):  # noqa: ARG002
        for name, val in trainer.callback_metrics.items():
            if name.startswith("val"):
                logger.info("%s=%.4f", name, float(val))


_BUILDER_FOR_LAZY: dict[str, Callable[[], Callback]] = {
    "model_checkpoint": lambda: ModelCheckpoint(
        monitor="val_loss", save_last=True, save_top_k=1, mode="min"
    ),
    "early_stopping": lambda: EarlyStopping(monitor="val_loss", mode="min", patience=5),
    "taichi_init": lambda: __import__(
        "convgamer.callbacks.taichi_init", fromlist=["TaichiInitCallback"]
    ).TaichiInitCallback(),
    "logger": lambda: ConvGamerLogger(),
}


def _build_callback(name: str) -> Callback:
    """Factory matching the names in ``configs/trainer.yaml``."""
    try:
        return _BUILDER_FOR_LAZY[name]()
    except KeyError:
        raise ValueError(f"Unknown callback '{name}'") from None


__all__ = ["_build_callback", "ConvGamerLogger"]
