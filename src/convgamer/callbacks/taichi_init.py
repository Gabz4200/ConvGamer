from __future__ import annotations

import pytorch_lightning as pl

from convgamer.kernels.taichi.runtime import ensure_initialized


class TaichiInitCallback(pl.Callback):
    """Initialize Taichi once at fit start — never at import time."""

    def on_fit_start(self, trainer, pl_module):
        ensure_initialized()
