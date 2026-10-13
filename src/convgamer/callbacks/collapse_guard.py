"""Representation-collapse guard as a ``pl.Callback`` (observer).

JEPA prediction loss is an unreliable health signal on its own: a collapsed
encoder (constant output for every input) drives the loss toward zero while
learning nothing transferable. This callback watches the ``val/pred_std``
sentinel logged by ``ConvGamerVJEPAModel.validation_step`` — the std of the
predictor output over a val batch — and stops training after ``patience``
consecutive checks below ``min_std``. Healthy runs sit at O(0.1-1) and never
trigger it; only true collapse (or NaN) does.
"""

from __future__ import annotations

import math

import pytorch_lightning as pl

__all__ = ["CollapseGuardCallback"]


class CollapseGuardCallback(pl.Callback):
    """Stop training when predictor output std collapses toward zero."""

    def __init__(
        self,
        monitor: str = "val/pred_std",
        min_std: float = 1e-4,
        patience: int = 3,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.min_std = min_std
        self.patience = patience
        self._bad_checks = 0
        self._missing_checks = 0

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer | None,
        pl_module: pl.LightningModule,
    ) -> None:
        # Respect the trainer's min_steps floor: early checks run on
        # near-init weights whose statistics are meaningless.
        if trainer is not None:
            floor = int(getattr(trainer, "min_steps", 0) or 0)
            if int(getattr(trainer, "global_step", 0)) < floor:
                self._bad_checks = 0
                return
        metrics = trainer.callback_metrics if trainer is not None else {}
        value = metrics.get(self.monitor)
        if value is None:
            # Val ran without the sentinel (foreign module or logging off);
            # never stop on a missing signal, but warn so miswiring is visible.
            self._missing_checks += 1
            if self._missing_checks == self.patience:
                pl.utilities.rank_zero_warn(
                    f"CollapseGuardCallback: monitor {self.monitor!r} missing for "
                    f"{self._missing_checks} val checks; guard is idle."
                )
            return
        self._missing_checks = 0
        try:
            std = float(value)
        except (TypeError, ValueError):
            return
        if math.isnan(std) or std < self.min_std:
            self._bad_checks += 1
        else:
            self._bad_checks = 0
        if self._bad_checks >= self.patience and trainer is not None:
            trainer.should_stop = True
