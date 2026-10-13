"""Hydra-CLI entrypoint for training ConvGamer / InceptionNeXt / V-JEPA.

Run via ``uv run train`` or ``python -m convgamer.scripts.train``.

Opt-in to Lightning's LR finder with ``+trainer.lr_finder.enabled=true``:
the suggestion is written back to ``model.lr`` before ``fit``.
"""

from __future__ import annotations

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from convgamer.training.engine import create_trainer, maybe_run_lr_finder

from .common import CONFIG_DIR, build_datamodule, build_model


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(int(cfg.get("seed", 42)), workers=True)
    # JEPA training is max_steps-driven; only force max_epochs for
    # epoch-driven (classification) trainers.
    if cfg.fast_dev_run and "max_epochs" in cfg.trainer and "max_steps" not in cfg.trainer:
        cfg.trainer.max_epochs = 1
    trainer = create_trainer(cfg)
    model = build_model(cfg)
    datamodule = build_datamodule(cfg)
    maybe_run_lr_finder(trainer, model, datamodule, cfg)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
