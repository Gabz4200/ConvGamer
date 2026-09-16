"""Hydra-CLI entrypoint for training ConvGamer / InceptionNeXt / V-JEPA.

Run via ``uv run train`` or ``python -m convgamer.scripts.train``.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from convgamer.training.engine import create_trainer

from .common import CONFIG_DIR, build_datamodule, build_model


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.fast_dev_run:
        cfg.trainer.max_epochs = 1
    trainer = create_trainer(cfg)
    model = build_model(cfg)
    datamodule = build_datamodule(cfg)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
