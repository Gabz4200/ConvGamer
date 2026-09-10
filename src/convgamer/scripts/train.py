"""Hydra-CLI entrypoint for training ConvGamer / InceptionNeXt.

Run via ``uv run train`` or ``python -m convgamer.scripts.train``.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from convgamer.data.datamemodule import ConvGamerDataModule
from convgamer.modules.lightning_module import ConvGamerModel
from convgamer.training.engine import create_trainer

# Absolute config path so Hydra finds configs regardless of CWD or __main__.
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.fast_dev_run:
        cfg.trainer.max_epochs = 1
    trainer = create_trainer(cfg)
    model = ConvGamerModel(cfg)
    datamodule = ConvGamerDataModule(**cfg.data)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
