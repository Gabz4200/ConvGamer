"""Hydra-CLI entrypoint for evaluating ConvGamer / InceptionNeXt.

Run via ``uv run eval`` or ``python -m convgamer.scripts.eval``.
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
    ckpt = Path(cfg.eval.checkpoint) if hasattr(cfg, "eval") else None
    model = (
        ConvGamerModel.load_from_checkpoint(str(ckpt), cfg=cfg)
        if ckpt and ckpt.exists()
        else ConvGamerModel(cfg)
    )
    datamodule = ConvGamerDataModule(**cfg.data)
    trainer = create_trainer(cfg)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
