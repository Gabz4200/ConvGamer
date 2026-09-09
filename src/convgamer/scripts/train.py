from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from convgamer.data.datamemodule import ConvGamerDataModule
from convgamer.modules.lightning_module import ConvGamerModel
from convgamer.training.engine import create_trainer

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"


def _load_cfg() -> DictConfig:
    cfg: DictConfig = OmegaConf.create({})
    for name in ("model", "data", "trainer", "ops"):
        cfg = OmegaConf.merge(cfg, OmegaConf.load(_CONFIG_DIR / f"{name}.yaml"))  # type: ignore[assignment]
    return OmegaConf.merge(cfg, OmegaConf.create({"fast_dev_run": False}))  # type: ignore[return-value]


def main() -> None:
    cfg = _load_cfg()
    if "--fast-dev-run" in sys.argv:
        cfg.fast_dev_run = True
    if cfg.fast_dev_run:
        cfg.trainer.max_epochs = 1
    trainer = create_trainer(cfg)
    model = ConvGamerModel(cfg)
    datamodule = ConvGamerDataModule(**cfg.data)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
