from __future__ import annotations

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
    return cfg


def main() -> None:
    cfg = _load_cfg()
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
