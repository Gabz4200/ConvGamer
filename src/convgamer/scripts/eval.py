"""Hydra-CLI entrypoint for evaluating ConvGamer / InceptionNeXt / V-JEPA.

Run via ``uv run eval`` or ``python -m convgamer.scripts.eval``.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from convgamer.training.engine import create_trainer

from .common import (
    _JEPA_TARGET,
    CONFIG_DIR,
    build_datamodule,
    build_model,
    model_class,
)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    ckpt = Path(cfg.eval.checkpoint) if hasattr(cfg, "eval") else None
    cls = model_class(cfg)
    target = str(cfg.model.get("target", ""))
    if target == _JEPA_TARGET:
        model = build_model(cfg)
    else:
        model = (
            cls.load_from_checkpoint(str(ckpt), cfg=cfg) if ckpt and ckpt.exists() else cls(cfg)  # type: ignore[call-arg]
        )
    datamodule = build_datamodule(cfg, num_workers=0)
    trainer = create_trainer(cfg)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
