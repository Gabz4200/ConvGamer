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
    build_backbone,
    build_datamodule,
    build_model,
    system_for_backbone,
)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    target = str(cfg.model.get("target", ""))
    if target == _JEPA_TARGET:
        model = build_model(cfg)
    else:
        # Rebuild the pure backbone from the config, then either inject it
        # straight into its system wrapper or hand it to load_from_checkpoint
        # (whose stored hyper-parameters fill lr/weight_decay/model_kwargs).
        backbone = build_backbone(cfg)
        system_cls = system_for_backbone(backbone)
        ckpt = Path(cfg.eval.checkpoint) if hasattr(cfg, "eval") else None
        model = (
            system_cls.load_from_checkpoint(str(ckpt), model=backbone)
            if ckpt is not None and ckpt.exists()
            else build_model(cfg)
        )
    datamodule = build_datamodule(cfg, num_workers=0)
    trainer = create_trainer(cfg)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
