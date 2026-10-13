"""Hydra-CLI entrypoint for evaluating ConvGamer / InceptionNeXt / V-JEPA.

Run via ``uv run eval`` or ``python -m convgamer.scripts.eval``.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from convgamer.modules.lightning_module import BackboneClassifier
from convgamer.training.engine import create_trainer

from .common import (
    _JEPA_TARGET,
    CONFIG_DIR,
    build_backbone,
    build_datamodule,
)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    target = str(cfg.model.get("target", ""))
    if target == _JEPA_TARGET:
        if not hasattr(cfg, "eval") or getattr(cfg.eval, "checkpoint", None) is None:
            raise ValueError("JEPA eval requires +eval.checkpoint=<path>")
        from convgamer.modules.jepa_module import ConvGamerVJEPAModel

        ckpt = Path(str(cfg.eval.checkpoint))
        if not ckpt.exists():
            raise FileNotFoundError(f"eval checkpoint missing: {ckpt}")
        model = ConvGamerVJEPAModel.load_from_checkpoint(str(ckpt))
    else:
        # Rebuild the pure backbone from the config, then either inject it
        # straight into its system wrapper or hand it to load_from_checkpoint
        # (whose stored hyper-parameters fill lr/weight_decay/model_kwargs).
        backbone = build_backbone(cfg)
        ckpt = Path(str(cfg.eval.checkpoint)) if hasattr(cfg, "eval") else None
        if ckpt is None or not ckpt.exists():
            raise FileNotFoundError(f"eval checkpoint missing: {ckpt}")
        model = BackboneClassifier.load_from_checkpoint(str(ckpt), model=backbone)
    datamodule = build_datamodule(cfg, num_workers=0)
    trainer = create_trainer(cfg)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
