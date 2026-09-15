"""Hydra-CLI entrypoint for evaluating ConvGamer / InceptionNeXt / V-JEPA.

Run via ``uv run eval`` or ``python -m convgamer.scripts.eval``.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from convgamer.data.datamemodule import ConvGamerDataModule
from convgamer.data.jepa_datamodule import VJEPAGamingDataModule
from convgamer.training.engine import create_trainer

from .common import CONFIG_DIR, build_model, model_class


def _build_datamodule(cfg: DictConfig):
    target = str(cfg.model.get("target", ""))
    if "jepa" in target.lower():
        mode = cfg.data.get("mode", "synthetic")
        return VJEPAGamingDataModule(
            batch_size=cfg.data.batch_size,
            num_workers=0,
            num_frames=cfg.data.num_frames,
            height=cfg.data.height,
            width=cfg.data.width,
            mask_ratio=cfg.data.mask_ratio,
            image_mask_ratio=cfg.data.get("image_mask_ratio", 0.1),
            to_oklab=cfg.data.to_oklab,
            video_dataset_ids=cfg.data.get("video_dataset_ids", []),
            image_dataset_ids=cfg.data.get("image_dataset_ids", []),
            regularization_dataset_ids=cfg.data.get("regularization_dataset_ids", []),
            data_dir=cfg.data.get("data_dir", "./data/jepa"),
            mode=mode,
            sample_stride=cfg.data.get("sample_stride", 1),
            max_frames=cfg.data.get("max_frames", 10_000),
        )
    return ConvGamerDataModule(**cfg.data)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    ckpt = Path(cfg.eval.checkpoint) if hasattr(cfg, "eval") else None
    cls = model_class(cfg)
    target = str(cfg.model.get("target", ""))
    if "jepa" in target.lower():
        model = build_model(cfg)
    else:
        model = (
            cls.load_from_checkpoint(str(ckpt), cfg=cfg) if ckpt and ckpt.exists() else cls(cfg)  # type: ignore[call-arg]
        )
    datamodule = _build_datamodule(cfg)
    trainer = create_trainer(cfg)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
