"""Hydra-CLI entrypoint for training ConvGamer / InceptionNeXt / V-JEPA.

Run via ``uv run train`` or ``python -m convgamer.scripts.train``.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from convgamer.data.datamemodule import ConvGamerDataModule
from convgamer.data.jepa_datamodule import VJEPAGamingDataModule
from convgamer.training.engine import create_trainer

from .common import CONFIG_DIR, build_model


def _build_datamodule(cfg: DictConfig):
    """Select the datamodule based on whether JEPA is configured."""
    target = str(cfg.model.get("target", ""))
    if "jepa" in target.lower():
        mode = cfg.data.get("mode", "synthetic")
        if mode == "synthetic":
            return VJEPAGamingDataModule(
                batch_size=cfg.data.batch_size,
                num_workers=cfg.data.num_workers,
                num_frames=cfg.data.num_frames,
                height=cfg.data.height,
                width=cfg.data.width,
                mask_ratio=cfg.data.mask_ratio,
                to_oklab=cfg.data.to_oklab,
                mode=mode,
                num_synthetic_samples=cfg.data.get("num_synthetic_samples", 64),
            )
        return VJEPAGamingDataModule(
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            num_frames=cfg.data.num_frames,
            height=cfg.data.height,
            width=cfg.data.width,
            mask_ratio=cfg.data.mask_ratio,
            image_mask_ratio=cfg.data.get("image_mask_ratio", 0.1),
            to_oklab=cfg.data.to_oklab,
            video_dataset_ids=cfg.data.get("video_dataset_ids", []),
            image_dataset_ids=cfg.data.get("image_dataset_ids", []),
            data_dir=cfg.data.get("data_dir", "./data/jepa"),
            mode=mode,
            image_interval=cfg.data.get("image_interval", 5),
            sample_stride=cfg.data.get("sample_stride", 1),
            max_frames=cfg.data.get("max_frames", 10_000),
        )
    return ConvGamerDataModule(**cfg.data)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.fast_dev_run:
        cfg.trainer.max_epochs = 1
    trainer = create_trainer(cfg)
    model = build_model(cfg)
    datamodule = _build_datamodule(cfg)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
