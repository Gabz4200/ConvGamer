"""LightningDataModule for V-JEPA 2.1 ConvGamer pretraining.

Wires together gaming video datasets (Tier 1/2/7) and static-image
datasets (Tier 3) into a unified JEPA training pipeline.

Datasets configured in ``configs/data/jepa.yaml``:
    - video_datasets: list of HF dataset IDs or local paths
    - image_datasets: list of HF dataset IDs or local paths
    - num_frames: temporal chunk size (T=8-12 for T4)
    - mask_ratio: fraction of spatio-temporal tokens to mask
"""

from __future__ import annotations

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from convgamer.data.dataset import (
    GameImageDataset,
    GameVideoDataset,
    JEPADataset,
    MixedGameDataset,
)


class VJEPAGamingDataModule(pl.LightningDataModule):
    """DataModule for V-JEPA 2.1 pretraining on gaming data.

    Supports three modes:
    - ``"synthetic"``: ``JEPADataset`` for smoke testing (no downloads).
    - ``"video"``: ``GameVideoDataset`` iterable over MP4 files.
    - ``"mixed"``: video + image datasets interleaved (20% images).
    """

    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 4,
        num_frames: int = 12,
        height: int = 224,
        width: int = 224,
        mask_ratio: float = 0.25,
        image_mask_ratio: float = 0.1,
        to_oklab: bool = True,
        video_dataset_ids: list[str] | None = None,
        image_dataset_ids: list[str] | None = None,
        data_dir: str = "./data/jepa",
        mode: str = "synthetic",
        num_synthetic_samples: int = 64,
        image_interval: int = 5,
        sample_stride: int = 1,
        max_frames: int = 10_000,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.mask_ratio = mask_ratio
        self.image_mask_ratio = image_mask_ratio
        self.to_oklab = to_oklab
        self.video_dataset_ids = video_dataset_ids or []
        self.image_dataset_ids = image_dataset_ids or []
        self.data_dir = data_dir
        self.mode = mode
        self.num_synthetic_samples = num_synthetic_samples
        self.image_interval = image_interval  # inject 1 image every N video batches
        self.sample_stride = sample_stride
        self.max_frames = max_frames

    def _build_video_dataset(self) -> GameVideoDataset:
        return GameVideoDataset(
            video_paths=self.video_dataset_ids,
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            mask_ratio=self.mask_ratio,
            to_oklab=self.to_oklab,
            sample_stride=self.sample_stride,
            max_frames=self.max_frames,
            data_dir=self.data_dir,
        )

    def _build_image_dataset(self) -> GameImageDataset:
        return GameImageDataset(
            image_paths=self.image_dataset_ids,
            size=min(self.height, self.width),
            mask_ratio=self.image_mask_ratio,
            to_oklab=self.to_oklab,
            data_dir=self.data_dir,
        )

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        if self.mode == "synthetic":
            self.train_ds = JEPADataset(
                num_samples=self.num_synthetic_samples,
                channels=3,
                num_frames=self.num_frames,
                size=min(self.height, self.width),
                mask_ratio=self.mask_ratio,
                to_oklab=self.to_oklab,
            )
        elif self.mode == "video":
            self.train_ds = self._build_video_dataset()
        elif self.mode == "mixed":
            self.train_ds = MixedGameDataset(
                video_dataset=self._build_video_dataset(),
                image_dataset=self._build_image_dataset(),
                image_interval=self.image_interval,
            )
        else:
            raise ValueError(f"mode must be 'synthetic', 'video', or 'mixed', got {self.mode!r}")

    def train_dataloader(self) -> DataLoader:
        if not hasattr(self, "train_ds"):
            raise RuntimeError("train_ds not initialized; call setup() first")
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader | None:
        # For self-supervised pretraining, validation uses synthetic data
        val_ds = JEPADataset(
            num_samples=max(8, self.num_synthetic_samples // 4),
            channels=3,
            num_frames=self.num_frames,
            size=min(self.height, self.width),
            mask_ratio=self.mask_ratio,
            to_oklab=self.to_oklab,
        )
        return DataLoader(val_ds, batch_size=self.batch_size, num_workers=0)
