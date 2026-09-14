from __future__ import annotations

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import RandomImageDataset, RandomVideoDataset


def _dataset(kind: str, **kwargs):
    if kind == "video":
        return RandomVideoDataset(
            num_samples=kwargs["num_samples"],
            channels=kwargs["channels"],
            num_frames=kwargs.get("num_frames", 8),
            size=kwargs["image_size"],
            num_classes=kwargs["num_classes"],
        )
    return RandomImageDataset(
        num_samples=kwargs["num_samples"],
        channels=kwargs["channels"],
        size=kwargs["image_size"],
        num_classes=kwargs["num_classes"],
    )


class ConvGamerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 4,
        num_workers: int = 0,
        num_samples: int = 64,
        channels: int = 3,
        image_size: int = 32,
        num_classes: int = 10,
        modality: str = "image",
        num_frames: int = 8,
    ):
        super().__init__()
        if modality not in ("image", "video"):
            raise ValueError(f"modality must be 'image' or 'video', got {modality}")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_samples = num_samples
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.modality = modality
        self.num_frames = num_frames

    def _params(self) -> dict:
        return {
            "channels": self.channels,
            "image_size": self.image_size,
            "num_classes": self.num_classes,
            "num_frames": self.num_frames,
        }

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            self.train_ds = _dataset(self.modality, num_samples=self.num_samples, **self._params())
            val_samples = max(1, self.num_samples // 4)
            self.val_ds = _dataset(self.modality, num_samples=val_samples, **self._params())
        if stage in (None, "test", "validate"):
            if not hasattr(self, "val_ds"):
                val_samples = max(1, self.num_samples // 4)
                self.val_ds = _dataset(self.modality, num_samples=val_samples, **self._params())
            self.test_ds = self.val_ds

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        ds = getattr(self, "test_ds", getattr(self, "val_ds", None))
        if ds is None:
            raise RuntimeError("test_ds/val_ds not initialized; call setup() first")
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers)
