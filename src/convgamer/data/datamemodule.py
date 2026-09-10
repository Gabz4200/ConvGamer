from __future__ import annotations

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import RandomImageDataset


class ConvGamerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 4,
        num_workers: int = 0,
        num_samples: int = 64,
        channels: int = 3,
        image_size: int = 32,
        num_classes: int = 10,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_samples = num_samples
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            self.train_ds = RandomImageDataset(
                num_samples=self.num_samples,
                channels=self.channels,
                size=self.image_size,
                num_classes=self.num_classes,
            )
            val_samples = max(1, self.num_samples // 4)
            self.val_ds = RandomImageDataset(
                num_samples=val_samples,
                channels=self.channels,
                size=self.image_size,
                num_classes=self.num_classes,
            )
        if stage in (None, "test", "validate"):
            if not hasattr(self, "val_ds"):
                val_samples = max(1, self.num_samples // 4)
                self.val_ds = RandomImageDataset(
                    num_samples=val_samples,
                    channels=self.channels,
                    size=self.image_size,
                    num_classes=self.num_classes,
                )
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
