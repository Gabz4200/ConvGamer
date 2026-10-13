from __future__ import annotations

import einops
import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

from .dataset import FixtureFrameDataset, _load_fixture_clip, _normalize_rgb


class FixtureVideoDataset(Dataset):
    """Consecutive fixture-frame groups as ``((C, T, H, W), label)`` smoke clips."""

    def __init__(
        self,
        num_frames: int = 8,
        size: int = 32,
        num_classes: int = 10,
    ):
        clip = _load_fixture_clip()  # float (C, Tf, Hf, Wf) in [0, 1]
        tf = clip.shape[1]
        groups = tf // num_frames
        if groups < 1:
            raise ValueError(f"fixture has {tf} frames, need >= {num_frames}")
        clips = clip[:, : groups * num_frames].reshape(3, groups, num_frames, *clip.shape[2:])
        clips = einops.rearrange(clips, "c g t h w -> g c t h w")
        resized = torch.stack(
            [torch.stack([TF.resize(f, [size, size]) for f in clip]) for clip in clips]
        )
        # Match GameVideoDataset: ImageNet-standardized inputs.
        self.clips = _normalize_rgb(resized, dim=1)
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.clips.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.clips[index], torch.tensor(index % self.num_classes, dtype=torch.long)


class ConvGamerDataModule(pl.LightningDataModule):
    """Offline smoke datamodule over real game frames (no downloads, no noise)."""

    def __init__(
        self,
        batch_size: int = 4,
        num_workers: int = 0,
        channels: int = 3,
        image_size: int = 32,
        num_classes: int = 10,
        modality: str = "image",
        num_frames: int = 8,
    ):
        super().__init__()
        if modality not in ("image", "video"):
            raise ValueError(f"modality must be 'image' or 'video', got {modality}")
        if channels != 3:
            raise ValueError(f"fixture frames are RGB, got channels={channels}")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.num_classes = num_classes
        self.modality = modality
        self.num_frames = num_frames
        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self.test_ds: Dataset | None = None

    def _dataset(self) -> Dataset:
        if self.modality == "video":
            return FixtureVideoDataset(
                num_frames=self.num_frames,
                size=self.image_size,
                num_classes=self.num_classes,
            )
        return FixtureFrameDataset(size=self.image_size, num_classes=self.num_classes)

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            self.train_ds = self._dataset()
            self.val_ds = self._dataset()
        if stage in (None, "test", "validate"):
            if self.val_ds is None:
                self.val_ds = self._dataset()
            self.test_ds = self.val_ds

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("train_ds not initialized; call setup() first")
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("val_ds not initialized; call setup() first")
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        ds = self.test_ds if self.test_ds is not None else self.val_ds
        if ds is None:
            raise RuntimeError("test_ds/val_ds not initialized; call setup() first")
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers)
