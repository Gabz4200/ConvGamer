from __future__ import annotations

import torch
from torch.utils.data import Dataset


class RandomImageDataset(Dataset):
    """Synthetic dataset — lets the smoke test run without downloading data."""

    def __init__(
        self,
        num_samples: int = 64,
        channels: int = 3,
        size: int = 32,
        num_classes: int = 10,
    ):
        self.num_samples = num_samples
        self.channels = channels
        self.size = size
        self.num_classes = num_classes
        self._data = torch.randn(num_samples, channels, size, size)
        self._labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._data[index], self._labels[index]


class RandomVideoDataset(Dataset):
    """Synthetic video dataset — (B, C, T, H, W) smoke input."""

    def __init__(
        self,
        num_samples: int = 64,
        channels: int = 3,
        num_frames: int = 8,
        size: int = 32,
        num_classes: int = 10,
    ):
        self.num_samples = num_samples
        self.channels = channels
        self.num_frames = num_frames
        self.size = size
        self.num_classes = num_classes
        self._data = torch.randn(num_samples, channels, num_frames, size, size)
        self._labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._data[index], self._labels[index]
