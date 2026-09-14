from __future__ import annotations

import torch
from torch.utils.data import Dataset


class RandomSyntheticDataset(Dataset):
    """Synthetic smoke dataset for images and video.

    With ``num_frames=None`` the sample shape is ``(C, H, W)``; otherwise
    ``(C, T, H, W)``.
    """

    def __init__(
        self,
        num_samples: int = 64,
        channels: int = 3,
        size: int = 32,
        num_classes: int = 10,
        num_frames: int | None = None,
    ):
        self.num_samples = num_samples
        self.channels = channels
        self.size = size
        self.num_classes = num_classes
        self.num_frames = num_frames
        shape = (
            (num_samples, channels, size, size)
            if num_frames is None
            else (num_samples, channels, num_frames, size, size)
        )
        self._data = torch.randn(*shape)
        self._labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._data[index], self._labels[index]


class RandomImageDataset(RandomSyntheticDataset):
    """Synthetic image dataset — lets the smoke test run without downloading data."""

    def __init__(
        self,
        num_samples: int = 64,
        channels: int = 3,
        size: int = 32,
        num_classes: int = 10,
    ):
        super().__init__(
            num_samples=num_samples,
            channels=channels,
            size=size,
            num_classes=num_classes,
            num_frames=None,
        )


class RandomVideoDataset(RandomSyntheticDataset):
    """Synthetic video dataset — (B, C, T, H, W) smoke input."""

    def __init__(
        self,
        num_samples: int = 64,
        channels: int = 3,
        num_frames: int = 8,
        size: int = 32,
        num_classes: int = 10,
    ):
        super().__init__(
            num_samples=num_samples,
            channels=channels,
            size=size,
            num_classes=num_classes,
            num_frames=num_frames,
        )
