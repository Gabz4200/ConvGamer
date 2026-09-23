from __future__ import annotations

import einops
import torch
from decord import VideoReader as _DecordVideoReader
from decord import cpu as _decord_cpu
from torch.utils.data import Dataset, IterableDataset

from convgamer.data.oklab import srgb_to_oklab
from convgamer.data.sources import _resolve_image_paths, _resolve_video_paths


def _resize_video_frames(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if frames.shape[-2:] != (height, width):
        import torchvision.transforms.functional as TF

        frames = torch.stack([TF.resize(f, [height, width]) for f in frames])
    return frames


def _worker_info() -> tuple[int, int]:
    import torch.utils.data as data_utils

    info = data_utils.get_worker_info()
    worker_id = info.id if info is not None else 0
    num_workers = info.num_workers if info is not None else 1
    return worker_id, num_workers


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


class JEPADataset(Dataset):
    """Synthetic JEPA dataset for smoke testing.

    Yields ``(x_view, y_view, mask)`` where:
    - ``x_view``: masked input in oklab space, shape ``(C, T, H, W)``.
    - ``y_view``: clean target in oklab space, shape ``(C, T, H, W)``.
    - ``mask``: boolean mask ``(T, H, W)`` — True at masked positions.

    With ``num_frames=1`` (static image mode), a dummy temporal axis
    is injected so the same pipeline handles T=1 and T>1.
    """

    def __init__(
        self,
        num_samples: int = 64,
        channels: int = 3,
        num_frames: int = 16,
        size: int = 224,
        mask_ratio: float = 0.25,
        to_oklab: bool = True,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.channels = channels
        self.num_frames = num_frames
        self.size = size
        self.mask_ratio = mask_ratio
        self.to_oklab = to_oklab
        g = torch.Generator().manual_seed(seed)
        shape = (num_samples, num_frames, channels, size, size)
        self._data = torch.rand(*shape, generator=g)
        self._masks = [self._generate_mask(g) for _ in range(num_samples)]

    def _generate_mask(self, g: torch.Generator) -> torch.Tensor:
        mask = torch.rand(self.num_frames, self.size, self.size, generator=g) < self.mask_ratio
        return mask

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self._data[index]  # (T, C, H, W)
        mask = self._masks[index]  # (T, H, W)

        x = y.clone()
        x[mask.unsqueeze(1).expand(-1, self.channels, -1, -1)] = 0.0

        if self.to_oklab:
            y = srgb_to_oklab(y, dim=1)
            x = srgb_to_oklab(x, dim=1)

        # Rearrange to (C, T, H, W)
        y = einops.rearrange(y, "t c h w -> c t h w")
        x = einops.rearrange(x, "t c h w -> c t h w")
        return x, y, mask


class GameVideoDataset(IterableDataset):
    """IterableDataset over gaming MP4 files using decord.

    Yields ``(x_masked, y_clean, mask)`` in oklab space.
    Supports Tier 1 (direct mp4), Tier 2 (extracted mp4s), and
    Tier 7 (cuphead chunks).

    Args:
        video_paths: List of path templates; each can be a glob string.
        num_frames: Temporal chunk size T (8-12 for T4 VRAM).
        height, width: Target resize resolution.
        mask_ratio: Fraction of spatio-temporal tokens to mask.
        to_oklab: Convert sRGB -> Oklab.
        sample_stride: Frames to skip between sampled frames.
        max_frames: Cap on frames to read per clip (prevents OOM on massive files).
    """

    def __init__(
        self,
        video_paths: list[str],
        num_frames: int = 12,
        height: int = 224,
        width: int = 224,
        mask_ratio: float = 0.25,
        to_oklab: bool = True,
        sample_stride: int = 1,
        max_frames: int = 10_000,
        data_dir: str = "./data/jepa",
    ):
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.mask_ratio = mask_ratio
        self.to_oklab = to_oklab
        self.sample_stride = sample_stride
        self.max_frames = max_frames
        self.data_dir = data_dir

    def _load_video(self, path: str) -> torch.Tensor:
        """Load frames from a single mp4 using decord.

        Returns: ``(T, C, H, W)`` float tensor in [0, 1].
        """
        from decord import VideoReader, cpu

        vr = VideoReader(path, ctx=cpu(0))
        total = len(vr)
        total = min(total, self.max_frames)

        if total < self.num_frames * self.sample_stride:
            # Pad by repeating last frame if clip too short
            start = 0
            stride = 1
        else:
            if total == self.num_frames * self.sample_stride:
                start = 0
            else:
                start = torch.randint(0, total - self.num_frames * self.sample_stride, (1,)).item()
            stride = self.sample_stride

        indices = list(
            range(int(start), min(int(start) + self.num_frames * stride, total), int(stride))
        )
        # Pad if needed
        if len(indices) < self.num_frames:
            indices += [indices[-1]] * (self.num_frames - len(indices))
        indices = indices[: self.num_frames]

        frames = vr.get_batch(indices).as_tensor()  # (T, H, W, C) from decord
        frames = einops.rearrange(frames, "t h w c -> t c h w")

        frames = frames.float() / 255.0
        frames = _resize_video_frames(frames, self.height, self.width)
        return frames

    def __iter__(self):
        import logging

        logger = logging.getLogger(__name__)
        worker_id, num_workers = _worker_info()
        resolved = _resolve_video_paths(
            self.video_paths,
            data_dir=self.data_dir,
            worker_id=worker_id,
            num_workers=num_workers,
        )
        for path in resolved:
            try:
                y = self._load_video(path)  # (T, C, H, W)
            except Exception as error:
                logger.warning("Skipping unreadable video %s (%s)", path, error)
                continue

            mask = torch.rand(self.num_frames, self.height, self.width) < self.mask_ratio
            x = y.clone()
            c = y.shape[1]
            x[mask.unsqueeze(1).expand(-1, c, -1, -1)] = 0.0

            if self.to_oklab:
                y = srgb_to_oklab(y, dim=1)
                x = srgb_to_oklab(x, dim=1)

            # (C, T, H, W)
            y = einops.rearrange(y, "t c h w -> c t h w")
            x = einops.rearrange(x, "t c h w -> c t h w")
            yield x, y, mask


class GameImageDataset(IterableDataset):
    """IterableDataset over static game images (T=1 spatial regularizer).

    Loads JPEG/PNG via torchvision.io, injects a dummy temporal axis,
    converts to oklab, and yields ``(x_masked, y_clean, mask)``.
    """

    def __init__(
        self,
        image_paths: list[str],
        size: int = 224,
        mask_ratio: float = 0.1,
        to_oklab: bool = True,
        data_dir: str = "./data/jepa",
    ):
        self.image_paths = image_paths
        self.size = size
        self.mask_ratio = mask_ratio
        self.to_oklab = to_oklab
        self.data_dir = data_dir

    def __iter__(self):
        import logging

        from torchvision.io import read_image

        logger = logging.getLogger(__name__)
        worker_id, num_workers = _worker_info()
        resolved = _resolve_image_paths(
            self.image_paths,
            data_dir=self.data_dir,
            worker_id=worker_id,
            num_workers=num_workers,
        )

        for path in resolved:
            try:
                img = read_image(path)  # (C, H, W) uint8
            except Exception as error:
                logger.warning("Skipping unreadable image %s (%s)", path, error)
                continue

            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            elif img.shape[0] == 4:
                img = img[:3]
            if img.shape[0] != 3:
                logger.warning("Skipping non-RGB image %s (C=%d)", path, img.shape[0])
                continue

            img = img.float() / 255.0
            img = (
                torch.nn.functional.interpolate(
                    img.unsqueeze(0),
                    size=(self.size, self.size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                if img.shape[-2:] != (self.size, self.size)
                else img
            )

            # Inject dummy temporal axis: (C, 1, H, W)
            y = img.unsqueeze(1)  # (C, 1, H, W)

            mask = torch.rand(1, self.size, self.size) < self.mask_ratio
            x = y.clone()
            # y is (C, 1, H, W); mask is (T=1, H, W) -> align on (C, T) dims.
            x[mask.unsqueeze(0).expand_as(y)] = 0.0

            if self.to_oklab:
                # Single sample (C, T, H, W): channel lives at dim 0.
                y = srgb_to_oklab(y, dim=0)
                x = srgb_to_oklab(x, dim=0)

            yield x, y, mask


class HFVideoDataset(IterableDataset):
    """IterableDataset over Hugging Face video datasets (Kinetics, UCF101).

    Streams rows from ``datasets`` (parquet/zip-backed repos) and decodes
    the video column with ``decord`` — no snapshot download of raw MP4s.
    Useful as a regularization source (the "ImageNet of videos") alongside
    the gaming MP4 pipeline.

    Args:
        repo_id: HF dataset id, e.g. ``"kiyoonkim/kinetics-400-splits"``.
        video_column: name of the column holding the video (path or bytes).
        split: HF split to stream.
        num_frames: temporal chunk size T.
        height, width: target resize resolution.
        mask_ratio: fraction of spatio-temporal tokens to mask.
        to_oklab: convert sRGB -> Oklab.
        max_rows: cap rows per worker (prevents unbounded iteration).
    """

    def __init__(
        self,
        repo_id: str,
        video_column: str = "0",
        split: str = "train",
        num_frames: int = 16,
        height: int = 224,
        width: int = 224,
        mask_ratio: float = 0.25,
        to_oklab: bool = True,
        max_rows: int | None = None,
        data_dir: str = "./data/jepa",
    ):
        self.repo_id = repo_id
        self.video_column = video_column
        self.split = split
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.mask_ratio = mask_ratio
        self.to_oklab = to_oklab
        self.max_rows = max_rows
        self.data_dir = data_dir

    def _load_frames(self, source) -> torch.Tensor:
        """Decode ``source`` (path or bytes) into a ``(T, C, H, W)`` tensor."""
        reader = _DecordVideoReader
        cpu = _decord_cpu
        if reader is None or cpu is None:
            raise ImportError(
                "decord is required for HFVideoDataset; install with `pip install decord`"
            )
        if isinstance(source, (bytes, bytearray)):
            import io

            vr = reader(io.BytesIO(bytes(source)), ctx=cpu(0))
        else:
            vr = reader(str(source), ctx=cpu(0))
        total = len(vr)
        total = min(total, 10_000)
        if total < self.num_frames:
            indices = list(range(total)) + [total - 1] * (self.num_frames - total)
        else:
            start = int(torch.randint(0, total - self.num_frames, (1,)).item())
            indices = list(range(start, start + self.num_frames))
        frames = vr.get_batch(indices).as_tensor()
        frames = einops.rearrange(frames, "t h w c -> t c h w")
        frames = frames.float() / 255.0
        frames = _resize_video_frames(frames, self.height, self.width)
        return frames

    def __iter__(self):
        import logging

        from datasets import load_dataset

        logger = logging.getLogger(__name__)
        worker_id, num_workers = _worker_info()
        ds = load_dataset(self.repo_id, split=self.split, streaming=True)
        if num_workers > 1:
            ds = ds.shard(num_shards=num_workers, index=worker_id)
        count = 0
        for row in ds:
            if self.max_rows is not None and count >= self.max_rows:
                break
            source = row.get(self.video_column, row.get("video"))
            if source is None:
                continue
            try:
                y = self._load_frames(source)
            except Exception as error:
                logger.warning("Skipping unreadable row %s (%s)", source, error)
                continue

            mask = torch.rand(self.num_frames, self.height, self.width) < self.mask_ratio
            x = y.clone()
            c = y.shape[1]
            x[mask.unsqueeze(1).expand(-1, c, -1, -1)] = 0.0

            if self.to_oklab:
                y = srgb_to_oklab(y, dim=1)
                x = srgb_to_oklab(x, dim=1)

            y = einops.rearrange(y, "t c h w -> c t h w")
            x = einops.rearrange(x, "t c h w -> c t h w")
            count += 1
            yield x, y, mask
