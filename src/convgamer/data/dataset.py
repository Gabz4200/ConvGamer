from __future__ import annotations

import glob
import os

import einops
import torch
from torch.utils.data import Dataset, IterableDataset

from convgamer.data.oklab import srgb_to_oklab


def _resolve_video_paths(
    video_ids: list[str],
    data_dir: str = "./data/jepa",
    worker_id: int = 0,
    num_workers: int = 1,
) -> list[str]:
    """Expand HF dataset IDs or local globs into a flat MP4 list.

    HF IDs are downloaded once via ``huggingface_hub.snapshot_download``
    with an MP4 allow-pattern; local globs are expanded directly.
    """
    from huggingface_hub import snapshot_download

    paths: list[str] = []
    for item in video_ids:
        if "/" in item and not os.path.exists(item):
            local_dir = snapshot_download(
                repo_id=item,
                repo_type="dataset",
                local_dir=os.path.join(data_dir, item.replace("/", "__")),
                allow_patterns=["**/*.mp4", "*.mp4"],
            )
            paths.extend(sorted(glob.glob(os.path.join(local_dir, "**", "*.mp4"), recursive=True)))
        else:
            paths.extend(sorted(glob.glob(item, recursive=True)))
    if num_workers > 1:
        paths = paths[worker_id::num_workers]
    return paths


def _resolve_image_paths(
    image_ids: list[str],
    data_dir: str = "./data/jepa",
    worker_id: int = 0,
    num_workers: int = 1,
) -> list[str]:
    """Expand HF image dataset IDs or local globs into a flat image list."""
    from huggingface_hub import snapshot_download

    paths: list[str] = []
    for item in image_ids:
        if "/" in item and not os.path.exists(item):
            local_dir = snapshot_download(
                repo_id=item,
                repo_type="dataset",
                local_dir=os.path.join(data_dir, item.replace("/", "__")),
                allow_patterns=["**/*.jpg", "**/*.jpeg", "**/*.png", "*.jpg", "*.png"],
            )
            paths.extend(
                sorted(
                    glob.glob(os.path.join(local_dir, "**", "*.jpg"), recursive=True)
                    + glob.glob(os.path.join(local_dir, "**", "*.jpeg"), recursive=True)
                    + glob.glob(os.path.join(local_dir, "**", "*.png"), recursive=True)
                )
            )
        else:
            paths.extend(sorted(glob.glob(item, recursive=True)))
    if num_workers > 1:
        paths = paths[worker_id::num_workers]
    return paths


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


def oklab_convert_srgb(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Convert sRGB tensor to Oklab along channel dim ``dim`` (default: C in ``(B, C, ...)``).

    Supports both image ``(B, C, H, W)`` and video ``(B, C, T, H, W)`` tensors.
    """
    return srgb_to_oklab(x, dim=dim)


class RandomVideoIterableDataset(IterableDataset):
    """Infinite synthetic video stream for JEPA training smoke tests.

    Yields ``(x_masked, y_clean, mask)`` tuples where ``x`` is the
    masked view (mask tokens replaced with zeros), ``y`` is the clean
    view in oklab space, and ``mask`` is a boolean tensor.
    """

    def __init__(
        self,
        channels: int = 3,
        num_frames: int = 16,
        height: int = 224,
        width: int = 224,
        mask_ratio: float = 0.25,
        to_oklab: bool = True,
    ):
        self.channels = channels
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.mask_ratio = mask_ratio
        self.to_oklab = to_oklab

    def __iter__(self):
        while True:
            y = torch.rand(self.num_frames, self.channels, self.height, self.width)
            x = y.clone()
            mask = torch.rand(self.num_frames, self.height, self.width) < self.mask_ratio
            # Zero out masked positions in x (mask tokens)
            x[mask.unsqueeze(1).expand(-1, self.channels, -1, -1)] = 0.0

            if self.to_oklab:
                y = oklab_convert_srgb(y, dim=1)
                x = oklab_convert_srgb(x, dim=1)

            yield x, y, mask


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
            y = oklab_convert_srgb(y, dim=1)
            x = oklab_convert_srgb(x, dim=1)

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
        emit_hw: bool = True,
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
        self.emit_hw = emit_hw

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
            start = (
                0
                if total == self.num_frames * self.sample_stride
                else (
                    torch.randint(0, total - self.num_frames * self.sample_stride, (1,)).item()
                    if total > self.num_frames * self.sample_stride
                    else 0
                )
            )
            stride = self.sample_stride

        indices = list(
            range(int(start), min(int(start) + self.num_frames * stride, total), int(stride))
        )
        # Pad if needed
        if len(indices) < self.num_frames:
            indices += [indices[-1]] * (self.num_frames - len(indices))
        indices = indices[: self.num_frames]

        frames = vr.get_batch(indices).as_tensor()  # (T, H, W, C) or (T, H, W, C)
        if frames.ndim == 4:
            frames = einops.rearrange(frames, "t h w c -> t c h w")
        else:
            frames = einops.rearrange(frames, "t h w c -> t c h w")

        frames = frames.float() / 255.0
        # Resize if needed
        if frames.shape[-2:] != (self.height, self.width):
            import torchvision.transforms.functional as TF

            frames = torch.stack([TF.resize(f, [self.height, self.width]) for f in frames])
        return frames

    def __iter__(self):
        import logging

        import torch.utils.data as data_utils

        logger = logging.getLogger(__name__)
        info = data_utils.get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1
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

            if self.emit_hw and tuple(y.shape[-2:]) != (self.height, self.width):
                import torch.nn.functional as torch_f

                flat = einops.rearrange(y, "t c h w -> (t c) 1 h w")
                flat = torch_f.interpolate(
                    flat, size=(self.height, self.width), mode="bilinear", align_corners=False
                )
                y = einops.rearrange(flat, "(t c) 1 h w -> t c h w", t=self.num_frames)

            mask = torch.rand(self.num_frames, self.height, self.width) < self.mask_ratio
            x = y.clone()
            c = y.shape[1]
            x[mask.unsqueeze(1).expand(-1, c, -1, -1)] = 0.0

            if self.to_oklab:
                y = oklab_convert_srgb(y, dim=1)
                x = oklab_convert_srgb(x, dim=1)

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

        import torch.utils.data as data_utils
        from torchvision.io import read_image

        logger = logging.getLogger(__name__)
        info = data_utils.get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1
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

            img = img.float() / 255.0
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(self.size, self.size), mode="bilinear", align_corners=False
            ).squeeze(0)

            # Inject dummy temporal axis: (C, 1, H, W)
            y = img.unsqueeze(1)  # (C, 1, H, W)

            mask = torch.rand(1, self.size, self.size) < self.mask_ratio
            x = y.clone()
            x[mask.unsqueeze(0).expand(-1, y.shape[1], -1, -1)] = 0.0

            if self.to_oklab:
                y = oklab_convert_srgb(y, dim=1)
                x = oklab_convert_srgb(x, dim=1)

            yield x, y, mask


class MixedGameDataset(IterableDataset):
    """Interleave video clips and T=1 image views (Tier 1-3 combined).

    Args:
        video_dataset: Source ``GameVideoDataset`` shared with ``mode="video"``.
        image_dataset: Source ``GameImageDataset`` shared with ``mode="mixed"``.
        image_interval: Yield one image sample every ``image_interval`` samples.
    """

    def __init__(
        self,
        video_dataset: GameVideoDataset,
        image_dataset: GameImageDataset,
        image_interval: int = 5,
    ):
        self.video_dataset = video_dataset
        self.image_dataset = image_dataset
        self.image_interval = max(1, image_interval)

    def __iter__(self):
        video_iter = iter(self.video_dataset)
        image_iter = iter(self.image_dataset)
        step = 0
        while True:
            if step % self.image_interval == self.image_interval - 1:
                step += 1
                try:
                    yield next(image_iter)
                    continue
                except StopIteration:
                    break
            step += 1
            try:
                yield next(video_iter)
            except StopIteration:
                break
