"""Video/image datasets for JEPA training and offline smoke runs.

Real data only — no noise-synthetic datasets. Offline smoke runs use the
checked-in real game clip (``tests/fixtures/terraria16_u8.pt``), never random
tensors. Masking is multi-block at token resolution (see
:mod:`convgamer.data.masking`); it is strictly a train-time augmentation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import einops
import numpy as np
import torch
import torchvision.transforms.functional as TF
from decord import VideoReader as _DecordVideoReader
from decord import cpu as _decord_cpu
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from convgamer.data.masking import multiblock_mask_3d
from convgamer.data.sources import _resolve_image_paths, _resolve_video_paths

__all__ = [
    "FixtureClipDataset",
    "FixtureFrameDataset",
    "GameImageDataset",
    "GameVideoDataset",
    "fixture_path",
]


def fixture_path() -> Path:
    """Locate the checked-in smoke clip, from CWD or the repo tree."""
    candidates = [
        Path("tests/fixtures/terraria16_u8.pt"),
        Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "terraria16_u8.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "smoke fixture not found; searched " + ", ".join(str(c) for c in candidates)
    )


def _resize_video_frames(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if frames.shape[-2:] != (height, width):
        frames = torch.stack([TF.resize(f, [height, width]) for f in frames])
    return frames


def _worker_info() -> tuple[int, int]:
    info = get_worker_info()
    worker_id = info.id if info is not None else 0
    num_workers = info.num_workers if info is not None else 1
    return worker_id, num_workers


# ImageNet standardization for sRGB inputs in [0, 1].
_RGB_MEAN = (0.485, 0.456, 0.406)
_RGB_STD = (0.229, 0.224, 0.225)


def _normalize_rgb(y: torch.Tensor, dim: int) -> torch.Tensor:
    mean = torch.tensor(_RGB_MEAN, device=y.device, dtype=y.dtype)
    std = torch.tensor(_RGB_STD, device=y.device, dtype=y.dtype)
    shape = [1] * y.ndim
    shape[dim] = 3
    return (y - mean.view(shape)) / std.view(shape)


def _mask_video_clip(y: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask ``y`` (T, C, H, W), standardize RGB, return (x, y) as (C, T, H, W).

    Masking happens *after* ImageNet normalization with the channel mean
    (0.0 post-norm) so masked tokens are neutral, not a constant outlier.
    """
    y = _normalize_rgb(y, dim=1)
    x = y.clone()
    c = y.shape[1]
    x[mask.unsqueeze(1).expand(-1, c, -1, -1)] = 0.0
    y = einops.rearrange(y, "t c h w -> c t h w")
    x = einops.rearrange(x, "t c h w -> c t h w")
    return x, y


def _load_fixture_clip(path: Path | None = None) -> torch.Tensor:
    """Load the smoke clip as float ``(C, Tf, Hf, Wf)`` in [0, 1]."""
    clip = torch.load(str(path or fixture_path()), weights_only=True)
    return clip.float() / 255.0


class FixtureClipDataset(IterableDataset):
    """Offline smoke dataset over the checked-in real game clip.

    Cycles the fixture clip ``repeats`` times, yielding ``(x, y, mask)``
    triplets with a fresh multi-block mask per sample. Finite length
    (``len == repeats``) so smoke runs terminate.
    """

    def __init__(
        self,
        num_frames: int = 16,
        height: int = 144,
        width: int = 256,
        mask_ratio: float = 0.25,
        repeats: int = 8,
        fixture: Path | None = None,
        base_seed: int = 0,
    ):
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.mask_ratio = mask_ratio
        self.repeats = repeats
        self.fixture = fixture
        self.base_seed = base_seed

    def __len__(self) -> int:
        return self.repeats

    def __iter__(self):
        clip = _load_fixture_clip(self.fixture)  # (C, Tf, Hf, Wf)
        tf = clip.shape[1]
        if self.num_frames > tf:
            raise ValueError(f"num_frames={self.num_frames} exceeds fixture length {tf}")
        step = max(1, tf // self.num_frames)
        idx = torch.arange(0, tf, step)[: self.num_frames]
        frames = clip[:, idx]  # (C, T, Hf, Wf)
        frames = einops.rearrange(frames, "c t h w -> t c h w")
        frames = _resize_video_frames(frames, self.height, self.width)
        for n in range(self.repeats):
            # Per-sample seed varies per epoch via base_seed so masks
            # don't freeze across epochs when the loader is re-iterated
            # with a new base_seed.
            mask = multiblock_mask_3d(
                self.num_frames,
                self.height,
                self.width,
                self.mask_ratio,
                seed=self.base_seed + n,
            )
            x, y = _mask_video_clip(frames, mask)
            yield x, y, mask


class FixtureFrameDataset(Dataset):
    """Real game frames as ``(image, label)`` pairs for classification smoke runs.

    Labels cycle deterministically over ``num_classes`` — they validate wiring,
    not quality.
    """

    def __init__(
        self,
        size: int = 32,
        num_classes: int = 10,
        fixture: Path | None = None,
    ):
        clip = _load_fixture_clip(fixture)  # (C, Tf, Hf, Wf)
        frames = einops.rearrange(clip, "c t h w -> t c h w")
        self.frames = _normalize_rgb(_resize_video_frames(frames, size, size), dim=1)
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.frames.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.frames[index], torch.tensor(index % self.num_classes, dtype=torch.long)


class GameVideoDataset(IterableDataset):
    """IterableDataset over gaming MP4 files using decord.

    Yields standardized-RGB ``(x_masked, y_clean, mask)``.
    Supports Tier 1 (direct mp4), Tier 2 (extracted mp4s), and
    Tier 7 (cuphead chunks).

    Args:
        video_paths: List of path templates; each can be a glob string.
        num_frames: Temporal chunk size T (16 for V-JEPA 2.1 App A).
        height, width: Target resize resolution.
        mask_ratio: Fraction of spatio-temporal tokens to mask.
        sample_stride: Frames to skip between sampled frames.
        max_frames: Cap on frames to read per clip (prevents OOM on massive files).
    """

    def __init__(
        self,
        video_paths: list[str],
        num_frames: int = 16,
        height: int = 144,
        width: int = 256,
        mask_ratio: float = 0.25,
        sample_stride: int = 1,
        max_frames: int = 10_000,
        data_dir: str = "./data/jepa",
    ):
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.mask_ratio = mask_ratio
        self.sample_stride = sample_stride
        self.max_frames = max_frames
        self.data_dir = data_dir

    def _load_video(self, path: str) -> torch.Tensor:
        """Load frames from a single mp4 using decord.

        Returns: ``(T, C, H, W)`` float tensor in [0, 1].
        """
        vr = _DecordVideoReader(path, ctx=_decord_cpu(0))
        total = len(vr)
        total = min(total, self.max_frames)

        if total < self.num_frames * self.sample_stride:
            # Short clip: wrap around instead of repeating the last frame
            # (avoids a static temporal edge the causal TCN could overfit).
            stride = 1
            indices = [(i % total) for i in range(self.num_frames)]
        else:
            stride = self.sample_stride
            span = total - self.num_frames * stride
            start = torch.randint(0, span, (1,)).item() if span > 0 else 0
            indices = list(
                range(int(start), min(int(start) + self.num_frames * stride, total), int(stride))
            )
            # Pad if needed
            if len(indices) < self.num_frames:
                indices += [indices[-1]] * (self.num_frames - len(indices))
            indices = indices[: self.num_frames]

        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C) uint8 from decord
        frames = torch.from_numpy(frames)
        frames = einops.rearrange(frames, "t h w c -> t c h w")

        frames = frames.float() / 255.0
        frames = _resize_video_frames(frames, self.height, self.width)
        return frames

    def __iter__(self):
        logger = logging.getLogger(__name__)
        worker_id, num_workers = _worker_info()
        # NOTE: path resolution/download runs per DataLoader worker. For
        # large-scale runs prefer resolving in DataModule.setup() (main
        # process) and passing resolved paths in; the per-worker call here
        # keeps the dataset self-contained for smoke/standalone use.
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

            # Deterministic per-worker stream under seed_everything(workers=True).
            seed = int(torch.initial_seed() % (2**31)) + hash(path) % (2**16)
            mask = multiblock_mask_3d(
                self.num_frames, self.height, self.width, self.mask_ratio, seed=seed
            )
            x, y = _mask_video_clip(y, mask)

            # (C, T, H, W)
            yield x, y, mask


class GameImageDataset(IterableDataset):
    """IterableDataset over static game images (T=1 spatial regularizer).

    Loads images via PIL (torchvision.io PNG decode conflicts with decord's
    FFmpeg symbols in-process), injects a dummy temporal axis, standardizes
    RGB, and yields ``(x_masked, y_clean, mask)``.
    """

    def __init__(
        self,
        image_paths: list[str],
        size: int = 224,
        mask_ratio: float = 0.1,
        data_dir: str = "./data/jepa",
    ):
        self.image_paths = image_paths
        self.size = size
        self.mask_ratio = mask_ratio
        self.data_dir = data_dir

    def __iter__(self):
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
                # PIL, not torchvision.io: torchvision's bundled libpng
                # segfaults once decord's FFmpeg is loaded in-process
                # (symbol interposition), while PIL decodes fine.
                with Image.open(path) as im:
                    img = torch.from_numpy(np.array(im.convert("RGB"))).permute(2, 0, 1)
            except Exception as error:
                logger.warning("Skipping unreadable image %s (%s)", path, error)
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

            seed = int(torch.initial_seed() % (2**31)) + hash(path) % (2**16)
            mask = multiblock_mask_3d(1, self.size, self.size, self.mask_ratio, seed=seed)
            # Mask after normalization with the mean (0.0 post-norm).
            y = _normalize_rgb(y, dim=0)
            x = y.clone()
            # y is (C, 1, H, W); mask is (T=1, H, W) -> align on (C, T) dims.
            x[mask.unsqueeze(0).expand_as(y)] = 0.0

            yield x, y, mask
