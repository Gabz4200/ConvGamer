"""Video/image datasets for JEPA training and offline smoke runs.

Real data only — no noise-synthetic datasets. Offline smoke runs use the
checked-in real game clip (``tests/fixtures/terraria16_u8.pt``), never random
tensors. Masking is multi-block at token resolution (see
:mod:`convgamer.data.masking`); it is strictly a train-time augmentation.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import os
import shutil
from itertools import zip_longest
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader as _DecordVideoReader
from decord import cpu as _decord_cpu
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from convgamer.data.masking import multiblock_mask_3d
from convgamer.data.sources import (
    _expand_archive_globs,
    _looks_like_hf_id,
    _resolve_image_paths,
    _resolve_video_paths,
    _shard_for_distributed,
    download_remote_file,
    list_remote_video_files,
    stride_cap,
)

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
        frames = F.interpolate(frames, size=(height, width), mode="bilinear", align_corners=False)
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


def _stable_seed(path: str) -> int:
    """Deterministic per-path salt, stable across ``PYTHONHASHSEED`` runs."""
    return int(hashlib.md5(path.encode()).hexdigest(), 16) % (2**16)


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


# Per-process sample counter. DataLoader workers are forked processes that
# persist across epochs within one fit(), so this salts every yielded sample
# uniquely per worker lifetime: repeated passes over the same file get fresh
# masks (and fresh RNG clip starts) instead of replaying pass 1 identically.
# Not reproducible across runs by design — variation is the point.
_SAMPLE_COUNTER = itertools.count()


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
        if repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats}")
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.mask_ratio = mask_ratio
        self.repeats = repeats
        self.fixture = fixture
        self.base_seed = base_seed

    def __len__(self) -> int:
        worker_id, num_workers = _worker_info()
        return len(range(self.repeats)[worker_id::num_workers])

    def __iter__(self):
        worker_id, num_workers = _worker_info()
        clip = _load_fixture_clip(self.fixture)  # (C, Tf, Hf, Wf)
        tf = clip.shape[1]
        if self.num_frames > tf:
            raise ValueError(f"num_frames={self.num_frames} exceeds fixture length {tf}")
        step = max(1, tf // self.num_frames)
        idx = torch.arange(0, tf, step)[: self.num_frames]
        frames = clip[:, idx]  # (C, T, Hf, Wf)
        frames = einops.rearrange(frames, "c t h w -> t c h w")
        frames = _resize_video_frames(frames, self.height, self.width)
        for n in range(self.repeats)[worker_id::num_workers]:
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
        streaming: When True, HF ids are listed (metadata only) and each
            ``.mp4`` is fetched just-in-time, one file at a time, so disk
            stays O(1 file) instead of O(repo). Required for Kaggle's ~20 GB
            working disk with the 5-repo gameplay list.
        max_files_per_repo: Cap remote ``.mp4`` per repo in non-streaming
            mode (bounded ``snapshot`` alternative).
        cleanup_downloads: In streaming mode, delete each downloaded file
            after yielding its sample and evict the scoped Hub cache every
            few files, so training loop is download-train-delete.
        subset_caps: Optional ``{repo_id: max_files}`` strided caps applied
            in streaming mode (regularize with a small slice of huge repos).
        oversample: Optional ``{repo_id: repeats}`` multipliers for small
            but important repos (repeats yield fresh clips/masks via the
            per-sample counter salt, never byte-identical replays).
        file_slice: ``"train"`` (default; drops each repo's ``val_tail``)
            or ``"val"`` (yields only the held-out tails, stable order).
        val_tail: Held-out files per repo for the ``"val"`` slice.
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
        streaming: bool = False,
        max_files_per_repo: int | None = None,
        cleanup_downloads: bool = False,
        subset_caps: dict | None = None,
        oversample: dict | None = None,
        file_slice: str = "train",
        val_tail: int = 2,
    ):
        if file_slice not in ("train", "val"):
            raise ValueError(f"file_slice must be 'train' or 'val', got {file_slice!r}")
        if sample_stride < 1:
            raise ValueError(f"sample_stride must be >= 1, got {sample_stride}")
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.mask_ratio = mask_ratio
        self.sample_stride = sample_stride
        self.max_frames = max_frames
        self.data_dir = data_dir
        self.streaming = streaming
        self.max_files_per_repo = max_files_per_repo
        self.cleanup_downloads = cleanup_downloads
        self.subset_caps = dict(subset_caps or {})
        self.oversample = dict(oversample or {})
        self.file_slice = file_slice
        self.val_tail = val_tail
        self._skipped = 0

    def _load_video(self, path: str) -> torch.Tensor:
        """Load frames from a single mp4 using decord.

        Returns: ``(T, C, H, W)`` float tensor in [0, 1].
        """
        vr = _DecordVideoReader(path, ctx=_decord_cpu(0))
        total = len(vr)
        if total == 0:
            raise ValueError(f"video has no frames: {path}")
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

    def _sample_from_path(self, path: str):
        """Load ``path`` and yield its ``(x, y, mask)`` triplet (or skip)."""
        logger = logging.getLogger(__name__)
        try:
            y = self._load_video(path)  # (T, C, H, W)
        except Exception as error:
            logger.warning("Skipping unreadable video %s (%s)", path, error)
            self._skipped += 1
            if self._skipped % 10 == 1:
                logger.warning("skipped %d unreadable videos so far", self._skipped)
            return
        # Deterministic per-worker stream under seed_everything(workers=True),
        # salted by a per-sample counter so repeated passes over one file
        # (multi-epoch streaming) never replay identical masks.
        seed = int(torch.initial_seed() % (2**31)) + _stable_seed(path)
        seed += next(_SAMPLE_COUNTER) % (2**16)
        mask = multiblock_mask_3d(
            self.num_frames, self.height, self.width, self.mask_ratio, seed=seed
        )
        x, y = _mask_video_clip(y, mask)
        # (C, T, H, W)
        yield x, y, mask

    def _iter_streaming_refs(self):
        """Yield local paths one file at a time (HF ids fetched just-in-time).

        Remote repos are listed without downloading; each ``.mp4`` is pulled
        via single-file download, yielded to the caller, then optionally
        deleted so disk stays flat across thousand-file repos.

        Sampling per pass: each repo contributes its listing minus the
        ``val_tail`` holdout (``"val"`` slice yields only those tails, in
        stable order), reduced by the strided ``subset_caps`` slice and
        multiplied by ``oversample`` repeats. Train groups round-robin
        across repos so every batch mixes domains instead of running
        repo-homogeneous stretches.
        """
        worker_id, num_workers = _worker_info()
        groups: list[list[tuple[str | None, str]]] = []
        for item in self.video_paths:
            group: list[tuple[str | None, str]]
            if _looks_like_hf_id(item) and not os.path.exists(item):
                remote = list_remote_video_files(item)
                if not remote:
                    raise ValueError(f"remote repo {item!r} lists no .mp4 files")
                if self.file_slice == "val":
                    tail = remote[-self.val_tail :] if self.val_tail > 0 else list(remote)
                    group = [(item, name) for name in tail]
                else:
                    if self.val_tail > 0 and len(remote) > self.val_tail:
                        pool = remote[: -self.val_tail]
                    else:
                        pool = list(remote)
                    if item in self.subset_caps:
                        pool = stride_cap(pool, int(self.subset_caps[item]))
                    repeat = max(1, int(self.oversample.get(item, 1)))
                    group = [(item, name) for name in pool] * repeat
            else:
                local = _expand_archive_globs(item, recursive=True)
                group = [(None, path) for path in local]
            if group:
                groups.append(group)
        if self.file_slice == "val":
            # Stable concatenated tails: same val batches every check.
            planned = [ref for group in groups for ref in group]
        else:
            planned = [ref for tup in zip_longest(*groups) for ref in tup if ref is not None]
        planned = _shard_for_distributed(planned)
        if num_workers > 1:
            planned = planned[worker_id::num_workers]
        seen_roots: set[str] = set()
        for n, (repo_id, name) in enumerate(planned):
            if repo_id is None:
                yield name
                continue
            local_path = download_remote_file(repo_id, name, data_dir=self.data_dir)
            yield local_path
            if self.cleanup_downloads:
                try:
                    if os.path.isfile(local_path):
                        os.remove(local_path)
                except OSError:
                    pass
                # Evict the scoped Hub cache across touched repos every few
                # files so the cache-blob half of downloads never accumulates.
                seen_roots.add(os.path.join(self.data_dir, repo_id.replace("/", "__")))
                if (n + 1) % 5 == 0:
                    for root in seen_roots:
                        shutil.rmtree(os.path.join(root, ".hf_cache"), ignore_errors=True)
        if self.cleanup_downloads:
            for root in seen_roots:
                shutil.rmtree(os.path.join(root, ".hf_cache"), ignore_errors=True)

    def __iter__(self):
        if self.streaming:
            for path in self._iter_streaming_refs():
                yield from self._sample_from_path(path)
            return
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
            max_files_per_repo=self.max_files_per_repo,
        )
        for path in resolved:
            yield from self._sample_from_path(path)


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

            seed = int(torch.initial_seed() % (2**31)) + _stable_seed(path)
            seed += next(_SAMPLE_COUNTER) % (2**16)
            mask = multiblock_mask_3d(1, self.size, self.size, self.mask_ratio, seed=seed)
            # Mask after normalization with the mean (0.0 post-norm).
            y = _normalize_rgb(y, dim=0)
            x = y.clone()
            # y is (C, 1, H, W); mask is (T=1, H, W) -> align on (C, T) dims.
            x[mask.unsqueeze(0).expand_as(y)] = 0.0

            yield x, y, mask
