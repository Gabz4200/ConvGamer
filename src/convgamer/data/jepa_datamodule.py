"""LightningDataModule for V-JEPA 2.1 ConvGamer pretraining.

Wires together gaming video datasets and static-image datasets into a
unified JEPA training pipeline. Datasets configured in
``configs/data/jepa.yaml``.
"""

from __future__ import annotations

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from convgamer.data.dataset import (
    FixtureClipDataset,
    GameImageDataset,
    GameVideoDataset,
)


def _make_loader(ds, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, drop_last=True)


class VJEPAGamingDataModule(pl.LightningDataModule):
    """DataModule for V-JEPA 2.1 pretraining on gaming data.

    Supports three modes:
    - ``"fixture"``: the checked-in real game clip (offline smoke, no downloads).
    - ``"video"``: ``GameVideoDataset`` iterable over MP4 files.
    - ``"mixed"``: separate video and image dataloaders (V-JEPA 2.1 §2.3.3),
      which requires non-empty ``video_dataset_ids`` and
      ``image_dataset_ids``.

    In ``"mixed"`` mode the video and image dataloaders are returned as a
    list from ``train_dataloader``; Lightning alternates between them so
    each batch stays modality-homogeneous. Pair with
    ``accumulate_grad_batches=2`` so each optimizer step aggregates one
    video and one image batch, mirroring the paper's per-modality workers
    with aggregated gradients.
    """

    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 4,
        num_frames: int = 16,
        height: int = 144,
        width: int = 256,
        mask_ratio: float = 0.25,
        image_mask_ratio: float = 0.1,
        video_dataset_ids: list[str] | None = None,
        image_dataset_ids: list[str] | None = None,
        data_dir: str = "./data/jepa",
        mode: str = "fixture",
        sample_stride: int = 1,
        max_frames: int = 10_000,
        streaming: bool = False,
        max_files_per_repo: int | None = None,
        cleanup_downloads: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.mask_ratio = mask_ratio
        self.image_mask_ratio = image_mask_ratio
        self.video_dataset_ids = video_dataset_ids or []
        self.image_dataset_ids = image_dataset_ids or []
        self.data_dir = data_dir
        self.mode = mode
        self.sample_stride = sample_stride
        self.max_frames = max_frames
        self.streaming = streaming
        self.max_files_per_repo = max_files_per_repo
        self.cleanup_downloads = cleanup_downloads
        self.train_ds: FixtureClipDataset | GameVideoDataset | None = None
        self.video_ds: GameVideoDataset | None = None
        self.image_ds: GameImageDataset | None = None

    def _build_video_dataset(self, video_ids: list[str] | None = None) -> GameVideoDataset:
        return GameVideoDataset(
            video_paths=list(video_ids) if video_ids is not None else list(self.video_dataset_ids),
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            mask_ratio=self.mask_ratio,
            sample_stride=self.sample_stride,
            max_frames=self.max_frames,
            data_dir=self.data_dir,
            streaming=self.streaming,
            max_files_per_repo=self.max_files_per_repo,
            cleanup_downloads=self.cleanup_downloads,
        )

    def _build_image_dataset(self) -> GameImageDataset:
        # Spatial regularizer: square crop at the video height so the
        # encoder sees comparable scale (video is height x width).
        return GameImageDataset(
            image_paths=self.image_dataset_ids,
            size=min(self.height, self.width),
            mask_ratio=self.image_mask_ratio,
            data_dir=self.data_dir,
        )

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        if self.mode == "fixture":
            self.train_ds = FixtureClipDataset(
                num_frames=self.num_frames,
                height=self.height,
                width=self.width,
                mask_ratio=self.mask_ratio,
            )
        elif self.mode == "video":
            if not self.video_dataset_ids:
                raise ValueError("mode='video' requires non-empty video_dataset_ids")
            # Hold out the last file for validation.
            train_ids = self.video_dataset_ids[:-1] or self.video_dataset_ids
            self.train_ds = self._build_video_dataset(train_ids)
        elif self.mode == "mixed":
            if not self.video_dataset_ids or not self.image_dataset_ids:
                raise ValueError(
                    "mode='mixed' requires non-empty video_dataset_ids and image_dataset_ids"
                )
            self.video_ds = self._build_video_dataset(
                self.video_dataset_ids[:-1] or self.video_dataset_ids
            )
            self.image_ds = self._build_image_dataset()
            self.train_ds = self.video_ds
        else:
            raise ValueError(f"mode must be 'fixture', 'video', or 'mixed', got {self.mode!r}")

    def train_dataloader(self) -> DataLoader | list[DataLoader]:
        if self.train_ds is None:
            raise RuntimeError("train_ds not initialized; call setup() first")
        if self.mode == "mixed":
            if self.video_ds is None:
                raise RuntimeError("video_ds not initialized; call setup() first")
            if self.image_ds is None:
                raise RuntimeError("image_ds not initialized; call setup() first")
            video_dl = _make_loader(self.video_ds, self.batch_size, self.num_workers)
            image_dl = _make_loader(self.image_ds, self.batch_size, self.num_workers)
            return [video_dl, image_dl]
        return _make_loader(self.train_ds, self.batch_size, self.num_workers)

    def val_dataloader(self) -> DataLoader | None:
        # Validation holds out data: fixture uses fewer repeats with an
        # offset seed (disjoint masks); video holds out the last file.
        if self.mode == "fixture" or self.train_ds is None:
            val_ds = FixtureClipDataset(
                num_frames=self.num_frames,
                height=self.height,
                width=self.width,
                mask_ratio=self.mask_ratio,
                repeats=2,
                base_seed=10_000,
            )
        else:
            ids = list(self.video_dataset_ids)
            val_ids = ids[-1:] if ids else []
            if not val_ids:
                raise ValueError("no video ids to hold out for validation")
            # Streaming val stays tiny: 2 files max, no cleanup (reused across
            # val checks), so validation never snapshot-downloads a repo.
            val_ds = GameVideoDataset(
                video_paths=val_ids,
                num_frames=self.num_frames,
                height=self.height,
                width=self.width,
                mask_ratio=self.mask_ratio,
                sample_stride=self.sample_stride,
                max_frames=self.max_frames,
                data_dir=self.data_dir,
                streaming=self.streaming,
                max_files_per_repo=2 if self.streaming else self.max_files_per_repo,
                cleanup_downloads=False,
            )
        # num_workers=0: val resolves paths in the main process, avoiding
        # the per-worker download race noted in dataset.GameVideoDataset.
        return DataLoader(val_ds, batch_size=self.batch_size, num_workers=0)
