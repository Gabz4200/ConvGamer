"""Behavior tests for the streaming HF video dataset (Kinetics regularizer).

Seams tested:
- ``HFVideoDataset._load_frames`` decodes a bytes-buffered video into a
  ``(T, C, H, W)`` tensor and applies the same masking/oklab pipeline as
  ``GameVideoDataset``.
"""

from __future__ import annotations

import torch

from convgamer.data.dataset import HFVideoDataset


class _FakeVR:
    """Minimal stand-in for ``decord.VideoReader``."""

    def __init__(self, frames: int, h: int = 8, w: int = 8, c: int = 3):
        self._frames = torch.randint(0, 256, (frames, h, w, c), dtype=torch.uint8)

    def __len__(self) -> int:
        return len(self._frames)

    def get_batch(self, indices):
        class _Batch:
            def __init__(self, tensor: torch.Tensor):
                self._t = tensor

            def as_tensor(self) -> torch.Tensor:
                return self._t

        return _Batch(self._frames[indices])


def test_hf_video_dataset_loads_bytes_buffer(monkeypatch) -> None:
    """A bytes-buffered video row must decode to (C, T, H, W) and mask."""
    import convgamer.data.dataset as ds_mod

    fake = _FakeVR(frames=24)
    monkeypatch.setattr(ds_mod, "_DecordVideoReader", lambda *a, **k: fake)
    monkeypatch.setattr(ds_mod, "_decord_cpu", lambda *a, **k: None)

    dataset = HFVideoDataset(
        repo_id="fake/repo",
        num_frames=8,
        height=8,
        width=8,
        mask_ratio=0.25,
        to_oklab=True,
        max_rows=1,
    )
    y = dataset._load_frames(b"fake-bytes")
    assert y.shape == (8, 3, 8, 8)
    assert y.min() >= 0.0 and y.max() <= 1.0
