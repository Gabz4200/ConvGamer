"""Behavior tests for JEPA datasets and oklab conversion.

Seams tested:
- ``oklab_convert_srgb``: RGB tensor (B,C,H,W) -> Oklab (B,3,H,W).
- ``GameVideoDataset``: returns (B,C,T,H,W) video tensor from mp4 paths.
- ``GameImageDataset``: returns (B,C,H,W) from image paths -> oklab.
- ``JEPADataset``: yields (x_view, y_view, mask) where x is masked.
- ``MixedGameDataset``: interleaves video clips and T=1 images.
"""

from __future__ import annotations

import torch

from convgamer.data.dataset import (
    GameImageDataset,
    GameVideoDataset,
    JEPADataset,
    MixedGameDataset,
    _resolve_image_paths,
    _resolve_video_paths,
    oklab_convert_srgb,
)
from convgamer.data.oklab import srgb_to_oklab


def test_oklab_convert_srgb_matches_oklab_py_and_preserves_shape() -> None:
    """Our batch converter must match the per-tensor oklab module function."""
    rgb = torch.rand(2, 3, 8, 8)
    converted = oklab_convert_srgb(rgb)
    expected = srgb_to_oklab(rgb.clone(), dim=1)
    torch.testing.assert_close(converted, expected, atol=1e-5, rtol=1e-4)
    video = torch.rand(2, 3, 8, 16, 16)
    assert oklab_convert_srgb(video, dim=1).shape == video.shape


def test_jepa_dataset_yields_masked_x_and_clean_y() -> None:
    """JEPADataset returns x (has mask), y (clean), and mask tensor."""
    dataset = JEPADataset(
        num_samples=4,
        channels=3,
        num_frames=8,
        size=32,
        mask_ratio=0.25,
        to_oklab=True,
    )
    x, y, mask = dataset[0]
    assert x.shape == (3, 8, 32, 32)
    assert y.shape == (3, 8, 32, 32)
    assert mask.shape == (8, 32, 32)
    assert mask.dtype == torch.bool
    assert mask.sum() > 0  # some tokens masked
    assert not torch.equal(x, y)  # masked view differs from clean target


def test_jepa_dataset_oklab_output() -> None:
    """When to_oklab=True, x and y are in oklab space (L channel in [0,1])."""
    dataset = JEPADataset(
        num_samples=2,
        channels=3,
        num_frames=4,
        size=16,
        mask_ratio=0.3,
        to_oklab=True,
    )
    x, y, mask = dataset[0]
    # Oklab L channel should be in ~[0, 1] for sRGB inputs
    assert x.shape[0] == 3  # still 3 channels in oklab
    assert torch.isfinite(x).all()
    assert x[0].min() >= -0.05 and x[0].max() <= 1.05


def test_jepa_dataset_image_mode_temporal_axis() -> None:
    """T=1 image path: dummy temporal axis injected."""
    dataset = JEPADataset(
        num_samples=4,
        channels=3,
        num_frames=1,
        size=32,
        mask_ratio=0.1,
        to_oklab=True,
    )
    x, y, mask = dataset[0]
    assert x.shape == (3, 1, 32, 32)
    assert y.shape == (3, 1, 32, 32)
    assert mask.shape == (1, 32, 32)


def _touch(tmp_path, name: str) -> str:
    path = tmp_path / name
    path.write_bytes(b"placeholder")
    return str(path)


def test_resolve_video_paths_expands_globs_and_shards(tmp_path, monkeypatch) -> None:
    """Local globs resolve deterministically and split across workers."""
    for name in ("a.mp4", "b.mp4", "c.mp4", "d.mp4"):
        _touch(tmp_path, name)
    monkeypatch.chdir(tmp_path)
    expected = ["a.mp4", "b.mp4", "c.mp4", "d.mp4"]
    all_paths = _resolve_video_paths(["*.mp4"])
    assert all_paths == expected
    assert _resolve_video_paths(["*.mp4"], worker_id=0, num_workers=2) == expected[0::2]
    assert _resolve_video_paths(["*.mp4"], worker_id=1, num_workers=2) == expected[1::2]


def test_resolve_image_paths_expands_globs(tmp_path, monkeypatch) -> None:
    """Local image globs resolve to the flat sorted file list."""
    for name in ("x.jpg", "y.png"):
        _touch(tmp_path, name)
    monkeypatch.chdir(tmp_path)
    assert _resolve_image_paths(["*.jpg", "*.png"]) == ["x.jpg", "y.png"]


def test_mixed_dataset_interleaves_video_and_images() -> None:
    """Every Nth sample comes from the image source with T=1."""

    class ListVideo(GameVideoDataset):
        def __init__(self, samples: list):
            self.samples = samples

        def __iter__(self):
            yield from self.samples

    class ListImage(GameImageDataset):
        def __init__(self, samples: list):
            self.samples = samples

        def __iter__(self):
            yield from self.samples

    videos = [
        (torch.zeros(3, 2, 4, 4), torch.ones(3, 2, 4, 4), torch.zeros(2, 4, 4, dtype=torch.bool))
        for _ in range(5)
    ]
    images = [
        (torch.zeros(3, 1, 4, 4), torch.ones(3, 1, 4, 4), torch.zeros(1, 4, 4, dtype=torch.bool))
        for _ in range(3)
    ]
    mixed = MixedGameDataset(
        video_dataset=ListVideo(videos),
        image_dataset=ListImage(images),
        image_interval=2,
    )
    samples = list(mixed)
    assert len(samples) == 7
    assert [sample[0].shape[1] for sample in samples] == [2, 1, 2, 1, 2, 1, 2]
    assert sum(1 for sample in samples if sample[0].shape[1] == 1) == 3
