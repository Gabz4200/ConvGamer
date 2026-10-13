"""Behavior tests for real-data JEPA datasets and RGB normalization.

Seams tested:
- ``FixtureClipDataset``: yields (x_view, y_view, mask) from the checked-in
  real clip, where x is masked.
- ``_normalize_rgb``: ImageNet standardization of [0, 1] sRGB.
"""

from __future__ import annotations

import torch

from convgamer.data.dataset import (
    FixtureClipDataset,
    _normalize_rgb,
    _resolve_image_paths,
    _resolve_video_paths,
)


def test_normalize_rgb_standardizes_channels() -> None:
    """ImageNet mean/std normalization on [0, 1] sRGB."""
    rgb = torch.ones(2, 3, 4, 4) * 0.485
    out = _normalize_rgb(rgb, dim=1)
    torch.testing.assert_close(out[:, 0], torch.zeros(2, 4, 4), atol=1e-6, rtol=1e-4)
    assert torch.isfinite(out).all()


def test_normalize_rgb_preserves_video_shape() -> None:
    """Video tensor (B,C,T,H,W) converts with shape preserved."""
    video = torch.rand(2, 3, 8, 16, 16)
    assert _normalize_rgb(video, dim=1).shape == video.shape


def test_fixture_dataset_yields_masked_x_and_clean_y() -> None:
    """FixtureClipDataset returns x (has mask), y (clean), and mask tensor."""
    dataset = FixtureClipDataset(
        num_frames=8,
        height=32,
        width=32,
        mask_ratio=0.25,
        repeats=2,
    )
    x, y, mask = next(iter(dataset))
    assert x.shape == (3, 8, 32, 32)
    assert y.shape == (3, 8, 32, 32)
    assert mask.shape == (8, 32, 32)
    assert mask.dtype == torch.bool
    assert mask.sum() > 0  # some tokens masked
    assert not torch.equal(x, y)  # masked view differs from clean target


def test_fixture_dataset_normalized_output() -> None:
    """x and y are standardized (zero-center-ish, finite)."""
    dataset = FixtureClipDataset(
        num_frames=4,
        height=16,
        width=16,
        mask_ratio=0.3,
        repeats=1,
    )
    x, y, mask = next(iter(dataset))
    assert x.shape[0] == 3
    assert torch.isfinite(x).all()
    assert abs(y.float().mean().item()) < 2.0


def test_fixture_dataset_is_finite_and_real_pixels() -> None:
    """Smoke data comes from real pixels: y has nonzero variance."""
    dataset = FixtureClipDataset(num_frames=4, height=16, width=16, repeats=1)
    _, y, _ = next(iter(dataset))
    assert y.shape == (3, 4, 16, 16)
    assert y.std() > 1e-3  # not noise-flat, not constant


def test_fixture_video_clips_are_unit_scale() -> None:
    """Video smoke clips are ImageNet-standardized like the real pipeline."""
    from convgamer.data.datamemodule import FixtureVideoDataset

    ds = FixtureVideoDataset(num_frames=8, size=16, num_classes=4)
    x, y = ds[0]
    assert x.shape == (3, 8, 16, 16)
    assert torch.isfinite(x).all()
    assert abs(x.float().mean().item()) < 2.0  # standardized, not [0, 1]
    assert y.item() == 0  # labels cycle from 0


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


def test_resolve_video_paths_extracts_tier2_archives(tmp_path, monkeypatch) -> None:
    """Tier 2 .tar.gz/.tgz shards extract once and feed the decord pipeline."""
    import tarfile

    inner = tmp_path / "clip.mp4"
    inner.write_bytes(b"placeholder")
    specs: list[tuple[str, str]] = [("a.tar.gz", "w:gz"), ("b.tgz", "w:gz"), ("c.tar", "w")]
    for name, mode in specs:
        with tarfile.open(tmp_path / name, mode=mode) as tf:  # type: ignore[call-overload]
            tf.add(inner, arcname="clip.mp4")
    monkeypatch.chdir(tmp_path)
    resolved = _resolve_video_paths(["*.tar.gz", "*.tgz", "*.tar"])
    assert len(resolved) == 3
    assert all(path.endswith("clip.mp4") for path in resolved)
    assert len({path.rsplit(".extracted", 1)[0] for path in resolved}) == 3


def test_game_image_dataset_converts_any_mode_to_rgb(tmp_path) -> None:
    """Grayscale/RGBA PNGs all yield (3, 1, H, W) triplets via convert('RGB')."""
    from PIL import Image

    from convgamer.data.dataset import GameImageDataset

    Image.new("L", (16, 16), color=128).save(tmp_path / "gray.png")
    Image.new("RGBA", (16, 16), color=(255, 0, 0, 128)).save(tmp_path / "alpha.png")
    ds = GameImageDataset(image_paths=[str(tmp_path / "*.png")], size=16, mask_ratio=0.0)
    batches = list(ds)
    assert len(batches) == 2
    for x, y, mask in batches:
        assert x.shape == y.shape == (3, 1, 16, 16)
        assert mask.shape == (1, 16, 16)
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
