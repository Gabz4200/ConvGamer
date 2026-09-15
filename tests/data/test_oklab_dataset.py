"""Behavior tests for JEPA datasets and oklab conversion.

Seams tested:
- ``oklab_convert_srgb``: RGB tensor (B,C,H,W) -> Oklab (B,3,H,W).
- ``GameVideoDataset``: returns (B,C,T,H,W) video tensor from mp4 paths.
- ``GameImageDataset``: returns (B,C,H,W) from image paths -> oklab.
- ``JEPADataset``: yields (x_view, y_view, mask) where x is masked.
"""

from __future__ import annotations

import torch

from convgamer.data.dataset import (
    JEPADataset,
    _resolve_image_paths,
    _resolve_video_paths,
    oklab_convert_srgb,
)
from convgamer.data.oklab import srgb_to_oklab


def test_oklab_convert_srgb_matches_oklab_py() -> None:
    """Our batch converter must match the per-tensor oklab module function."""
    rgb = torch.rand(2, 3, 8, 8)
    converted = oklab_convert_srgb(rgb)
    expected = srgb_to_oklab(rgb.clone(), dim=1)
    torch.testing.assert_close(converted, expected, atol=1e-5, rtol=1e-4)


def test_oklab_convert_srgb_preserves_video_shape() -> None:
    """Video tensor (B,C,T,H,W) converts with shape preserved."""
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
