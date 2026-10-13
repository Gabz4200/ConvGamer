"""Reachability + sampling tests for the JEPA dataset layer.

Hermetic by default (no network): HF downloads are monkeypatched, archives
are built on the fly, and video decode is stubbed. A live reachability probe
against huggingface_hub runs only with ``JEPA_LIVE_HF=1``.
"""

from __future__ import annotations

import os
import tarfile
import zipfile
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from convgamer.data import dataset as dsmod
from convgamer.data.dataset import (
    FixtureClipDataset,
    GameImageDataset,
    GameVideoDataset,
)
from convgamer.data.jepa_datamodule import VJEPAGamingDataModule
from convgamer.data.sources import (
    _looks_like_hf_id,
    _resolve_video_paths,
)

_JEPA_DATA_CFG = Path("configs/data/jepa.yaml")


def test_jepa_video_ids_are_wellformed_hf_ids() -> None:
    """Every configured video source must be a reachable-shaped HF id."""
    cfg = OmegaConf.load(_JEPA_DATA_CFG)
    ids = list(cfg["video_dataset_ids"])
    assert len(ids) >= 1
    assert len(set(ids)) == len(ids)  # no duplicates
    for item in ids:
        assert _looks_like_hf_id(item), f"not an owner/repo id: {item!r}"


def test_looks_like_hf_id_rejects_local_paths() -> None:
    assert _looks_like_hf_id("owner/repo")
    assert not _looks_like_hf_id("data/videos")
    assert not _looks_like_hf_id("./rel/path")
    assert not _looks_like_hf_id("/abs/path")
    assert not _looks_like_hf_id("*.mp4")
    assert not _looks_like_hf_id("clip.mp4")
    assert not _looks_like_hf_id("a/b/c")  # not owner/repo shape
    assert not _looks_like_hf_id("photo.jpg")


def test_resolve_video_paths_downloads_hf_id(tmp_path, monkeypatch) -> None:
    """HF ids flow through snapshot_download into an MP4 list (no network)."""

    def fake_snapshot_download(*args, **kwargs):
        local_dir = kwargs["local_dir"]
        assert "__" in local_dir  # owner/repo scoped under data_dir
        mp4 = Path(local_dir) / "clip.mp4"
        mp4.parent.mkdir(parents=True, exist_ok=True)
        mp4.write_bytes(b"placeholder")
        return local_dir

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    resolved = _resolve_video_paths(["owner/repo"], data_dir=str(tmp_path))
    assert resolved == [str(tmp_path / "owner__repo" / "clip.mp4")]


def test_extract_zip_roundtrip(tmp_path, monkeypatch) -> None:
    """A .zip of MP4s feeds the same pipeline as direct MP4s."""
    archive = tmp_path / "shard.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("clip.mp4", b"placeholder")
    monkeypatch.chdir(tmp_path)
    assert _resolve_video_paths([str(tmp_path / "*.zip")]) == [str(archive) + ".extracted/clip.mp4"]


def test_extract_archive_rejects_traversal(tmp_path, monkeypatch) -> None:
    """ZipSlip / tar traversal members fail loudly instead of extracting."""
    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("../evil.mp4", b"placeholder")
    evil_tar = tmp_path / "evil.tar"
    with tarfile.open(evil_tar, "w") as tf:
        info = tarfile.TarInfo(name="../evil.mp4")
        info.size = 0
        tf.addfile(info)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="unsafe zip member"):
        _resolve_video_paths(["evil.zip"])
    with pytest.raises(ValueError, match="unsafe tar member"):
        _resolve_video_paths(["evil.tar"])


def test_game_video_dataset_samples_triplets(monkeypatch) -> None:
    """GameVideoDataset yields standardized (x, y, mask) without decord."""
    frames = torch.rand(4, 3, 16, 16)  # (T, C, H, W) in [0, 1]
    monkeypatch.setattr(dsmod, "_resolve_video_paths", lambda *a, **k: ["fake.mp4"])
    monkeypatch.setattr(GameVideoDataset, "_load_video", lambda self, path: frames)
    ds = GameVideoDataset(video_paths=["fake.mp4"], num_frames=4, height=16, width=16)
    x, y, mask = next(iter(ds))
    assert x.shape == y.shape == (3, 4, 16, 16)
    assert mask.shape == (4, 16, 16) and mask.dtype == torch.bool
    assert mask.any()
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    assert not torch.equal(x, y)  # masked view differs from clean target


def test_game_image_dataset_samples_triplet(tmp_path) -> None:
    """Tiny real PNGs yield a normalized (C, 1, H, W) triplet."""
    from PIL import Image

    Image.new("RGB", (20, 20), color=(10, 200, 30)).save(tmp_path / "a.png")
    ds = GameImageDataset(image_paths=[str(tmp_path / "*.png")], size=16, mask_ratio=0.5)
    x, y, mask = next(iter(ds))
    assert x.shape == y.shape == (3, 1, 16, 16)
    assert mask.shape == (1, 16, 16) and mask.dtype == torch.bool
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    assert abs(float(y.mean())) < 2.0  # standardized, not [0, 1]


def test_fixture_clip_masks_vary_with_base_seed() -> None:
    """Same base_seed reproduces masks; a new one varies them (no frozen masks)."""
    kwargs: dict = {"num_frames": 4, "height": 16, "width": 16, "repeats": 2}
    a = [m for _, _, m in FixtureClipDataset(base_seed=0, **kwargs)]
    b = [m for _, _, m in FixtureClipDataset(base_seed=0, **kwargs)]
    c = [m for _, _, m in FixtureClipDataset(base_seed=999, **kwargs)]
    for ma, mb in zip(a, b, strict=True):
        torch.testing.assert_close(ma, mb)
    assert any(not torch.equal(ma, mc) for ma, mc in zip(a, c, strict=True))


def test_video_mode_holds_out_last_file() -> None:
    """Train uses all-but-last id; val uses the last (no downloads at setup)."""
    mod = VJEPAGamingDataModule(video_dataset_ids=["a", "b", "c"], mode="video")
    mod.setup("fit")
    assert mod.train_ds is not None
    assert mod.train_ds.video_paths == ["a", "b"]
    val_ds = mod.val_dataloader().dataset
    assert isinstance(val_ds, GameVideoDataset)
    assert val_ds.video_paths == ["c"]


@pytest.mark.skipif(
    os.environ.get("JEPA_LIVE_HF") != "1",
    reason="live HF reachability probe; rerun with JEPA_LIVE_HF=1",
)
def test_live_hf_datasets_reachable() -> None:
    """Each configured video id resolves on the Hub (needs network)."""
    from huggingface_hub import HfApi

    api = HfApi()
    cfg = OmegaConf.load(_JEPA_DATA_CFG)
    for repo_id in cfg["video_dataset_ids"]:
        info = api.dataset_info(str(repo_id))
        assert info.id.lower() == str(repo_id).lower()
