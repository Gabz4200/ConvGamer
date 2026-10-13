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
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import convgamer.data.sources as srcmod
from convgamer.data import dataset as dsmod
from convgamer.data.dataset import (
    FixtureClipDataset,
    GameImageDataset,
    GameVideoDataset,
)
from convgamer.data.jepa_datamodule import VJEPAGamingDataModule
from convgamer.data.sources import (
    _expand_archive_globs,
    _looks_like_hf_id,
    _resolve_video_paths,
    cleanup_chunk,
    iter_video_chunks,
    stride_cap,
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


def test_video_mode_trains_on_all_ids_with_tail_slices() -> None:
    """Train sees every id (tails sliced in-dataset); val takes tails (no downloads)."""
    mod = VJEPAGamingDataModule(video_dataset_ids=["a", "b", "c"], mode="video", streaming=True)
    mod.setup("fit")
    assert mod.train_ds is not None
    assert isinstance(mod.train_ds, GameVideoDataset)
    assert mod.train_ds.video_paths == ["a", "b", "c"]
    assert mod.train_ds.file_slice == "train"
    val_loader = mod.val_dataloader()
    assert val_loader is not None
    val_ds = val_loader.dataset
    assert isinstance(val_ds, GameVideoDataset)
    assert val_ds.video_paths == ["a", "b", "c"]
    assert val_ds.file_slice == "val"


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


def test_iter_video_chunks_slices_local_globs(tmp_path, monkeypatch) -> None:
    """Local globs chunk without any Hub contact (bounded working set)."""
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        (tmp_path / name).write_bytes(b"placeholder")
    monkeypatch.chdir(tmp_path)
    chunks = list(iter_video_chunks(["*.mp4"], data_dir=str(tmp_path), chunk_size=2))
    assert chunks == [["a.mp4", "b.mp4"], ["c.mp4"]]


def test_cleanup_chunk_removes_only_repo_dirs(tmp_path) -> None:
    """cleanup_chunk frees per-repo trees; outside files are never deleted."""
    repo_file = tmp_path / "owner__repo" / "videos" / "clip.mp4"
    repo_file.parent.mkdir(parents=True)
    repo_file.write_bytes(b"placeholder")
    (tmp_path / "owner__repo" / ".hf_cache").mkdir(exist_ok=True)
    outside = tmp_path / "keep.mp4"
    outside.write_bytes(b"placeholder")
    cleanup_chunk([str(repo_file), str(outside)], data_dir=str(tmp_path))
    assert not (tmp_path / "owner__repo").exists()
    assert outside.is_file()


def test_resolve_video_paths_caps_remote_files(tmp_path, monkeypatch) -> None:
    """max_files_per_repo fetches N single files, never a full snapshot."""
    seen: list[str] = []

    def fake_download(
        repo_id: str, filenames: list[str], data_dir: str = "", **kwargs
    ) -> list[str]:
        seen.extend(filenames)
        return [f"{data_dir}/{repo_id.replace('/', '__')}/{n}" for n in filenames]

    monkeypatch.setattr(
        srcmod, "list_remote_video_files", lambda repo_id, **kwargs: ["a.mp4", "b.mp4"]
    )
    monkeypatch.setattr(srcmod, "download_video_chunk", fake_download)
    resolved = _resolve_video_paths(["owner/repo"], data_dir=str(tmp_path), max_files_per_repo=1)
    assert seen == ["a.mp4"]
    assert resolved == [f"{tmp_path}/owner__repo/a.mp4"]


def test_game_video_dataset_streams_one_file_at_a_time(monkeypatch, tmp_path) -> None:
    """Streaming mode lists (no bytes), downloads just-in-time, cleans up."""
    frames = torch.rand(4, 3, 16, 16)
    downloaded: list[str] = []

    def fake_download(repo_id: str, filename: str, data_dir: str = "", **kwargs) -> str:
        path = str(tmp_path / filename)
        Path(path).write_bytes(b"placeholder")
        downloaded.append(path)
        return path

    monkeypatch.setattr(
        dsmod, "list_remote_video_files", lambda repo_id, **kwargs: [f"{c}.mp4" for c in "abcd"]
    )
    monkeypatch.setattr(dsmod, "download_remote_file", fake_download)
    monkeypatch.setattr(GameVideoDataset, "_load_video", lambda self, path: frames)
    ds = GameVideoDataset(
        video_paths=["owner/repo"],
        num_frames=4,
        height=16,
        width=16,
        data_dir=str(tmp_path),
        streaming=True,
        cleanup_downloads=True,
    )
    batches = list(ds)
    # Default val_tail=2 holds out c/d; train yields a/b.
    assert len(batches) == 2
    for x, y, mask in batches:
        assert x.shape == y.shape == (3, 4, 16, 16)
        assert mask.shape == (4, 16, 16) and mask.dtype == torch.bool
    # Each just-in-time file is deleted after its sample (flat disk).
    assert downloaded and all(not os.path.exists(p) for p in downloaded)


def test_game_video_masks_vary_across_passes(monkeypatch) -> None:
    """Repeated passes over one file yield fresh masks (no frozen replay)."""
    frames = torch.rand(4, 3, 16, 16)  # (T, C, H, W) in [0, 1]
    monkeypatch.setattr(GameVideoDataset, "_load_video", lambda self, path: frames)
    ds = GameVideoDataset(video_paths=["a.mp4"], num_frames=4, height=16, width=16)
    monkeypatch.setattr(dsmod, "_resolve_video_paths", lambda *a, **k: ["a.mp4"])
    masks = [mask for _, _, mask in ds]
    masks += [mask for _, _, mask in ds]  # second pass over the same file
    assert any(not torch.equal(masks[0], m) for m in masks[1:])


def test_streaming_val_takes_tails_across_all_repos() -> None:
    """Streaming val slices tails from every repo (multi-domain, cached)."""
    mod = VJEPAGamingDataModule(
        video_dataset_ids=["a", "b"], mode="video", streaming=True, cleanup_downloads=True
    )
    mod.setup("fit")
    val_loader = mod.val_dataloader()
    assert val_loader is not None
    val_ds = val_loader.dataset
    assert isinstance(val_ds, GameVideoDataset)
    assert val_ds.streaming is True
    assert val_ds.video_paths == ["a", "b"]
    assert val_ds.file_slice == "val"
    assert val_ds.val_tail == 2
    assert val_ds.cleanup_downloads is False


def test_stride_cap_covers_repo_evenly() -> None:
    """Strided caps span the listing instead of clustering at the top."""
    files = [f"{i:03d}.mp4" for i in range(310)]
    capped = stride_cap(files, 60)
    assert len(capped) == 60
    assert capped[0] == "000.mp4"
    assert capped[-1] < "310.mp4" and int(capped[-1][:3]) > 250
    assert stride_cap(files, 1000) == files
    with pytest.raises(ValueError, match="cap must be"):
        stride_cap(files, 0)


def test_streaming_oversample_and_interleave(monkeypatch, tmp_path) -> None:
    """Caps shrink huge repos; repeats expand small important ones; batches mix."""
    listings = {
        "big/repo": [f"b{i}.mp4" for i in range(10)],
        "small/repo": [f"s{i}.mp4" for i in range(3)],
    }
    seen: list[str] = []

    def fake_download(repo_id: str, filename: str, data_dir: str = "", **kwargs) -> str:
        seen.append(f"{repo_id}/{filename}")
        return f"{repo_id}/{filename}"

    monkeypatch.setattr(
        dsmod, "list_remote_video_files", lambda repo_id, **kwargs: listings[repo_id]
    )
    monkeypatch.setattr(dsmod, "download_remote_file", fake_download)
    ds = GameVideoDataset(
        video_paths=["big/repo", "small/repo"],
        num_frames=4,
        height=16,
        width=16,
        data_dir=str(tmp_path),
        streaming=True,
        subset_caps={"big/repo": 4},
        oversample={"small/repo": 2},
        val_tail=1,
    )
    refs = list(ds._iter_streaming_refs())
    # big: 10-1 tail=9 pooled -> stride to 4; small: (3-1) x2 = 4; total 8.
    assert len(refs) == 8
    assert seen.count("small/repo/s0.mp4") == 2
    assert sum(1 for r in seen if r.startswith("big/repo")) == 4
    # Round-robin alternates repos instead of running homogeneous stretches.
    repos = [r.split("/")[0] for r in seen]
    assert repos[0] != repos[1]


def test_train_val_tails_are_disjoint(monkeypatch, tmp_path) -> None:
    """Train pools and val tails never share a file within one repo."""
    files = [f"f{i}.mp4" for i in range(6)]
    monkeypatch.setattr(dsmod, "list_remote_video_files", lambda repo_id, **kwargs: files)
    monkeypatch.setattr(
        dsmod, "download_remote_file", lambda repo_id, fn, data_dir="", **kwargs: fn
    )
    kwargs: dict = {
        "video_paths": ["owner/repo"],
        "num_frames": 4,
        "height": 16,
        "width": 16,
        "data_dir": str(tmp_path),
        "streaming": True,
        "val_tail": 2,
    }
    train_refs = set(GameVideoDataset(**kwargs)._iter_streaming_refs())
    val_refs = set(GameVideoDataset(**kwargs, file_slice="val")._iter_streaming_refs())
    assert val_refs == {"f4.mp4", "f5.mp4"}
    assert train_refs == {"f0.mp4", "f1.mp4", "f2.mp4", "f3.mp4"}
    assert not train_refs & val_refs


def test_describe_sampling_reports_effective_counts(monkeypatch) -> None:
    """Sampling table math: pool excludes tail, cap binds, repeat multiplies."""
    monkeypatch.setattr(
        srcmod,
        "list_remote_video_files",
        lambda repo_id, **kwargs: [f"f{i}.mp4" for i in range(10)],
    )
    rows = srcmod.describe_sampling(
        ["owner/repo"], caps={"owner/repo": 4}, repeats={"owner/repo": 3}, val_tail=2
    )
    assert rows == [("owner/repo", 10, 4, 3, 12)]


def test_fixture_clip_dataset_shards_repeats_across_workers(monkeypatch) -> None:
    """Each worker yields a disjoint shard of repeats (no N× duplication)."""
    monkeypatch.setattr(dsmod, "get_worker_info", lambda: SimpleNamespace(id=1, num_workers=2))
    ds = FixtureClipDataset(num_frames=4, height=16, width=16, repeats=4)
    assert len(ds) == 2
    masks = [m for _, _, m in ds]
    assert len(masks) == 2


def test_nonstreaming_video_mode_holds_out_last_id() -> None:
    """Non-streaming train excludes the val id (no train/val leakage)."""
    mod = VJEPAGamingDataModule(video_dataset_ids=["a", "b", "c"], mode="video", streaming=False)
    mod.setup("fit")
    assert mod.train_ds is not None
    assert isinstance(mod.train_ds, GameVideoDataset)
    assert mod.train_ds.video_paths == ["a", "b"]


def test_stable_seed_is_hashseed_independent() -> None:
    """Per-path salt is identical regardless of PYTHONHASHSEED."""
    from convgamer.data.dataset import _stable_seed

    assert _stable_seed("owner/repo/clip.mp4") == _stable_seed("owner/repo/clip.mp4")
    assert _stable_seed("a.mp4") != _stable_seed("b.mp4")


def test_extract_archive_rejects_symlink(tmp_path) -> None:
    """Symlink members are refused even when they resolve inside the root."""
    target = tmp_path / "outside.mp4"
    target.write_bytes(b"placeholder")
    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo("link.mp4")
        info.type = tarfile.SYMTYPE
        info.linkname = str(target)
        tf.addfile(info)
    with pytest.raises(ValueError, match="link"):
        _expand_archive_globs(str(archive))
