from __future__ import annotations

import glob
import os
import tarfile
import zipfile


def _looks_like_hf_id(item: str) -> bool:
    """True for ``owner/repo`` ids; false for local files, globs, and abs paths."""
    if os.path.exists(item) or os.path.isabs(item):
        return False
    if "*" in item or "?" in item or "[" in item:
        return False
    if item.endswith((".mp4", ".jpg", ".jpeg", ".png")):
        return False
    if item.startswith((".", "/", "~", "data/")) or "\\" in item:
        # Local relative/absolute paths and the repo's own data/ tree are
        # never HF ids (e.g. "data/videos" must glob locally, not download).
        return False
    parts = [p for p in item.split("/") if p not in ("", ".", "..")]
    return len(parts) == 2


def _shard_for_distributed(paths: list[str]) -> list[str]:
    """Shard ``paths`` across DDP ranks (no-op outside distributed)."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            rank, world = dist.get_rank(), dist.get_world_size()
            return paths[rank::world]
    except Exception:
        pass
    return paths


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
        if _looks_like_hf_id(item) and not os.path.exists(item):
            local_dir = snapshot_download(
                repo_id=item,
                repo_type="dataset",
                local_dir=os.path.join(data_dir, item.replace("/", "__")),
                allow_patterns=["**/*.mp4", "*.mp4", "**/*.tar", "*.tar", "**/*.zip", "*.zip"],
            )
            paths.extend(_expand_archive_globs(local_dir, recursive=True))
        else:
            paths.extend(_expand_archive_globs(item, recursive=True))
    paths = _shard_for_distributed(paths)
    if num_workers > 1:
        paths = paths[worker_id::num_workers]
    return paths


def _expand_archive_globs(pattern: str, recursive: bool = True) -> list[str]:
    """Glob ``pattern`` and flatten any ``.tar``/``.zip`` archives into MP4 paths.

    Archives are extracted once to a sibling cache directory; subsequent
    globs reuse the extracted tree.  This lets Tier 2 datasets (Pixel2Play,
    UCF101 shards, Kinetics tarballs) feed the same ``decord`` pipeline as
    Tier 1 direct MP4s.
    """
    matches = sorted(glob.glob(pattern, recursive=recursive))
    out: list[str] = []
    for match in matches:
        if os.path.isdir(match):
            out.extend(sorted(glob.glob(os.path.join(match, "**", "*.mp4"), recursive=True)))
            continue
        ext = os.path.splitext(match)[1].lower()
        # Tier 2 archives arrive as .tar, .tar.gz, .tgz, or .zip.
        name = os.path.basename(match).lower()
        if ext in {".tar", ".zip", ".tgz"} or (ext == ".gz" and ".tar" in name):
            out.extend(_extract_archive(match))
        elif ext == ".mp4":
            out.append(match)
    return out


def _extract_archive(archive_path: str) -> list[str]:
    """Extract ``.tar``/``.zip`` to a sibling cache dir and return MP4 paths."""
    from pathlib import Path

    extract_dir = archive_path + ".extracted"
    os.makedirs(extract_dir, exist_ok=True)
    root = Path(extract_dir).resolve()
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            for member in zf.namelist():
                dest = (root / member).resolve()
                if not dest.is_relative_to(root):
                    raise ValueError(f"unsafe zip member: {member!r}")
            zf.extractall(extract_dir)
    else:
        # tarfile auto-detects compression (.tar, .tar.gz, .tgz).
        with tarfile.open(archive_path, mode="r:*") as tf:
            for member in tf.getmembers():
                dest = (root / member.name).resolve()
                if not dest.is_relative_to(root):
                    raise ValueError(f"unsafe tar member: {member.name!r}")
            try:
                tf.extractall(extract_dir, filter="data")
            except TypeError:
                # Older Python without filter=: members already validated above.
                tf.extractall(extract_dir)
    return sorted(glob.glob(os.path.join(extract_dir, "**", "*.mp4"), recursive=True))


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
        if _looks_like_hf_id(item) and not os.path.exists(item):
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
    paths = _shard_for_distributed(paths)
    if num_workers > 1:
        paths = paths[worker_id::num_workers]
    return paths
