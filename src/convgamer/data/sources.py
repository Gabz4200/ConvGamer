from __future__ import annotations

import glob
import logging
import os
import shutil
import tarfile
import zipfile
from collections.abc import Iterator

logger = logging.getLogger(__name__)

_VIDEO_EXTS = (".mp4",)
_ARCHIVE_PATTERNS = ["**/*.mp4", "*.mp4", "**/*.tar", "*.tar", "**/*.zip", "*.zip"]


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


def disk_free_bytes(path: str = ".") -> int:
    """Free bytes on the filesystem holding ``path`` (for pre-download guards)."""
    return shutil.disk_usage(path).free


def list_remote_video_files(
    repo_id: str,
    repo_type: str = "dataset",
) -> list[str]:
    """List remote ``.mp4`` files in a Hub dataset without downloading anything.

    Uses ``HfApi.list_repo_files`` (metadata only, bytes not kilobytes), so
    counting files across the 5 gameplay repos costs no disk. Sorted for
    deterministic chunking.
    """
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id=repo_id, repo_type=repo_type)
    return sorted(f for f in files if f.lower().endswith(_VIDEO_EXTS))


def download_remote_file(
    repo_id: str,
    filename: str,
    data_dir: str = "./data/jepa",
    repo_type: str = "dataset",
) -> str:
    """Download one Hub file into an isolated per-repo dir; return its path.

    Unlike ``snapshot_download`` (whole repo + cache duplication), this pulls
    a single blob via ``hf_hub_download`` with both the file copy and the
    Hub cache scoped under ``data_dir``. Deleting the returned file's
    top-level repo dir frees everything that download consumed.
    """
    from huggingface_hub import hf_hub_download

    local_dir = os.path.join(data_dir, repo_id.replace("/", "__"))
    cache_dir = os.path.join(local_dir, ".hf_cache")
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        local_dir=local_dir,
        cache_dir=cache_dir,
    )


def download_video_chunk(
    repo_id: str,
    filenames: list[str],
    data_dir: str = "./data/jepa",
    min_free_gb: float = 1.0,
) -> list[str]:
    """Download exactly ``filenames`` from ``repo_id``; skip when disk is low.

    Files already on disk are reused. Raises ``OSError`` (errno 28) when
    free space drops below ``min_free_gb`` so the caller can train on what
    it has, clean up, and continue instead of crashing mid-reconstruction.
    """
    paths: list[str] = []
    for name in filenames:
        expected = os.path.join(data_dir, repo_id.replace("/", "__"), name)
        if os.path.isfile(expected):
            paths.append(expected)
            continue
        free_gb = disk_free_bytes(data_dir) / 1e9
        if free_gb < min_free_gb:
            raise OSError(
                f"[Errno 28] No space left: {free_gb:.2f} GB free in {data_dir!r}, "
                f"need >={min_free_gb:.1f} GB to fetch {repo_id}/{name}; "
                "train on the current chunk, delete it, then continue."
            )
        paths.append(download_remote_file(repo_id, name, data_dir=data_dir))
    return paths


def iter_video_chunks(
    video_ids: list[str],
    data_dir: str = "./data/jepa",
    chunk_size: int = 30,
) -> Iterator[list[str]]:
    """Yield successive local-path chunks, downloading one chunk at a time.

    Remote HF ids are listed (no bytes) and fetched ``chunk_size`` files at
    a time; local globs are expanded directly and sliced the same way. The
    caller trains on the yielded paths, then calls :func:`cleanup_chunk` to
    free the bytes before the next chunk is fetched — disk stays O(chunk),
    not O(dataset). Chunk ``local_dir`` trees double as the cleanup unit.
    """
    for item in video_ids:
        if _looks_like_hf_id(item) and not os.path.exists(item):
            remote = list_remote_video_files(item)
            for start in range(0, len(remote), chunk_size):
                yield download_video_chunk(item, remote[start : start + chunk_size], data_dir)
        else:
            local = _expand_archive_globs(item, recursive=True)
            for start in range(0, len(local), chunk_size):
                yield local[start : start + chunk_size]


def cleanup_chunk(paths: list[str], data_dir: str = "./data/jepa") -> None:
    """Delete the per-repo ``local_dir`` trees backing ``paths``.

    Removes both the file copies and the scoped ``.hf_cache`` written by
    :func:`download_remote_file`. Local-glob paths outside ``data_dir`` are
    left untouched (never delete user data).
    """
    base = os.path.abspath(data_dir)
    seen: set[str] = set()
    for path in paths:
        abs_path = os.path.abspath(path)
        if abs_path != base and not abs_path.startswith(base + os.sep):
            continue
        rel = os.path.relpath(abs_path, base)
        top = rel.split(os.sep)[0]
        if top in ("", ".", "..", ".hf_cache"):
            continue
        repo_root = os.path.join(base, top)
        if repo_root not in seen and os.path.isdir(repo_root):
            seen.add(repo_root)
            shutil.rmtree(repo_root, ignore_errors=True)


def _resolve_video_paths(
    video_ids: list[str],
    data_dir: str = "./data/jepa",
    worker_id: int = 0,
    num_workers: int = 1,
    max_files_per_repo: int | None = None,
) -> list[str]:
    """Expand HF dataset IDs or local globs into a flat MP4 list.

    HF IDs download via ``huggingface_hub.snapshot_download`` with an MP4
    allow-pattern; local globs expand directly. Pass ``max_files_per_repo``
    to fetch only the first N remote ``.mp4`` per repo (single-file
    downloads, O(chunk) disk) instead of the whole snapshot — the Kaggle
    path for 20 GB working disks.
    """
    from huggingface_hub import snapshot_download

    paths: list[str] = []
    for item in video_ids:
        if _looks_like_hf_id(item) and not os.path.exists(item):
            if max_files_per_repo is not None:
                remote = list_remote_video_files(item)[:max_files_per_repo]
                paths.extend(download_video_chunk(item, remote, data_dir))
                continue
            free_gb = disk_free_bytes(data_dir) / 1e9
            if free_gb < 1.0:
                raise OSError(
                    f"[Errno 28] No space left: {free_gb:.2f} GB free in {data_dir!r}; "
                    "refusing full snapshot_download. Pass max_files_per_repo=N "
                    "(or use iter_video_chunks + cleanup_chunk) to train in "
                    "download-train-delete chunks."
                )
            local_dir = snapshot_download(
                repo_id=item,
                repo_type="dataset",
                local_dir=os.path.join(data_dir, item.replace("/", "__")),
                allow_patterns=_ARCHIVE_PATTERNS,
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
