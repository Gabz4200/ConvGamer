from __future__ import annotations

import os
import threading

import taichi as ti

_initialized = False
_initialized_arch: str | None = None
_lock = threading.Lock()

_ARCH_MAP = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan, "metal": ti.metal}


def ensure_initialized(arch: str | None = None) -> None:
    """Initialize Taichi exactly once per process.

    Reads ``CONVGAMER_TAICHI_ARCH`` env var (defaults to ``cpu``).
    Subsequent calls with a different arch are ignored (Taichi allows one init).
    """
    global _initialized, _initialized_arch
    if _initialized:
        requested = arch or os.getenv("CONVGAMER_TAICHI_ARCH", "cpu")
        if requested != _initialized_arch:
            import warnings

            warnings.warn(
                f"Taichi already initialized with arch '{_initialized_arch}'; "
                f"ignoring re-init with '{requested}'",
                UserWarning,
                stacklevel=2,
            )
        return
    with _lock:
        if _initialized:
            return
        name = arch or os.getenv("CONVGAMER_TAICHI_ARCH", "cpu")
        if name not in _ARCH_MAP:
            raise ValueError(f"Unknown Taichi arch '{name}'. Valid: {list(_ARCH_MAP)}")
        ti.init(arch=_ARCH_MAP[name])
        _initialized = True
        _initialized_arch = name
