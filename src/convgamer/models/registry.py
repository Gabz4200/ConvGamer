from __future__ import annotations

from collections.abc import Callable
from typing import Any

MODEL_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_model(name: str):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {list(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[name]
    return cls(**kwargs)
