from __future__ import annotations

from collections.abc import Callable
from typing import Any

MODEL_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_model(name: str):
    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered")
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {list(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[name]
    return cls(**kwargs)


def unregister_model(name: str) -> None:
    """Remove ``name`` from the registry (test teardown helper)."""
    MODEL_REGISTRY.pop(name, None)


def clear_registry() -> None:
    """Remove all registered models.

    Test-isolation helper only — never call in production code; use
    :func:`unregister_model` for surgical teardown.
    """
    MODEL_REGISTRY.clear()
