"""Model package — re-exports from the InceptionNeXt sub-package."""

from .inception_next.blocks import InceptionDWConv2d, InceptionNeXtBlock  # noqa: F401 — re-export
from .inception_next.encoder import Encoder  # noqa: F401 — triggers @register_model
from .registry import get_model, register_model  # noqa: F401

__all__ = [
    "Encoder",
    "InceptionDWConv2d",
    "InceptionNeXtBlock",
    "register_model",
    "get_model",
]
