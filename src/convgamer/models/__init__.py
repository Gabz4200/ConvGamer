"""Model package — re-exports backbones (triggers registry as a side effect)."""

from .convgamer.encoder import ConvGamerEncoder  # noqa: F401 — triggers @register_model
from .inception_next.blocks import InceptionDWConv2d, InceptionNeXtBlock  # noqa: F401 — re-export
from .inception_next.encoder import InceptionNeXtEncoder  # noqa: F401 — triggers @register_model
from .registry import get_model, register_model  # noqa: F401

__all__ = [
    "ConvGamerEncoder",
    "InceptionNeXtEncoder",
    "InceptionDWConv2d",
    "InceptionNeXtBlock",
    "register_model",
    "get_model",
]
