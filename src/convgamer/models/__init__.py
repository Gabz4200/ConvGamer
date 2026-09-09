"""Model package — re-exports from the InceptionNeXt sub-package."""

from .inception_next.blocks import (  # noqa: F401 — re-export
    AttentionBlock,
    FeedForwardBlock,
    InceptionDWConv2d,
    InceptionNeXtBlock,
)
from .inception_next.encoder import Encoder  # noqa: F401 — triggers @register_model
from .registry import get_model, register_model  # noqa: F401

__all__ = [
    "Encoder",
    "InceptionDWConv2d",
    "InceptionNeXtBlock",
    "AttentionBlock",
    "FeedForwardBlock",
    "register_model",
    "get_model",
]
