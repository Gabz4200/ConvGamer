"""Model package — raw nn.Module backbones (no training logic).

Training loops live in modules/ (Lightning wrappers around these backbones).
Re-exports backbones (triggers registry as a side effect)."""

from .convgamer.encoder import ConvGamerEncoder  # noqa: F401 — triggers @register_model
from .inception_next.blocks import InceptionDWConv2d, InceptionNeXtBlock  # noqa: F401 — re-export
from .inception_next.encoder import InceptionNeXtEncoder  # noqa: F401 — triggers @register_model
from .io import StepOutput, StreamingState  # noqa: F401 — re-export
from .protocols import DenseFeatureEncoder  # noqa: F401 — re-export
from .registry import get_model, register_model  # noqa: F401

__all__ = [
    "ConvGamerEncoder",
    "InceptionNeXtEncoder",
    "InceptionDWConv2d",
    "InceptionNeXtBlock",
    "DenseFeatureEncoder",
    "StepOutput",
    "StreamingState",
    "register_model",
    "get_model",
]
