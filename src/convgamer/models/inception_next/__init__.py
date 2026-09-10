"""InceptionNeXt model implementation (arXiv:2303.16900).

Self-contained: imports nothing ConvGamer-specific.  ConvGamer composes
this package from above — never the reverse.
"""

from .blocks import InceptionDWConv2d, InceptionNeXtBlock
from .encoder import Encoder

__all__ = [
    "Encoder",
    "InceptionDWConv2d",
    "InceptionNeXtBlock",
]
