"""Lightning modules for InceptionNeXt and ConvGamer — training wrappers.

Backbones live in models/; this package only adds steps/optimizers/EMA.
EMA *bookkeeping* (parameter state) lives in ``modules/ema.py``; the EMA
*lifecycle hook* lives in ``callbacks/ema_update.py``.
"""

from .ema import EMAEncoder, EncoderT
from .jepa_module import ConvGamerVJEPAModel
from .lightning_module import (
    ClassificationLightningModule,
    ConvGamerModel,
    InceptionNeXtModule,
)

__all__ = [
    "ClassificationLightningModule",
    "ConvGamerModel",
    "ConvGamerVJEPAModel",
    "EMAEncoder",
    "EncoderT",
    "InceptionNeXtModule",
]
