"""Lightning modules for InceptionNeXt and ConvGamer — training wrappers.

Backbones live in models/; this package only adds steps/optimizers/EMA."""

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
    "InceptionNeXtModule",
]
