"""V-JEPA 2.1 style components for ConvGamer.

Public seams:

- ``VJEPAPredictor``: dense mask-token predictor following §2.3.1.
- ``JEPALoss``: L_predict + L_ctx distance-weighted loss.
- ``EMAEncoder``: shadow-copy EMA wrapper for the target encoder.
"""

from convgamer.models.jepa.ema import EMAEncoder
from convgamer.models.jepa.loss import JEPALoss, compute_context_lambdas
from convgamer.models.jepa.predictor import VJEPAPredictor

__all__ = [
    "EMAEncoder",
    "JEPALoss",
    "VJEPAPredictor",
    "compute_context_lambdas",
]
