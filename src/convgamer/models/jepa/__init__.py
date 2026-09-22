"""V-JEPA 2.1 style components for ConvGamer.

Public seams:

- ``VJEPAPredictor``: dense mask-token predictor following §2.3.1.
- ``JEPALoss``: L_predict + L_ctx distance-weighted loss.
"""

from convgamer.models.jepa.loss import JEPALoss, compute_context_lambdas
from convgamer.models.jepa.predictor import VJEPAPredictor

__all__ = [
    "JEPALoss",
    "VJEPAPredictor",
    "compute_context_lambdas",
]
