"""V-JEPA 2.1 style components for ConvGamer.

Public seams:

- ``VJEPAPredictor``: dense mask-token predictor following §2.3.1.
- ``JEPALoss``: L_predict + L_ctx distance-weighted loss, optional LeJEPA
  SIGReg isotropy pressure.
- ``SIGReg``: sketched Gaussianity regularizer (LeJEPA, arXiv:2511.08544).
"""

from convgamer.models.jepa.loss import JEPALoss, compute_context_lambdas
from convgamer.models.jepa.masking import resize_token_mask
from convgamer.models.jepa.predictor import VJEPAPredictor
from convgamer.models.jepa.sigreg import SIGReg

__all__ = [
    "JEPALoss",
    "SIGReg",
    "VJEPAPredictor",
    "compute_context_lambdas",
    "resize_token_mask",
]
