"""Re-export shim — split of the former 835-line blocks.py.

Paper mapping (unchanged behavior):
- ``causal``: custom streaming primitives (CausalConv3d, norms, mixer).
- ``downsampler``: custom learned downsampler + temporal subsample helper.
- ``stem``: custom multi-scale ConvGamerStem.
- ``minconv``: MinConvLSTM/MinConvExpLSTM per arXiv:2508.03614v1 §3.1-3.2
  (input-only gates, normalized f/i, parallel prefix scan).
"""

from __future__ import annotations

from .causal import CausalConv3d, CausalLayerNorm, CausalTemporalMixer
from .downsampler import (
    LearnedSpatialTemporalDownsampler,
    spatial_softmax,
    uniform_temporal_subsample,
)
from .minconv import (
    MinConvExpLSTM,
    MinConvLSTM,
    _linear_prefix_scan,
    _normalized_sigmoid_gates,
)
from .stem import ConvGamerStem

__all__ = [
    "CausalConv3d",
    "CausalLayerNorm",
    "CausalTemporalMixer",
    "LearnedSpatialTemporalDownsampler",
    "ConvGamerStem",
    "MinConvLSTM",
    "MinConvExpLSTM",
    "spatial_softmax",
    "uniform_temporal_subsample",
    "_linear_prefix_scan",
    "_normalized_sigmoid_gates",
]
