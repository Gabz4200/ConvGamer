"""Structural interfaces (PEP 544) for swappable backbone components.

``nn.Module`` subclasses satisfy these *structurally*: no nominal inheritance,
no runtime ``getattr``/``callable`` probing.  Type checkers enforce the seam
statically, so a backbone missing a required method fails at check time rather
than mid-training on a remote worker.

The protocols are deliberately narrow — only the surface the consumer actually
uses — so third-party modules conform without inheriting from this project.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Protocol

import torch
from torch import nn

__all__ = ["DenseFeatureEncoder"]


class DenseFeatureEncoder(Protocol):
    """A video backbone usable as the JEPA context/target encoder.

    Required by ``ConvGamerVJEPAModel`` and ``EMAEncoder``:

    - ``forward_feature_maps`` produces the spatial feature maps that the
      predictor predicts and the target encoder supplies as ground truth.
    - The ``nn.Module`` bookkeeping surface below is what EMA tracking needs
      (enumerate parameters, snapshot/restore the shadow copy, move it to the
      online encoder's device/dtype).  Signatures mirror ``nn.Module`` so any
      ordinary ``nn.Module`` backbone conforms without adapter code.
    """

    def forward_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, T, H, W)`` video to maps ``(B, F, T', H', W')``."""
        ...

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...

    def state_dict(self, *args: Any, **kwargs: Any) -> Any: ...

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, **kwargs: Any
    ) -> Any: ...
    def eval(self) -> Any: ...

    def to(self, *args: Any, **kwargs: Any) -> Any: ...
