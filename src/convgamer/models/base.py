from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseModel(nn.Module, ABC):
    """Base contract for all models in the registry.

    Subclasses must implement ``forward``.  ``__init__`` receives its
    configuration as keyword arguments so the registry and the
    Transformers wrapper can both instantiate from a plain dict.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
