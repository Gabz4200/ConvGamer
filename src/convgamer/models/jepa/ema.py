"""EMA shadow copy for the JEPA target encoder.

Follows V-JEPA 2.1 Appendix A: EMA coefficient = 0.99925.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

__all__ = ["EMAEncoder"]


class EMAEncoder:
    """Maintains a shadow copy of an encoder updated via EMA.

    Usage::

        ema = EMAEncoder(encoder, decay=0.99925)
        target = ema.encode(x)          # forward on shadow copy
        ...  # train online encoder
        ema.update()                    # sync shadow weights
    """

    def __init__(self, module: nn.Module, decay: float = 0.99925):
        self.module = module
        self.decay = decay
        self.shadow_module = copy.deepcopy(module).eval()
        self.shadow_params = list(self.shadow_module.parameters())

    @torch.no_grad()
    def encode(self, *args, **kwargs) -> torch.Tensor:
        return self.shadow_module(*args, **kwargs)

    @torch.no_grad()
    def update(self) -> None:
        ema_d = self.decay
        for s_param, param in zip(self.shadow_params, self.module.parameters(), strict=True):
            if param.requires_grad:
                s_param.mul_(ema_d).add_(param.data, alpha=1 - ema_d)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow_module.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.shadow_module.load_state_dict(state_dict)
