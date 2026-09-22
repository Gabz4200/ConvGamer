"""EMA shadow copy for the JEPA target encoder.

Lives in the shell (``modules/``), not the pure-math layer: it holds and
mutates parameter state across optimizer steps. Follows V-JEPA 2.1 Appendix A:
EMA coefficient = 0.99925.
"""

from __future__ import annotations

import copy
from typing import Any, Generic, TypeVar, cast

import torch

from convgamer.models.protocols import DenseFeatureEncoder

__all__ = ["EMAEncoder", "EncoderT"]

EncoderT = TypeVar("EncoderT", bound=DenseFeatureEncoder)


class EMAEncoder(Generic[EncoderT]):  # noqa: UP046
    """Maintains a shadow copy of an encoder updated via EMA.

    Generic over the swappable dense-feature seam: any backbone conforming to
    :class:`~convgamer.models.protocols.DenseFeatureEncoder` can be shadowed,
    and the checker — not by a runtime ``getattr`` probe — guarantees the shadow
    exposes ``forward_feature_maps``.

    Usage::

        ema = EMAEncoder(encoder, decay=0.99925)
        target = ema.shadow_module.forward_feature_maps(y)  # stop-grad target
        ...  # train online encoder
        ema.update()                    # sync shadow weights
    """

    def __init__(self, module: EncoderT, decay: float = 0.99925):
        self.module: EncoderT = module
        self.decay = decay
        self.shadow_module: EncoderT = cast(EncoderT, copy.deepcopy(module).eval())
        self.shadow_params = list(self.shadow_module.parameters())

    @torch.no_grad()
    def update(self) -> None:
        ema_d = self.decay
        for s_param, param in zip(self.shadow_params, self.module.parameters(), strict=True):
            if param.requires_grad:
                s_param.mul_(ema_d).add_(param.data, alpha=1 - ema_d)

    def state_dict(self) -> Any:
        return self.shadow_module.state_dict()

    def load_state_dict(self, state_dict: Any) -> None:
        self.shadow_module.load_state_dict(state_dict)
