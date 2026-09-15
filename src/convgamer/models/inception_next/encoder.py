"""InceptionNeXt encoder — the full network from stem to classification head.

Architecture follows Section 3.3 of arXiv:2303.16900: a ConvNeXt-style
backbone whose token mixer in every block is the Inception depthwise
convolution (``InceptionDWConv2d``) from ``blocks.py``.

Key deviations from the paper are intentional design choices:
- LayerNorm instead of BatchNorm (numerically stable, works on any batch size).
- MLP ratio 4 in stages 1–3, 3 in stage 4 (paper uses 3 in stage 4 to save FLOPs).
- Configurable per-stage layer counts via ``num_layers`` (int or 4-tuple).
"""

from __future__ import annotations

import torch
from torch import nn

from ..base import BaseModel
from ..registry import register_model
from .blocks import InceptionNeXtBlock


def _normalize_layer_count(
    num_layers: int | list[int] | tuple[int, ...], stages: int = 4
) -> list[int]:
    """Broadcast ``num_layers`` to a per-stage list of length *stages*.

    Accepts OmegaConf ListConfig (Hydra) as well as plain list/tuple.
    """
    if isinstance(num_layers, int) and not isinstance(num_layers, bool):
        return [num_layers] * stages
    try:
        layers = list(num_layers)  # type: ignore[arg-type]
    except TypeError:
        raise ValueError(
            f"num_layers must be int or sequence, got {type(num_layers).__name__}"
        ) from None
    if len(layers) != stages:
        raise ValueError(f"num_layers tuple must have length {stages}, got {len(layers)}")
    return layers


@register_model("InceptionNeXtEncoder")
class InceptionNeXtEncoder(BaseModel):
    """InceptionNeXt backbone.

    Parameters
    ----------
    input_dim:
        Number of input image channels (3 for RGB).
    hidden_dim:
        Channel width for the first stage; doubles each subsequent stage.
        Default 96 matches the paper's Tiny/Small configs (Table 3).
    num_layers:
        Number of ``InceptionNeXtBlock`` per stage.  Pass an ``int`` to use
        the same count in every stage, or a 4-tuple to match the paper's
        stage layout (e.g. ``(3, 3, 9, 3)`` for Tiny/Small).
    num_classes:
        Output dimension of the classification head.  ``0`` omits the head
        so the model can be used as a pure feature extractor.
    layer_scale_init:
        Initial value for the per-block ``gamma`` parameter (Section 3.3,
        paper uses ``1e-6``).
    mlp_ratios:
        Expansion ratios for the four stages.  The paper uses 4 for stages
        1–3 and 3 for stage 4 (to save ~3% FLOPs in Base).  Defaults to
        ``(4, 4, 4, 3)``.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 96,
        num_layers: int | tuple[int, ...] = 3,
        num_classes: int = 1000,
        layer_scale_init: float = 1e-6,
        mlp_ratios: tuple[int, int, int, int] = (4, 4, 4, 3),
        widths: tuple[int, int, int, int] | None = None,
    ):
        super().__init__()

        if len(mlp_ratios) != 4:
            raise ValueError(f"mlp_ratios must have length 4, got {len(mlp_ratios)}")
        if widths is None:
            widths = (hidden_dim, hidden_dim * 2, hidden_dim * 4, hidden_dim * 8)
        if len(widths) != 4:
            raise ValueError(f"widths must have length 4, got {len(widths)}")
        self.stem = nn.Conv2d(input_dim, widths[0], kernel_size=4, stride=4)
        self.stem_norm = nn.LayerNorm(widths[0])

        layers_per_stage = _normalize_layer_count(num_layers, stages=4)

        stages: list[nn.Module] = []
        last_out_ch = widths[0]
        for stage_idx in range(4):
            in_ch = widths[stage_idx] if stage_idx == 0 else widths[stage_idx - 1]
            out_ch = widths[stage_idx]
            last_out_ch = out_ch
            stage: list[nn.Module] = []
            if stage_idx > 0:
                stage.append(nn.Conv2d(in_ch, out_ch, kernel_size=1))
            mlp_ratio = mlp_ratios[stage_idx]
            for _ in range(layers_per_stage[stage_idx]):
                stage.append(InceptionNeXtBlock(out_ch, out_ch * mlp_ratio, layer_scale_init))
            stages.append(nn.Sequential(*stage))

        self.stages = nn.Sequential(*stages)
        self.feature_dim = last_out_ch
        self.head = nn.Linear(last_out_ch, num_classes) if num_classes > 0 else nn.Identity()

    def _stem_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.stem_norm(x)
        return x.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_feature_map(x)
        x = x.mean(dim=[2, 3])
        return self.head(x)

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial feature map (B, F, H', W'); no pooling, no head."""
        x = self._stem_forward(x)
        return self.stages(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Spatially pooled frame features (B, F); resolution collapsed by mean."""
        return self.forward_feature_map(x).mean(dim=[2, 3])
