"""V-JEPA 2.1 predictor (§2.3.1, §2.3.2).

Takes the fused x-encoder context (plus mask tokens) and produces one dense
prediction per encoder level. The predictor is a stack of local conv blocks —
no attention, no GRN, no layer-scale: the predictor is train-time only, so
capacity is better spent on the encoder.
"""

from __future__ import annotations

import torch
from torch import nn

from .masking import resize_token_mask

__all__ = ["VJEPAPredictor"]


class _LocalPredictorBlock(nn.Module):
    """Depthwise conv + LayerNorm + MLP with residual (local token mixing)."""

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.dw = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        residual = x
        x = self.dw(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 4, 1, 2, 3)
        return residual + x


class VJEPAPredictor(nn.Module):
    """Local-conv predictor for V-JEPA 2.1 multi-level target prediction.

    Input is the fused context ``(B, F, T, H, W)``; output is one prediction
    per encoder level ``(B, num_outputs, F, T, H, W)`` via a single 1x1 head.

    Args:
        feature_dim: Channel dimension of fused encoder output.
        predictor_dim: Hidden width of the predictor stack.
        num_layers: Number of local predictor blocks.
        num_outputs: Number of encoder levels to predict (default 4).
        expansion: MLP expansion ratio inside each predictor block.
    """

    def __init__(
        self,
        feature_dim: int = 384,
        predictor_dim: int = 384,
        num_layers: int = 4,
        num_outputs: int = 4,
        expansion: int = 4,
    ):
        super().__init__()
        if num_outputs < 1:
            raise ValueError(f"num_outputs must be >= 1, got {num_outputs}")

        self.feature_dim = feature_dim
        self.predictor_dim = predictor_dim
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.expansion = expansion

        self.input_proj = nn.Conv3d(feature_dim, predictor_dim, kernel_size=1)
        self.mask_token = nn.Parameter(torch.zeros(1, predictor_dim))

        self.blocks = nn.ModuleList(
            [_LocalPredictorBlock(predictor_dim, expansion) for _ in range(num_layers)]
        )

        self.head = nn.Conv3d(predictor_dim, num_outputs * feature_dim, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Truncated-normal init for stable JEPA training (no large projections)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.mask_token)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Dense prediction on visible + masked tokens.

        Args:
            x: Fused encoder features ``(B, F, T, H, W)``.
            mask: Boolean mask ``(B, T, H, W)`` — True = masked (to be predicted).

        Returns:
            Per-level predictions ``(B, num_outputs, F, T, H, W)``.
        """
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (B, F, T, H, W), got {x.ndim}D")
        b = x.shape[0]

        h = self.input_proj(x)  # (B, predictor_dim, T, H, W)
        _, _, ft, fh, fw = h.shape

        mask_resized = resize_token_mask(mask, (ft, fh, fw))

        # Inject the (shared) mask token at masked positions. The singleton
        # dim broadcasts — never materialize a full B×D×T×H×W mask.
        mask_key = mask_resized.unsqueeze(1)  # (B, 1, T, H, W)
        h = torch.where(mask_key, self.mask_token.view(1, -1, 1, 1, 1), h)

        for blk in self.blocks:
            h = blk(h)

        out = self.head(h)  # (B, num_outputs*F, T, H, W)
        return out.reshape(b, self.num_outputs, self.feature_dim, ft, fh, fw)
