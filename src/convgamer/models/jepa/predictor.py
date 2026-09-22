"""V-JEPA 2.1 predictor (§2.3.1, §2.3.2).

Takes the x-encoder output (context + mask tokens) and produces dense
predictions at multiple encoder levels.

Architecture (per Appendix A):
    - 24 predictor blocks, embedding dim 3584 for ViT-G.
    Here adapted for ConvNeXt-Tiny scale (~35M params).
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["VJEPAPredictor"]


class _GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2): L2 aggregate, divisive norm."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(1, 2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        shape = (1, *[1] * (x.ndim - 2), -1)
        out = self.gamma.view(shape) * (x * nx) + self.beta.view(shape) + x
        return out


class _ConvNeXtBlock(nn.Module):
    """ConvNeXt V2-style block: depthwise conv, LayerNorm, MLP, GRN."""

    def __init__(self, dim: int, expansion: int = 4, layer_scale_init: float = 1e-6):
        super().__init__()
        self.dw = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, expansion * dim)
        self.pw2 = nn.Linear(expansion * dim, dim)
        self.grn = _GRN(expansion * dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        residual = x
        x = self.dw(x)
        x = x.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
        x = self.norm(x)
        x = self.pw1(x)
        x = torch.nn.functional.gelu(x)
        x = self.grn(x)
        x = self.pw2(x)
        x = x.permute(0, 4, 1, 2, 3)  # back to (B, C, T, H, W)
        return residual + self.gamma.view(1, -1, 1, 1, 1) * x


class VJEPAPredictor(nn.Module):
    """Dense predictor for V-JEPA 2.1 latent target prediction.

    Single-level output: the encoder exposes only its final feature map, so
    ``forward`` returns a dense prediction over the full spatial-temporal grid.
    Deep multi-level supervision (§2.3.2) can be added when intermediate
    encoder levels are wired in.

    Args:
        feature_dim: Channel dimension of encoder output (B, F, T, H, W).
        predictor_dim: Hidden width of predictor blocks.
        num_layers: Number of ConvNeXt-style blocks in the predictor trunk.
        projection_dim: Output channel dim of the prediction head (must match
            the target encoder's feature dim).
        expansion: MLP expansion ratio inside each predictor block.
        layer_scale_init: Initial value for the per-block ``gamma`` residual
            scale. Small values start each block near identity.
    """

    def __init__(
        self,
        feature_dim: int = 384,
        predictor_dim: int = 384,
        num_layers: int = 4,
        projection_dim: int | None = None,
        expansion: int = 4,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.predictor_dim = predictor_dim
        self.projection_dim = projection_dim or feature_dim
        self.expansion = expansion
        self.layer_scale_init = layer_scale_init

        # Project encoder features to predictor working dimension
        self.input_proj = nn.Conv3d(feature_dim, predictor_dim, kernel_size=1)
        self.mask_token = nn.Parameter(torch.zeros(1, predictor_dim))

        self.blocks = nn.Sequential(
            *[
                _ConvNeXtBlock(
                    predictor_dim, expansion=expansion, layer_scale_init=layer_scale_init
                )
                for _ in range(num_layers)
            ]
        )

        # Single prediction head (V-JEPA 2.1 §2.3.2: dense 1×1×1 conv).
        self.head = nn.Conv3d(predictor_dim, self.projection_dim, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Truncated-normal init for stable JEPA training (no large projections)."""
        from torch.nn.init import trunc_normal_

        all_convs = [self.input_proj, self.head]
        for m in all_convs:
            w = m.weight
            if not isinstance(w, torch.Tensor):
                raise TypeError(f"Conv weight must be a Tensor, got {type(w).__name__}")
            trunc_normal_(w, std=0.02)
            b = m.bias
            if b is not None:
                if not isinstance(b, torch.Tensor):
                    raise TypeError(f"Conv bias must be a Tensor, got {type(b).__name__}")
                nn.init.zeros_(b)
        nn.init.zeros_(self.mask_token)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Dense prediction on visible + masked tokens.

        Args:
            x: Encoder features ``(B, F, T, H, W)``.
            mask: Boolean mask ``(B, T, H, W)`` — True = masked (to be predicted).

        Returns:
            Dense predictions ``(B, projection_dim, T, H, W)``.
        """
        h = self.input_proj(x)  # (B, predictor_dim, T, H, W)

        # Nearest resize: mask marks discrete tokens, not a smooth field.
        _, _, ft, fh, fw = h.shape
        mask_resized = (
            nn.functional.interpolate(
                mask.float().unsqueeze(1),
                size=(ft, fh, fw),
                mode="nearest",
            )
            .bool()
            .squeeze(1)
        )

        # Inject mask token at masked positions
        mask_b = mask_resized.unsqueeze(1).expand(-1, self.predictor_dim, -1, -1, -1)
        h = torch.where(mask_b, self.mask_token.view(1, -1, 1, 1, 1), h)

        h = self.blocks(h)
        return self.head(h)
