"""V-JEPA 2.1 predictor (§2.3.1, §2.3.2).

Takes the x-encoder output (context + mask tokens) and produces dense
predictions. The predictor is a Transformer with multi-head attention
where cross-head interaction happens in the FFN / output projection.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["VJEPAPredictor"]


class _GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2): L2 aggregate, divisive norm."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, C) - channel last format
        gx = torch.norm(x, p=2, dim=(1, 2, 3), keepdim=True)  # (B, 1, 1, 1, C)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)  # mean over channel dim
        out = x * (nx * self.gamma.view(1, 1, 1, 1, -1) + self.beta.view(1, 1, 1, 1, -1))
        return out


class _ConvNeXtBlock(nn.Module):
    """ConvNeXt V2-style block: depthwise conv, LayerNorm, MLP, GRN."""

    def __init__(self, dim: int, expansion: int = 4, layer_scale_init: float = 1e-6):
        super().__init__()
        self.dw = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            _GRN(dim * expansion),
            nn.Linear(dim * expansion, dim),
        )
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        residual = x
        x = self.dw(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 4, 1, 2, 3)
        return residual + self.gamma.view(1, -1, 1, 1, 1) * x


class _MultiHeadAttentionBlock(nn.Module):
    """Multi-head self-attention over spatial-temporal tokens with per-head QKV."""

    def __init__(self, dim: int, num_heads: int, layer_scale_init: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv3d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        residual = x

        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, T, H, W)
        q, k, v = qkv.unbind(1)  # each (B, num_heads, head_dim, T, H, W)

        # Flatten spatial-temporal dims for attention
        q = q.reshape(B, self.num_heads, self.head_dim, -1)  # (B, H, d, N)
        k = k.reshape(B, self.num_heads, self.head_dim, -1)
        v = v.reshape(B, self.num_heads, self.head_dim, -1)

        attn = (q.transpose(-2, -1) @ k) * self.scale  # (B, H, N, N)
        attn = F.softmax(attn, dim=-1)
        out = (v @ attn.transpose(-2, -1)).reshape(B, self.num_heads, self.head_dim, T, H, W)

        out = out.reshape(B, C, T, H, W)
        out = self.proj(out)

        # LayerNorm over channels
        out = out.permute(0, 2, 3, 4, 1)
        out = self.norm(out)
        out = out.permute(0, 4, 1, 2, 3)

        return residual + self.gamma.view(1, -1, 1, 1, 1) * out


class VJEPAPredictor(nn.Module):
    """Transformer predictor for V-JEPA 2.1 latent target prediction.

    Multi-head attention (§2.3.1) with cross-head interaction in the FFN.
    ``num_heads`` controls the number of attention heads inside the predictor.
    ``num_levels`` controls the number of encoder levels to predict.
    Input can be single-level (B, F, T, H, W) or multi-level (B, L, F, T, H, W).
    Output matches input level layout: single-level returns (B, projection_dim, T, H, W),
    multi-level returns (B, num_levels, projection_dim, T, H, W).

    Args:
        feature_dim: Channel dimension of encoder output (B, F, T, H, W).
        predictor_dim: Hidden width of predictor Transformer.
        num_layers: Number of Transformer blocks.
        num_heads: Number of attention heads in multi-head attention (default 4).
        num_levels: Number of encoder levels to predict (default 1). When >1,
            input and output gain a level axis at position 1.
        projection_dim: Output channel dim of the prediction head.
        expansion: MLP expansion ratio inside each predictor block.
        layer_scale_init: Initial value for the per-block residual scale.
    """

    def __init__(
        self,
        feature_dim: int = 384,
        predictor_dim: int = 384,
        num_layers: int = 4,
        num_heads: int = 4,
        num_levels: int = 1,
        projection_dim: int | None = None,
        expansion: int = 4,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")
        if predictor_dim % num_heads != 0:
            raise ValueError(
                f"predictor_dim {predictor_dim} must be divisible by num_heads {num_heads}"
            )

        self.feature_dim = feature_dim
        self.predictor_dim = predictor_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.projection_dim = projection_dim or feature_dim
        self.expansion = expansion
        self.layer_scale_init = layer_scale_init

        # Input projection: shared across levels
        self.input_proj = nn.Conv3d(feature_dim, predictor_dim, kernel_size=1)
        self.mask_token = nn.Parameter(torch.zeros(1, predictor_dim))

        # Transformer blocks: alternating attention + ConvNeXt (local) blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                # Global attention block
                self.blocks.append(
                    _MultiHeadAttentionBlock(predictor_dim, num_heads, layer_scale_init)
                )
            else:
                # Local ConvNeXt block
                self.blocks.append(_ConvNeXtBlock(predictor_dim, expansion, layer_scale_init))

        # Output projection with cross-level interaction when num_levels > 1
        proj_dim = self.projection_dim
        if num_levels > 1:
            # Cross-level FFN: (B, L, C, T, H, W) -> (B, L, C, T, H, W)
            self.level_ffn = nn.Sequential(
                nn.Conv3d(
                    num_levels * predictor_dim,
                    num_levels * predictor_dim,
                    kernel_size=1,
                    groups=num_levels,
                ),
                nn.GELU(),
                nn.Conv3d(
                    num_levels * predictor_dim,
                    num_levels * proj_dim,
                    kernel_size=1,
                    groups=num_levels,
                ),
            )
        else:
            self.head = nn.Conv3d(predictor_dim, proj_dim, kernel_size=1)

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
            x: Encoder features ``(B, F, T, H, W)`` (single-level) or
                ``(B, num_levels, F, T, H, W)`` (multi-level).
            mask: Boolean mask ``(B, T, H, W)`` — True = masked (to be predicted).

        Returns:
            ``num_levels == 1``: predictions ``(B, projection_dim, T, H, W)``.
            ``num_levels > 1``: predictions ``(B, num_levels, projection_dim, T, H, W)``.
        """
        multi = x.ndim == 6
        if multi:
            if x.shape[1] != self.num_levels:
                raise ValueError(f"input level axis {x.shape[1]} != num_levels={self.num_levels}")
            if self.num_levels == 1:
                x = x.squeeze(1)
                multi = False
                b = x.shape[0]
            else:
                b = x.shape[0]
                x = x.reshape(b * self.num_levels, *x.shape[2:])
        else:
            b = x.shape[0]

        h = self.input_proj(x)  # (B*L, predictor_dim, T, H, W)
        _, _, ft, fh, fw = h.shape
        h = h.reshape(b * (self.num_levels if multi else 1), self.predictor_dim, ft, fh, fw)

        # Nearest resize: mask marks discrete tokens, not a smooth field.
        mask_resized = (
            F.interpolate(
                mask.float().unsqueeze(1),
                size=(ft, fh, fw),
                mode="nearest",
            )
            .bool()
            .squeeze(1)
        )

        # Inject the (shared) mask token at masked positions.
        nb = b * (self.num_levels if multi else 1)
        mask_b = (
            mask_resized.unsqueeze(1)
            .expand(-1, self.num_levels if multi else 1, ft, fh, fw)
            .reshape(nb, 1, ft, fh, fw)
            .expand(-1, self.predictor_dim, -1, -1, -1)
        )
        h = torch.where(mask_b, self.mask_token.view(1, -1, 1, 1, 1), h)

        for blk in self.blocks:
            h = blk(h)

        if self.num_levels == 1 or not multi:
            return self.head(h)

        # Cross-level FFN: fold levels into channel dim, apply grouped convs,
        # then unfold back to (B, L, projection_dim, T, H, W)
        h = h.reshape(b, self.num_levels, self.predictor_dim, ft, fh, fw)
        h_flat = h.reshape(b, self.num_levels * self.predictor_dim, ft, fh, fw)
        h_flat = self.level_ffn(h_flat)
        return h_flat.reshape(b, self.num_levels, self.projection_dim, ft, fh, fw)
