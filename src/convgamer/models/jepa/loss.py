"""V-JEPA 2.1 dense predictive loss (§2.3.1).

L_dense = L_predict + L_ctx

  L_predict = (1/|M|) * sum_{i in M} || P(E_theta(x), Delta_y)_i - sg(E_theta_bar(y)_i) ||_1
  L_ctx     = (1/|C|) * sum_{i in C} lambda_i * || P(...) - sg(E_theta_bar(y)_i) ||_1

  lambda_i = lambda / sqrt(d_min(i, M))     (Eq. 3, Appendix A: lambda=0.5 video, 0.7 image)
"""

from __future__ import annotations

import warnings

import torch
from torch import nn

from .masking import resize_token_mask

__all__ = ["JEPALoss", "compute_context_lambdas"]


def compute_context_lambdas(
    mask: torch.Tensor,
    lambda_base: float = 0.5,
) -> torch.Tensor:
    """Distance-weighted context lambda (Eq. 3).

    ``mask`` is ``(B, T, H, W)`` boolean — True at masked positions.
    Returns ``(B, T, H, W)`` floats; 0 at masked positions, decreasing
    with spatio-temporal distance to nearest mask token.
    """
    B, T, H, W = mask.shape
    device = mask.device

    # Build coordinate grid
    coords = []
    for d in (T, H, W):
        coords.append(torch.arange(d, device=device, dtype=torch.float))
    t_grid, h_grid, w_grid = torch.meshgrid(*coords, indexing="ij")
    t_grid = t_grid.expand(B, -1, -1, -1)
    h_grid = h_grid.expand(B, -1, -1, -1)
    w_grid = w_grid.expand(B, -1, -1, -1)

    # Flatten for distance computation
    mask_flat = mask.reshape(B, -1)  # (B, T*H*W)
    t_flat = t_grid.reshape(B, -1)
    h_flat = h_grid.reshape(B, -1)
    w_flat = w_grid.reshape(B, -1)

    lambdas = torch.zeros_like(mask_flat, dtype=torch.float)
    has_mask = mask_flat.any(dim=1)

    if not has_mask.any():
        warnings.warn(
            "compute_context_lambdas got a batch with no masked tokens.",
            UserWarning,
            stacklevel=2,
        )
        return lambdas.reshape(B, T, H, W)

    # Chunked: for every token, distance to its nearest masked token.
    # Chunk the query dim so peak memory is O(chunk * M), not O(N * M).
    for b in range(B):
        if not mask_flat[b].any():
            continue
        masked_idx = mask_flat[b].nonzero(as_tuple=True)[0]  # (M,)
        mt = t_flat[b][masked_idx]
        mh = h_flat[b][masked_idx]
        mw = w_flat[b][masked_idx]
        d_min = torch.empty(mask_flat.shape[1], device=device, dtype=torch.float)
        for chunk in torch.split(torch.arange(mask_flat.shape[1], device=device), 1024):
            dist = (
                (t_flat[b][chunk].unsqueeze(1) - mt.unsqueeze(0)).abs()
                + (h_flat[b][chunk].unsqueeze(1) - mh.unsqueeze(0)).abs()
                + (w_flat[b][chunk].unsqueeze(1) - mw.unsqueeze(0)).abs()
            )
            d_min[chunk] = dist.min(dim=1).values
        # Adjacent (distance-1) context keeps full lambda_base; masked get 0.
        d_min = torch.clamp(d_min, min=1.0)  # avoid div-by-zero at masked positions
        weight = lambda_base / torch.sqrt(d_min)
        # Masked positions: lambda = 0 (no L_ctx on masked)
        weight = weight * (~mask_flat[b]).float()
        lambdas[b] = weight

    return lambdas.reshape(B, T, H, W)


class JEPALoss(nn.Module):
    """Dense predictive loss for V-JEPA 2.1, averaged over encoder levels.

    Args:
        lambda_base: Base context-loss weight (0.5 video, 0.7 image per Appendix A).
        lambda_image: Context weight for static-image (T=1) samples.
        lambda_warmup_steps: Linear warmup steps for ``lambda`` (epochs 50-100
            in the paper); 0 disables the schedule.
    """

    def __init__(
        self,
        lambda_base: float = 0.5,
        lambda_image: float = 0.7,
        lambda_warmup_steps: int = 0,
    ):
        super().__init__()
        self.lambda_base = lambda_base
        self.lambda_image = lambda_image
        self.lambda_warmup_steps = max(0, int(lambda_warmup_steps))
        # Warmup schedule (V-JEPA 2.1 ramps lambda epochs 50-100): the step
        # counter must survive resume, so the buffer stays persistent.
        self._step_counter: torch.Tensor
        self.register_buffer("_step_counter", torch.zeros((), dtype=torch.long))
        self._step_int: int = 0

    def set_step(self, step: int) -> None:
        """Update the running step counter used by the lambda warmup."""
        self._step_int = int(step)
        self._step_counter.fill_(int(step))

    def _effective_lambda(self, mask: torch.Tensor) -> float:
        """Pick per-modality lambda (video vs image) and apply warmup."""
        # T=1 samples are static images; T>1 are video.
        is_image = mask.shape[1] == 1
        base = self.lambda_image if is_image else self.lambda_base
        if self.lambda_warmup_steps <= 0:
            return base
        # Linear ramp from 0 to base over the warmup window (paper: epochs 50-100).
        ramp = min(1.0, float(self._step_int) / float(self.lambda_warmup_steps))
        return base * ramp

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute L_dense = L_predict + L_ctx, averaged over levels.

        Args:
            pred: Predicted features ``(B, F, T, H, W)`` or per-level
                ``(B, L, F, T, H, W)``.
            target: Target encoder features, same layout as ``pred`` (already
                stop-grad'd).
            mask: Boolean mask ``(B, T, H, W)`` — True at masked positions.

        Returns:
            Scalar loss (mean over levels when ``L > 1``).
        """
        # Stop-gradient on target (V-JEPA 2.1 §2.1)
        target = target.detach()

        if pred.ndim == 5:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)
        if pred.ndim != 6 or target.ndim != 6:
            raise ValueError(f"pred/target must be 5D or 6D, got {pred.ndim}D/{target.ndim}D")
        if pred.shape != target.shape:
            raise ValueError(f"pred {tuple(pred.shape)} must match target {tuple(target.shape)}")
        b, num_levels = pred.shape[0], pred.shape[1]
        flat_pred = pred.reshape(b * num_levels, *pred.shape[2:])
        flat_target = target.reshape(b * num_levels, *target.shape[2:])

        _, _, ft, fh, fw = flat_pred.shape
        mask_resized = resize_token_mask(mask, (ft, fh, fw))
        eff_lambda = self._effective_lambda(mask)
        if not mask_resized.any():
            warnings.warn(
                "JEPALoss got a batch with no masked tokens; loss is 0.",
                UserWarning,
                stacklevel=2,
            )
        # The level fold shares one mask and one lambda map: tile across levels.
        mask_folded = mask_resized.repeat_interleave(num_levels, dim=0)
        lambdas = compute_context_lambdas(mask_resized, lambda_base=eff_lambda).repeat_interleave(
            num_levels, dim=0
        )

        # Per-token L1 distance
        l1 = (flat_pred - flat_target).abs()
        token_loss = l1.mean(dim=1)

        # L_predict: only on masked tokens
        pred_loss = token_loss.masked_fill(~mask_folded, 0.0)
        n_masked = mask_folded.sum().clamp(min=1)
        l_predict = pred_loss.sum() / n_masked

        # L_ctx: distance-weighted, only on context (visible) tokens
        ctx_loss = token_loss * lambdas  # already 0 at masked
        n_ctx = (~mask_folded).sum().clamp(min=1)
        l_ctx = ctx_loss.sum() / n_ctx

        return l_predict + l_ctx
