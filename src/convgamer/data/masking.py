"""Multi-block masking for JEPA training (V-JEPA 2/2.1 App A style).

Masks are unions of random 3D (time × height × width) blocks sampled until
the masked fraction reaches ``mask_ratio`` — block structure at token
resolution, never iid per-pixel noise. Masking is strictly a train-time
data-layer augmentation; the encoder and streaming path never see masks.
"""

from __future__ import annotations

import torch

__all__ = ["multiblock_mask_3d"]


def _uniform(g: torch.Generator, lo: float, hi: float) -> float:
    return lo + (hi - lo) * float(torch.rand((), generator=g))


def multiblock_mask_3d(
    num_frames: int,
    height: int,
    width: int,
    mask_ratio: float = 0.25,
    seed: int | None = None,
    min_scale: float = 0.15,
    max_scale: float = 0.7,
) -> torch.Tensor:
    """Sample a boolean ``(T, H, W)`` mask as a union of random 3D blocks.

    Block spatial sides are uniform in ``[min_scale, max_scale]`` of each dim
    (V-JEPA 2 spatial mask scale); the temporal span is uniform in
    ``[1, T]``. Blocks are unioned until coverage reaches ``mask_ratio``.
    """
    if num_frames <= 0 or height <= 0 or width <= 0:
        raise ValueError("T, H, W must be positive")
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    mask = torch.zeros(num_frames, height, width, dtype=torch.bool)
    if mask_ratio == 0.0:
        return mask
    target = mask_ratio * mask.numel()
    guard = 0
    while mask.sum().item() < target and guard < 64:
        guard += 1
        bt = int(torch.randint(1, num_frames + 1, (1,), generator=g).item())
        bh = max(1, int(height * _uniform(g, min_scale, max_scale)))
        bw = max(1, int(width * _uniform(g, min_scale, max_scale)))
        t0 = int(torch.randint(0, num_frames - bt + 1, (1,), generator=g).item())
        h0 = int(torch.randint(0, height - bh + 1, (1,), generator=g).item())
        w0 = int(torch.randint(0, width - bw + 1, (1,), generator=g).item())
        mask[t0 : t0 + bt, h0 : h0 + bh, w0 : w0 + bw] = True
    return mask
