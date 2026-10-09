"""Tests for multi-block spatio-temporal masking (V-JEPA 2/2.1 App A).

Masks are unions of 3D blocks at token resolution — not iid pixel noise.
"""

import torch

from convgamer.data.masking import multiblock_mask_3d


def test_mask_shape_dtype_and_coverage() -> None:
    torch.manual_seed(0)
    mask = multiblock_mask_3d(8, 32, 32, mask_ratio=0.25, seed=0)
    assert mask.shape == (8, 32, 32)
    assert mask.dtype == torch.bool
    cov = mask.float().mean().item()
    assert 0.25 <= cov <= 0.55  # at least ratio, bounded overshoot


def test_mask_is_block_structured_not_noise() -> None:
    """A block mask has far fewer connected speckles than iid noise."""
    mask = multiblock_mask_3d(4, 32, 32, mask_ratio=0.25, seed=3)
    col_changes = (mask[:, :, 1:] != mask[:, :, :-1]).float().mean().item()
    assert col_changes < 0.20  # iid noise would sit at ~0.375 for p=0.25


def test_mask_reproducible_with_seed() -> None:
    a = multiblock_mask_3d(4, 16, 16, mask_ratio=0.3, seed=11)
    b = multiblock_mask_3d(4, 16, 16, mask_ratio=0.3, seed=11)
    torch.testing.assert_close(a, b)


def test_mask_t1_images_get_spatial_blocks() -> None:
    """T=1 (static images) still yields 2D block structure."""
    mask = multiblock_mask_3d(1, 32, 32, mask_ratio=0.1, seed=5)
    assert mask.shape == (1, 32, 32)
    assert mask.any() and not mask.all()
