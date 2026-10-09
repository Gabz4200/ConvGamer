"""Tests for the shared token-mask resize helper.

Both the predictor and the loss resize the boolean token mask to feature-map
dims with nearest interpolation (mask marks discrete tokens, so binary
boundaries must survive). One helper, tested once, used twice.
"""

import pytest
import torch

from convgamer.models.jepa.masking import resize_token_mask


def test_resize_token_mask_matches_feature_dims() -> None:
    mask = torch.zeros(2, 4, 8, 8, dtype=torch.bool)
    mask[:, :, :4, :4] = True
    out = resize_token_mask(mask, (2, 4, 4))
    assert out.shape == (2, 2, 4, 4)
    assert out.dtype == torch.bool


def test_resize_token_mask_preserves_binary_boundaries() -> None:
    # Upsampling replicates blocks exactly (no blending introduces fractions).
    mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
    mask[0, 0, 0, 0] = True
    out = resize_token_mask(mask, (1, 4, 4))
    assert out[0, 0, :2, :2].all()
    assert not out[0, 0, 2:, :].any()
    assert not out[0, 0, :, 2:].any()


def test_resize_token_mask_identity_size_is_exact() -> None:
    # Same-size resize is a no-op on values.
    torch.manual_seed(0)
    mask = torch.rand(2, 3, 5, 5) < 0.5
    out = resize_token_mask(mask, (3, 5, 5))
    torch.testing.assert_close(out, mask)


def test_resize_token_mask_rejects_non_bool() -> None:
    with pytest.raises(ValueError, match="bool"):
        resize_token_mask(torch.zeros(1, 2, 4, 4), (2, 4, 4))


def test_resize_token_mask_rejects_bad_rank() -> None:
    with pytest.raises(ValueError, match="4D"):
        resize_token_mask(torch.zeros(2, 4, 4, dtype=torch.bool), (2, 4, 4))
