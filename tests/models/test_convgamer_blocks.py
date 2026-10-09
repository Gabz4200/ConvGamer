"""Behavior tests for LearnedSpatialTemporalDownsampler.

Behavioral contracts covered:

  - LearnedSpatialTemporalDownsampler: forward rejects non-5D input and channel
    mismatches, maps (B, C, T, H, W) -> (B, C*out_factor, T, H/P, W/P) via a
    strided patchify stem plus a low-res residual correction. T is preserved
    (only spatial downsampling); aspect ratio is preserved (no forced square).
"""

import pytest
import torch

from convgamer.models.convgamer.downsampler import (
    LearnedSpatialTemporalDownsampler,
)

# ── Input validation ─────────────────────────────────────────────────────────


def test_when_non_5d_input_then_raises() -> None:
    """forward expects (B, C, T, H, W); 4D must raise ValueError."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3)
    x = torch.randn(1, 3, 8, 8)
    with pytest.raises(ValueError, match="Expected 5D input"):
        op(x)


def test_when_wrong_channel_count_then_raises() -> None:
    """Channel dim must match in_channels."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3)
    x = torch.randn(1, 5, 8, 16, 16)
    with pytest.raises(ValueError, match="in_channels=3"):
        op(x)


def test_when_non_positive_expansion_then_raises() -> None:
    """Correction width stays a positive multiple of base width by construction."""
    with pytest.raises(ValueError, match="expansion"):
        LearnedSpatialTemporalDownsampler(in_channels=3, expansion=0)


def test_when_bad_correction_kernel_then_raises() -> None:
    """Only 'dw7' and 'dense4' correction kernels are supported."""
    with pytest.raises(ValueError, match="correction_kernel"):
        LearnedSpatialTemporalDownsampler(in_channels=3, correction_kernel="big")


def test_when_custom_expansion_then_correction_width_scales() -> None:
    """expansion sets the correction width as base_channels * expansion."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3, expansion=4)
    assert op.intermediate_channels == 3 * 2 * 4
    x = torch.randn(1, 3, 8, 32, 32)
    with torch.no_grad():
        out = op(x)
    assert out.shape == (1, 6, 8, 4, 4)  # 32/8 patchify


@pytest.mark.parametrize("correction_kernel", ["dw7", "dense4"])
def test_when_either_kernel_then_shapes_match(correction_kernel: str) -> None:
    """Both kernel choices preserve the (B, C*F, T, H/P, W/P) layout."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3, correction_kernel=correction_kernel)
    x = torch.randn(1, 3, 4, 32, 32)
    with torch.no_grad():
        out = op(x)
    assert out.shape == (1, 6, 4, 4, 4)


# ── Output shape / channel layout ────────────────────────────────────────────


def test_when_temporal_preserving_then_temporal_dim_unchanged() -> None:
    """T dimension is preserved: only spatial downsampling happens."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3)
    x = torch.randn(1, 3, 10, 32, 32)
    with torch.no_grad():
        out = op(x)
    assert out.shape[2] == 10  # T preserved


def test_when_16_by_9_input_then_aspect_preserved() -> None:
    """Native game aspect is preserved: 256x144 -> 32x18, never squared."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3)
    x = torch.randn(1, 3, 4, 144, 256)
    with torch.no_grad():
        out = op(x)
    assert out.shape == (1, 6, 4, 18, 32)


@pytest.mark.parametrize("out_factor", [1, 2, 4, 8])
def test_when_scaled_out_factor_then_channels_scale_linearly(out_factor: int) -> None:
    """Output channels scale by out_factor."""
    op = LearnedSpatialTemporalDownsampler(
        in_channels=3,
        out_factor=out_factor,
    )
    x = torch.randn(1, 3, 4, 16, 16)
    with torch.no_grad():
        out = op(x)
    assert out.shape[1] == 3 * out_factor


@pytest.mark.parametrize("patch_size", [4, 8])
def test_when_patch_size_then_spatial_scales(patch_size: int) -> None:
    """Output spatial dims follow the patchify stride."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3, patch_size=patch_size)
    x = torch.randn(1, 3, 2, 32, 32)
    with torch.no_grad():
        out = op(x)
    assert out.shape[3:] == (32 // patch_size, 32 // patch_size)


# ── Numerical behavior ───


def test_output_finite_including_zero_input() -> None:
    """No NaN/Inf for standard float or zero input (LayerNorm-safe)."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3)
    x = torch.randn(1, 3, 8, 32, 32)
    with torch.no_grad():
        assert torch.isfinite(op(x)).all()
        assert torch.isfinite(op(torch.zeros_like(x))).all()


def test_when_correction_zeroed_then_output_matches_patch_path() -> None:
    """Zeroed correction with identity norm: output equals the patchified base."""
    op = LearnedSpatialTemporalDownsampler(
        in_channels=3,
        out_factor=1,
        expansion=5,
    )
    with torch.no_grad():
        for p in list(op.correction_conv.parameters()) + list(op.out_correction_conv.parameters()):
            torch.nn.init.zeros_(p)
        torch.nn.init.ones_(op.norm.weight)
        torch.nn.init.zeros_(op.norm.bias)

        x = torch.randn(1, 3, 8, 32, 32)
        out = op(x)
        expected = op.norm(op.patch_embed(x))

    torch.testing.assert_close(out, expected)
