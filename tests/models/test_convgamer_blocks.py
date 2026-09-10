"""Behavior tests for LearnedSpatialTemporalDownsampler.

Behavioral contracts covered:

  - forward rejects non-5D input and channel mismatches (boundary validation).
  - forward maps (B, C, T, H, W) -> (B, C*out_factor + (C if concat) ?,
    T // temporal_reduction_factor, target_h, target_w).
  - concat_original=False yields base_channels = C*out_factor output channels.
  - output is finite for typical float and zero inputs.
  - with the correction branch zeroed, concat=False and out_factor=1, the
    output equals the normalized downsampled path (residual algebra sanity).
  - uniform_temporal_subsample selects equispaced indices, includes endpoints,
    preserves ordering, clamps on oversample, and rejects invalid counts.
"""

import einops
import pytest
import torch

from convgamer.models.convgamer.blocks import (
    LearnedSpatialTemporalDownsampler,
    uniform_temporal_subsample,
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


def test_when_depthwise_and_intermediate_not_divisible_then_raises() -> None:
    """Depthwise conv requires intermediate_channels % in_channels == 0."""
    with pytest.raises(ValueError, match="must be divisible"):
        LearnedSpatialTemporalDownsampler(in_channels=3, intermediate_channels=256, depthwise=True)


def test_when_not_depthwise_accepts_any_intermediate() -> None:
    """Non-depthwise conv accepts intermediate_channels not divisible by in_channels."""
    op = LearnedSpatialTemporalDownsampler(
        in_channels=3, intermediate_channels=256, depthwise=False
    )
    x = torch.randn(1, 3, 8, 32, 32)
    with torch.no_grad():
        out = op(x)
    assert out.shape == (1, 15, 4, 64, 64)


# ── Output shape / channel layout ────────────────────────────────────────────


def test_when_default_config_then_output_channels_match_formula() -> None:
    """out_channels = in_channels*out_factor (+ in_channels if concat_original)."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3, out_factor=4, concat_original=True)
    x = torch.randn(1, 3, 8, 32, 32)
    with torch.no_grad():
        out = op(x)
    expected_c = 3 * 4 + 3  # base + original concat
    assert out.shape == (1, expected_c, 4, 64, 64)


def test_when_concat_original_false_then_no_extra_channels() -> None:
    """Without concat, output channels = in_channels * out_factor."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3, out_factor=2, concat_original=False)
    x = torch.randn(1, 3, 8, 32, 32)
    with torch.no_grad():
        out = op(x)
    assert out.shape == (1, 3 * 2, 4, 64, 64)


def test_when_temporal_reduction_factor_then_temporal_dim_halved() -> None:
    """target_t = T // temporal_reduction_factor (min 1)."""
    op = LearnedSpatialTemporalDownsampler(
        in_channels=3, temporal_reduction_factor=2, concat_original=False
    )
    x = torch.randn(1, 3, 10, 32, 32)
    with torch.no_grad():
        out = op(x)
    assert out.shape[2] == 10 // 2


def test_when_odd_temporal_then_target_is_floor() -> None:
    """T=7, factor=2 → target_t = 7//2 = 3 (floor, per code)."""
    op = LearnedSpatialTemporalDownsampler(
        in_channels=3, temporal_reduction_factor=2, concat_original=False
    )
    x = torch.randn(1, 3, 7, 32, 32)
    with torch.no_grad():
        out = op(x)
    assert out.shape[2] == 7 // 2


def test_when_int_target_size_then_normalized_to_square() -> None:
    """Integer target_size produces (s, s) spatial output."""
    op = LearnedSpatialTemporalDownsampler(
        in_channels=3, target_size=32, out_factor=2, concat_original=False
    )
    x = torch.randn(1, 3, 4, 32, 32)
    with torch.no_grad():
        out = op(x)
    assert out.shape[3:] == (32, 32)


@pytest.mark.parametrize("out_factor", [1, 2, 4, 8])
def test_when_scaled_out_factor_then_channels_scale_linearly(out_factor: int) -> None:
    """Output channels (without concat) scale by out_factor."""
    op = LearnedSpatialTemporalDownsampler(
        in_channels=3, out_factor=out_factor, concat_original=False
    )
    x = torch.randn(1, 3, 4, 16, 16)
    with torch.no_grad():
        out = op(x)
    assert out.shape[1] == 3 * out_factor


def test_when_batch_gt_one_then_preserved() -> None:
    """Batch dimension preserved through forward."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3, concat_original=False)
    x = torch.randn(4, 3, 8, 32, 32)
    with torch.no_grad():
        out = op(x)
    assert out.shape[0] == 4


# ── Numerical behavior ───────────────────────────────────────────────────────


def test_output_is_finite() -> None:
    """No NaN/Inf in output for standard float input."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3)
    x = torch.randn(1, 3, 8, 32, 32)
    with torch.no_grad():
        out = op(x)
    assert torch.isfinite(out).all()


def test_zero_input_yields_finite_output() -> None:
    """Zero input must not produce NaN through GroupNorm (single group)."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3, concat_original=False)
    x = torch.zeros(1, 3, 8, 32, 32)
    with torch.no_grad():
        out = op(x)
    assert torch.isfinite(out).all()


def test_when_correction_zeroed_then_output_matches_downsample_path() -> None:
    """Correction branch zeroed, concat=False, out_factor=1:
    out = GroupNorm(ones, zeros)(tiled_x_down). With norm biased to identity,
    output equals the spatial+temporal downsampled input path."""
    op = LearnedSpatialTemporalDownsampler(
        in_channels=3, out_factor=1, concat_original=False, intermediate_channels=15
    )
    with torch.no_grad():
        torch.nn.init.zeros_(op.correction_conv.weight)
        assert op.correction_conv.bias is not None
        torch.nn.init.zeros_(op.correction_conv.bias)
        torch.nn.init.zeros_(op.out_correction_conv.weight)
        assert op.out_correction_conv.bias is not None
        torch.nn.init.zeros_(op.out_correction_conv.bias)
        torch.nn.init.ones_(op.norm.weight)
        torch.nn.init.zeros_(op.norm.bias)

        x = torch.randn(1, 3, 8, 32, 32)
        out = op(x)

        # Replicate the downsampled path the module uses internally.
        xd = einops.rearrange(x, "b c t h w -> b t c h w")
        xd = op.downsample_x_down(xd)
        xd = einops.rearrange(xd, "b t c h w -> b c t h w")
        target_t = max(1, xd.shape[2] // op.temporal_reduction_factor)
        xd = uniform_temporal_subsample(xd, num_samples=target_t, temporal_dim=-3)
        tiled = xd.repeat_interleave(op.out_factor, dim=1)
        expected = op.norm(tiled)

    torch.testing.assert_close(out, expected)


# ── uniform_temporal_subsample (in-house replacement for pytorchvideo) ────────


def test_temporal_subsample_reduces_length() -> None:
    x = torch.arange(10, dtype=torch.float).reshape(1, 1, 10, 1, 1)
    out = uniform_temporal_subsample(x, num_samples=5)
    assert out.shape[2] == 5


def test_temporal_subsample_preserves_endpoints() -> None:
    """linspace(0, t-1, n).long() always includes first and last index."""
    x = torch.arange(8, dtype=torch.float).reshape(1, 1, 8, 1, 1)
    out = uniform_temporal_subsample(x, num_samples=4)
    vals = out[0, 0, :, 0, 0].tolist()
    assert vals[0] == 0.0 and vals[-1] == 7.0


def test_temporal_subsample_no_reorder() -> None:
    """Equispaced selection must preserve temporal ordering."""
    x = torch.arange(10, dtype=torch.float).reshape(1, 1, 10, 1, 1)
    out = uniform_temporal_subsample(x, num_samples=5)
    vals = out[0, 0, :, 0, 0].tolist()
    assert vals == sorted(vals)


def test_temporal_subsample_clamps_oversample() -> None:
    """num_samples > t → nearest-neighbour clamp, no error."""
    x = torch.arange(3, dtype=torch.float).reshape(1, 1, 3, 1, 1)
    out = uniform_temporal_subsample(x, num_samples=5)
    assert out.shape[2] == 5
    assert torch.isfinite(out).all()


def test_temporal_subsample_raises_on_invalid() -> None:
    x = torch.zeros(1, 1, 4, 1, 1)
    with pytest.raises((AssertionError, ValueError)):
        uniform_temporal_subsample(x, num_samples=0)
