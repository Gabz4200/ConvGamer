"""Behavior tests for LearnedSpatialTemporalDownsampler.

Behavioral contracts covered:

  - LearnedSpatialTemporalDownsampler: forward rejects non-5D input and channel
    mismatches, maps (B, C, T, H, W) -> (B, C*out_factor + (C if concat) ?,
    T, target_h, target_w), concat flags, finite outputs, residual algebra
    sanity. T is preserved (only spatial downsampling).
  - uniform_temporal_subsample: strided causal prefix, ordering, clamp,
    invalid counts.
"""

import pytest
import torch

from convgamer.models.convgamer.blocks import (
    LearnedSpatialTemporalDownsampler,
    spatial_softmax,
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


def test_when_non_positive_multiple_then_raises() -> None:
    """Correction width stays a positive multiple of inputs by construction."""
    with pytest.raises(ValueError, match="channel_multiple"):
        LearnedSpatialTemporalDownsampler(in_channels=3, channel_multiple=0)


def test_when_custom_multiple_then_correction_width_scales() -> None:
    """channel_multiple sets the correction width as in_channels * multiple."""
    op = LearnedSpatialTemporalDownsampler(
        in_channels=3,
        channel_multiple=4,
        depthwise=False,
        target_size=(64, 64),
    )
    assert op.intermediate_channels == 12
    x = torch.randn(1, 3, 8, 32, 32)
    with torch.no_grad():
        out = op(x)
    assert out.shape == (1, 6, 8, 64, 64)


def test_correction_branch_processes_full_resolution_before_pooling() -> None:
    """Non-square target resolves from full-res input with T preserved."""
    op = LearnedSpatialTemporalDownsampler(
        in_channels=3,
        target_size=(8, 10),
        concat_original=False,
    )
    with torch.no_grad():
        out = op(torch.randn(1, 3, 12, 64, 64))
    assert out.shape == (1, 6, 12, 8, 10)


# ── Output shape / channel layout ────────────────────────────────────────────


def test_when_default_config_then_output_channels_match_formula() -> None:
    """out_channels = in_channels*out_factor (+ in_channels if concat_original)."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3, out_factor=4, concat_original=True)
    x = torch.randn(1, 3, 8, 64, 64)
    with torch.no_grad():
        out = op(x)
    expected_c = 3 * 4 + 3  # base + original concat
    assert out.shape == (1, expected_c, 8, 64, 64)


def test_when_concat_original_false_then_no_extra_channels() -> None:
    """Without concat, output channels = in_channels * out_factor."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3, out_factor=2, concat_original=False)
    x = torch.randn(1, 3, 8, 64, 64)
    with torch.no_grad():
        out = op(x)
    assert out.shape == (1, 3 * 2, 8, 64, 64)


def test_when_temporal_preserving_then_temporal_dim_unchanged() -> None:
    """T dimension is preserved: temporal subsampling removed, only spatial downsample."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3, concat_original=False)
    x = torch.randn(1, 3, 10, 32, 32)
    with torch.no_grad():
        out = op(x)
    assert out.shape[2] == 10  # T preserved


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
        in_channels=3,
        out_factor=out_factor,
        concat_original=False,
        target_size=(16, 16),
    )
    x = torch.randn(1, 3, 4, 16, 16)
    with torch.no_grad():
        out = op(x)
    assert out.shape[1] == 3 * out_factor


# ── Numerical behavior ───


def test_output_finite_including_zero_input() -> None:
    """No NaN/Inf for standard float or zero input (LayerNorm-safe)."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3)
    x = torch.randn(1, 3, 8, 32, 32)
    with torch.no_grad():
        assert torch.isfinite(op(x)).all()
        assert torch.isfinite(op(torch.zeros_like(x))).all()


def test_when_correction_zeroed_then_output_matches_tiled_input_path() -> None:
    """Zeroed correction with identity norm: output equals tiled downsampled input."""
    op = LearnedSpatialTemporalDownsampler(
        in_channels=3,
        out_factor=1,
        concat_original=False,
        channel_multiple=5,
        target_size=(32, 32),
    )
    with torch.no_grad():
        for p in list(op.correction_conv.parameters()) + list(op.out_correction_conv.parameters()):
            torch.nn.init.zeros_(p)
        torch.nn.init.ones_(op.norm.weight)
        torch.nn.init.zeros_(op.norm.bias)

        x = torch.randn(1, 3, 8, 32, 32)
        out = op(x)
        identity = LearnedSpatialTemporalDownsampler(
            in_channels=3,
            out_factor=1,
            concat_original=False,
            channel_multiple=1,
            target_size=(32, 32),
            depthwise=False,
        )
        with torch.no_grad():
            for p in list(identity.correction_conv.parameters()) + list(
                identity.out_correction_conv.parameters()
            ):
                torch.nn.init.zeros_(p)
            torch.nn.init.ones_(identity.norm.weight)
            torch.nn.init.zeros_(identity.norm.bias)
            expected = identity(x)

    torch.testing.assert_close(out, expected)


def test_when_small_input_then_never_upsampled() -> None:
    """Relative sizing caps at input resolution: 16px stays 16px."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3)
    x = torch.randn(1, 3, 8, 16, 16)
    with torch.no_grad():
        out = op(x)
    assert out.shape == (1, 6, 8, 16, 16)


def test_when_large_input_then_capped_at_max_spatial() -> None:
    """224px input downsamples to the 64px cap."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3)
    x = torch.randn(1, 3, 8, 224, 224)
    with torch.no_grad():
        out = op(x)
    assert out.shape == (1, 6, 8, 64, 64)


# ── uniform_temporal_subsample (in-house replacement for pytorchvideo) ────────


def test_temporal_subsample_reduces_length() -> None:
    x = torch.arange(10, dtype=torch.float).reshape(1, 1, 10, 1, 1)
    out = uniform_temporal_subsample(x, num_samples=5)
    assert out.shape[2] == 5


def test_temporal_subsample_preserves_prefix_causality() -> None:
    """Strided subsample: output i depends only on input frames <= i*step."""
    x = torch.arange(8, dtype=torch.float).reshape(1, 1, 8, 1, 1)
    out = uniform_temporal_subsample(x, num_samples=4)
    vals = out[0, 0, :, 0, 0].tolist()
    assert vals == [0.0, 2.0, 4.0, 6.0]


def test_temporal_subsample_clamps_oversample() -> None:
    """num_samples > t → step=1, outputs a frame-count prefix of length t."""
    x = torch.arange(3, dtype=torch.float).reshape(1, 1, 3, 1, 1)
    out = uniform_temporal_subsample(x, num_samples=5)
    assert out.shape[2] == 3
    assert torch.isfinite(out).all()


def test_temporal_subsample_raises_on_invalid() -> None:
    x = torch.zeros(1, 1, 4, 1, 1)
    with pytest.raises((AssertionError, ValueError)):
        uniform_temporal_subsample(x, num_samples=0)


# ── spatial_softmax ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (2, 4, 3, 4, 5)])
def test_spatial_softmax_normalizes_each_channel_spatially(shape: tuple[int, ...]) -> None:
    x = torch.randn(shape)
    out = spatial_softmax(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out.sum(dim=(-2, -1)), torch.ones(shape[:-2]))


def test_spatial_softmax_does_not_mix_channels() -> None:
    x = torch.zeros(1, 2, 2, 2)
    x[:, 0, 0, 0] = 2
    out = spatial_softmax(x)
    torch.testing.assert_close(out[0, 0].sum(), torch.tensor(1.0))
    torch.testing.assert_close(out[0, 1], torch.full((2, 2), 0.25))


def test_spatial_softmax_rejects_unsupported_rank() -> None:
    with pytest.raises(ValueError, match="Expected 4D"):
        spatial_softmax(torch.randn(2, 3, 4))


def test_spatial_softmax_preserves_input_dtype() -> None:
    out = spatial_softmax(torch.randn(2, 3, 4, 5, dtype=torch.float16))
    assert out.dtype == torch.float16
