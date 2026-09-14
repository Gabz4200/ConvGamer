"""Behavior tests for LearnedSpatialTemporalDownsampler.

Behavioral contracts covered:

  - LearnedSpatialTemporalDownsampler: forward rejects non-5D input and channel
    mismatches, maps (B, C, T, H, W) -> (B, C*out_factor + (C if concat) ?,
    T // temporal_reduction_factor, target_h, target_w), concat flags, finite
    outputs, residual algebra sanity.
  - uniform_temporal_subsample: strided causal prefix, ordering, clamp,
    invalid counts.
"""

import einops
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


def test_when_depthwise_and_intermediate_not_divisible_then_raises() -> None:
    """Depthwise conv requires intermediate_channels % in_channels == 0."""
    with pytest.raises(ValueError, match="must be divisible"):
        LearnedSpatialTemporalDownsampler(in_channels=3, intermediate_channels=256, depthwise=True)


def test_when_not_depthwise_accepts_any_intermediate() -> None:
    """Non-depthwise conv accepts intermediate_channels not divisible by in_channels."""
    op = LearnedSpatialTemporalDownsampler(
        in_channels=3,
        intermediate_channels=256,
        depthwise=False,
        target_size=(64, 64),
    )
    x = torch.randn(1, 3, 8, 32, 32)
    with torch.no_grad():
        out = op(x)
    assert out.shape == (1, 6, 4, 64, 64)


def test_correction_branch_processes_full_resolution_before_pooling() -> None:
    op = LearnedSpatialTemporalDownsampler(
        in_channels=3,
        target_size=(8, 10),
        temporal_reduction_factor=2,
        concat_original=False,
    )
    seen: list[tuple[int, ...]] = []
    op.correction_conv.register_forward_hook(
        lambda _, inputs, __: seen.append(tuple(inputs[0].shape))
    )

    with torch.no_grad():
        op(torch.randn(1, 3, 12, 64, 64))

    assert seen == [(1, 3, 12, 64, 64)]


# ── Output shape / channel layout ────────────────────────────────────────────


def test_when_default_config_then_output_channels_match_formula() -> None:
    """out_channels = in_channels*out_factor (+ in_channels if concat_original)."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3, out_factor=4, concat_original=True)
    x = torch.randn(1, 3, 8, 64, 64)
    with torch.no_grad():
        out = op(x)
    expected_c = 3 * 4 + 3  # base + original concat
    assert out.shape == (1, expected_c, 4, 64, 64)


def test_when_concat_original_false_then_no_extra_channels() -> None:
    """Without concat, output channels = in_channels * out_factor."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3, out_factor=2, concat_original=False)
    x = torch.randn(1, 3, 8, 64, 64)
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


def test_when_temporal_reduction_factor_non_positive_then_raises() -> None:
    with pytest.raises(ValueError, match="temporal_reduction_factor"):
        LearnedSpatialTemporalDownsampler(temporal_reduction_factor=0)


def test_downsampler_correction_is_causal_after_temporal_reduction() -> None:
    op = LearnedSpatialTemporalDownsampler(
        in_channels=1,
        intermediate_channels=1,
        out_factor=1,
        target_size=(2, 2),
        temporal_reduction_factor=2,
        concat_original=False,
    )
    x = torch.zeros(1, 1, 4, 4, 4)
    with torch.no_grad():
        baseline = op(x)
        x[:, :, 1] = 1
        changed = op(x)

    torch.testing.assert_close(changed[:, :, 0], baseline[:, :, 0])


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
        in_channels=3,
        out_factor=out_factor,
        concat_original=False,
        target_size=(16, 16),
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
    """Zero input must not produce NaN through LayerNorm."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3, concat_original=False)
    x = torch.zeros(1, 3, 8, 32, 32)
    with torch.no_grad():
        out = op(x)
    assert torch.isfinite(out).all()


def test_when_correction_zeroed_then_output_matches_downsample_path() -> None:
    """Correction branch zeroed, concat=False, out_factor=1:
    out = LayerNorm(ones, zeros)(tiled_x_down). With norm biased to identity,
    output equals the spatial+temporal downsampled input path."""
    op = LearnedSpatialTemporalDownsampler(
        in_channels=3,
        out_factor=1,
        concat_original=False,
        intermediate_channels=15,
        target_size=(32, 32),
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
        xd = op._resize_spatial(xd, (32, 32))
        xd = einops.rearrange(xd, "b t c h w -> b c t h w")
        target_t = max(1, xd.shape[2] // op.temporal_reduction_factor)
        xd = uniform_temporal_subsample(xd, num_samples=target_t, temporal_dim=-3)
        tiled = xd.repeat_interleave(op.out_factor, dim=1)
        expected = op.norm(tiled)

    torch.testing.assert_close(out, expected)


def test_when_small_input_then_never_upsampled() -> None:
    """Relative sizing caps at input resolution: 16px stays 16px."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3)
    x = torch.randn(1, 3, 8, 16, 16)
    with torch.no_grad():
        out = op(x)
    assert out.shape == (1, 6, 4, 16, 16)


def test_when_large_input_then_capped_at_max_spatial() -> None:
    """224px input downsamples to the 64px cap."""
    op = LearnedSpatialTemporalDownsampler(in_channels=3)
    x = torch.randn(1, 3, 8, 224, 224)
    with torch.no_grad():
        out = op(x)
    assert out.shape == (1, 6, 4, 64, 64)


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


def test_temporal_subsample_no_reorder() -> None:
    """Equispaced selection must preserve temporal ordering."""
    x = torch.arange(10, dtype=torch.float).reshape(1, 1, 10, 1, 1)
    out = uniform_temporal_subsample(x, num_samples=5)
    vals = out[0, 0, :, 0, 0].tolist()
    assert vals == sorted(vals)


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
