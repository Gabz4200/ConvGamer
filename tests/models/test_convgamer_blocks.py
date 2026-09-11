"""Behavior tests for LearnedSpatialTemporalDownsampler and SmoothPWAct.

Behavioral contracts covered:

  - LearnedSpatialTemporalDownsampler: forward rejects non-5D input and channel
    mismatches, maps (B, C, T, H, W) -> (B, C*out_factor + (C if concat) ?,
    T // temporal_reduction_factor, target_h, target_w), concat flags, finite
    outputs, residual algebra sanity.
  - uniform_temporal_subsample: equispaced indices, endpoints, ordering, clamp,
    invalid counts.
  - SmoothPWAct: arbitrary input shape preserved via reshape (not view),
    non-contiguous safe, output bounded as convex combination of y_cords,
    finite, gradient flows to input and params, smooth at knots (squared
    Gaussian kernel), temperature controls sharpness, validation of
    num_anchors and temperature.
"""

from typing import Any, cast

import einops
import pytest
import torch

from convgamer.models.convgamer.blocks import (
    LearnedSpatialTemporalDownsampler,
    PWInterpolationAct,
    SmoothPWAct,
    StrictLearnableGrid,
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
        in_channels=3, intermediate_channels=256, depthwise=False
    )
    x = torch.randn(1, 3, 8, 32, 32)
    with torch.no_grad():
        out = op(x)
    assert out.shape == (1, 27, 4, 64, 64)


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


def test_when_temporal_reduction_factor_non_positive_then_raises() -> None:
    with pytest.raises(ValueError, match="temporal_reduction_factor"):
        LearnedSpatialTemporalDownsampler(temporal_reduction_factor=0)


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


# ── SmoothPWAct ────────────────────────────────────────────────────────────────


def test_smooth_preserves_arbitrary_shapes() -> None:
    """Output shape must equal input shape for arbitrary ranks (reshape, not view)."""
    act = SmoothPWAct(num_anchors=8, temperature=0.5)
    for shape in [(5,), (2, 4), (2, 3, 4), (2, 3, 4, 5)]:
        x = torch.randn(shape)
        assert act(x).shape == x.shape


def test_smooth_non_contiguous_input() -> None:
    """Non-contiguous inputs must not fail (reshape vs view)."""
    act = SmoothPWAct(num_anchors=8, temperature=0.5)
    x = torch.randn(4, 6).t()
    assert not x.is_contiguous()
    out = act(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_smooth_output_bounded_as_convex_combination() -> None:
    """Softmax weights sum to 1, so output is weighted avg of y_cords."""
    act = SmoothPWAct(num_anchors=8, temperature=0.5)
    x = torch.randn(10, 10) * 5
    out = act(x)
    y_min, y_max = act.y_cords.min().item(), act.y_cords.max().item()
    assert (out >= y_min - 1e-6).all() and (out <= y_max + 1e-6).all()


def test_smooth_output_finite() -> None:
    act = SmoothPWAct(num_anchors=16, temperature=0.1)
    x = torch.randn(4, 8) * 3
    assert torch.isfinite(act(x)).all()


def test_smooth_gradient_flows_to_input_and_params() -> None:
    act = SmoothPWAct(num_anchors=4, temperature=0.5)
    x = torch.randn(2, 3, requires_grad=True)
    loss = act(x).sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert not torch.isnan(x.grad).any()
    assert act.x_cords.grad is not None and torch.isfinite(act.x_cords.grad).all()
    assert act.y_cords.grad is not None and torch.isfinite(act.y_cords.grad).all()


def test_smooth_gradient_finite_at_knot() -> None:
    """Squared distance is C1 smooth at x == x_cord (abs would cusp)."""
    act = SmoothPWAct(num_anchors=4, temperature=0.5)
    knot = act.x_cords[1].item()
    x = torch.tensor([[knot]], requires_grad=True)
    out = act(x).sum()
    out.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    eps = 1e-4
    x_lo = torch.tensor([[knot - eps]])
    x_hi = torch.tensor([[knot + eps]])
    assert torch.isfinite(act(x_lo)).all() and torch.isfinite(act(x_hi)).all()
    grad_lo = torch.autograd.grad(act(x_lo).sum(), act.x_cords, retain_graph=True)[0]
    assert torch.isfinite(grad_lo).all()


def test_smooth_temperature_controls_sharpness() -> None:
    """Small temperature -> near nearest-neighbor; large -> blended average."""
    torch.manual_seed(0)
    sharp = SmoothPWAct(num_anchors=8, temperature=1e-3)
    smooth = SmoothPWAct(num_anchors=8, temperature=10.0)
    # copy cords so only temperature differs
    with torch.no_grad():
        smooth.x_cords.copy_(sharp.x_cords)
        smooth.y_cords.copy_(sharp.y_cords)
    # point exactly at a cord should map near that y value when sharp
    cord_x = sharp.x_cords[3].item()
    cord_y = sharp.y_cords[3].item()
    x = torch.tensor([[cord_x]])
    out_sharp = sharp(x).item()
    out_smooth = smooth(x).item()
    assert abs(out_sharp - cord_y) < 0.05
    # smooth is pulled toward mean of y_cords
    y_mean = sharp.y_cords.mean().item()
    assert abs(out_smooth - y_mean) < abs(out_sharp - y_mean)


def test_smooth_invalid_num_anchors_raises() -> None:
    with pytest.raises(ValueError, match="num_anchors"):
        SmoothPWAct(num_anchors=1, temperature=0.5)
    with pytest.raises(ValueError, match="num_anchors"):
        SmoothPWAct(num_anchors=0, temperature=0.5)


def test_smooth_invalid_temperature_raises() -> None:
    with pytest.raises(ValueError, match="temperature"):
        SmoothPWAct(num_anchors=4, temperature=0)
    with pytest.raises(ValueError, match="temperature"):
        SmoothPWAct(num_anchors=4, temperature=-1)


def test_smooth_params_are_learnable() -> None:
    act = SmoothPWAct(num_anchors=6, temperature=0.5)
    assert isinstance(act.x_cords, torch.nn.Parameter)
    assert isinstance(act.y_cords, torch.nn.Parameter)
    assert act.x_cords.requires_grad and act.y_cords.requires_grad
    assert act.x_cords.shape == (6,) and act.y_cords.shape == (6,)


def test_smooth_cords_initialized_linspace() -> None:
    act = SmoothPWAct(num_anchors=4, temperature=0.5)
    torch.testing.assert_close(act.x_cords, torch.linspace(-2, 2, 4))
    torch.testing.assert_close(act.y_cords, torch.linspace(-2, 2, 4))


def test_smooth_preserves_input_dtype() -> None:
    act = SmoothPWAct(num_anchors=4).half()
    out = act(torch.randn(2, 3, dtype=torch.float16))
    assert out.dtype == torch.float16


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


# ── StrictLearnableGrid ──────────────────────────────────────────────────────


def test_grid_has_fixed_endpoints_and_minimum_spacing() -> None:
    grid = StrictLearnableGrid(n_points=5, low=-2, high=3, min_spacing=0.5)
    points = grid()
    torch.testing.assert_close(points[[0, -1]], torch.tensor([-2.0, 3.0]))
    assert torch.all(points[1:] - points[:-1] >= 0.5)
    assert torch.all(points[1:] > points[:-1])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_points": 1, "low": 0, "high": 1, "min_spacing": 0}, "n_points"),
        ({"n_points": 3, "low": 1, "high": 1, "min_spacing": 0}, "high"),
        ({"n_points": 3, "low": 0, "high": 1, "min_spacing": -1}, "non-negative"),
        ({"n_points": 3, "low": 0, "high": 1, "min_spacing": 1}, "too large"),
    ],
)
def test_grid_rejects_invalid_configuration(kwargs: dict[str, float | int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        StrictLearnableGrid(**cast(Any, kwargs))


# ── PWInterpolationAct behavior ─────────────────────────────────────────────


@pytest.mark.parametrize("monotonic", [True, False])
def test_piecewise_clamps_outside_domain_and_hits_endpoints(monotonic: bool) -> None:
    act = PWInterpolationAct(num_anchors=4, monotonic=monotonic)
    x = torch.tensor([[-100.0, -2.0, 2.0, 100.0]])
    out = act(x)
    torch.testing.assert_close(out[0, 0], out[0, 1])
    torch.testing.assert_close(out[0, 2], out[0, 3])


def test_piecewise_rejects_non_matrix_input() -> None:
    out = PWInterpolationAct()(torch.randn(2, 3, 4))
    assert out.shape == (2, 3, 4)


@pytest.mark.parametrize("monotonic", [True, False])
def test_piecewise_preserves_arbitrary_shape_and_dtype(monotonic: bool) -> None:
    act = PWInterpolationAct(num_anchors=4, monotonic=monotonic).half()
    x = torch.randn(2, 3, 4, dtype=torch.float16)
    out = act(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_anchors": 1}, "num_anchors"),
        ({"min_range": 1, "max_range": 1}, "max_range"),
        ({"min_spacing": -1}, "non-negative"),
        ({"num_anchors": 3, "min_spacing": 2.1}, "too large"),
    ],
)
def test_piecewise_rejects_invalid_configuration(
    kwargs: dict[str, float | int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PWInterpolationAct(**cast(Any, kwargs))


# ── PWInterpolationAct ───────────────────────────────────────────────────────


def test_piecewise_monotonic_mode_registers_only_grid_parameters() -> None:
    act = PWInterpolationAct(num_anchors=4, monotonic=True)

    assert act.y_cords_gen is not None
    assert act.y_cords is None
    assert [name for name, _ in act.named_parameters()] == [
        "x_cords_gen.raw",
        "y_cords_gen.raw",
    ]
    assert act(torch.randn(2, 3)).shape == (2, 3)


def test_piecewise_non_monotonic_mode_registers_unconstrained_y_parameter() -> None:
    act = PWInterpolationAct(num_anchors=4, monotonic=False)

    assert act.y_cords_gen is None
    assert isinstance(act.y_cords, torch.nn.Parameter)
    assert {name for name, _ in act.named_parameters()} == {
        "x_cords_gen.raw",
        "y_cords",
    }
    assert act(torch.randn(2, 3)).shape == (2, 3)
