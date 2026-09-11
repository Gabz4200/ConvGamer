"""Behavior tests for causal primitives and end-to-end causality."""

import torch
import torch.nn as nn

from convgamer.models.convgamer.blocks import (
    CausalConv3d,
    CausalGroupNorm,
    ConvGamerStem,
    LearnedSpatialTemporalDownsampler,
)


# helper: assert temporal causality – future perturbation does not leak to past
def _assert_causal(module: nn.Module, x_shape: tuple[int, ...], perturb_from: int) -> None:
    torch.manual_seed(0)
    x1 = torch.randn(*x_shape)
    x2 = x1.clone()
    # perturb all temporal slices >= perturb_from
    x2[:, :, perturb_from:, :, :] += 5.0
    module.eval()
    with torch.no_grad():
        y1 = module(x1)
        y2 = module(x2)
    # past slices must be identical
    torch.testing.assert_close(y1[:, :, :perturb_from], y2[:, :, :perturb_from])


# ── CausalConv3d ────────────────────────────────────────────────────────────


def test_causal_conv_preserves_shape_for_odd_kernels() -> None:
    for k in [1, 3, 7]:
        m = CausalConv3d(2, 2, kernel_size=k)
        x = torch.randn(2, 2, 4, 8, 8)
        assert m(x).shape == x.shape


def test_causal_conv_preserves_shape_for_tuple_kernel_772() -> None:
    m = CausalConv3d(3, 6, kernel_size=(7, 7, 2))
    x = torch.randn(1, 3, 4, 8, 8)
    assert m(x).shape == (1, 6, 4, 8, 8)


def test_causal_conv_preserves_shape_even_kernel_2() -> None:
    m = CausalConv3d(2, 2, kernel_size=2)
    x = torch.randn(1, 2, 3, 5, 5)
    assert m(x).shape == x.shape


def test_causal_conv_temporal_causality_kernel_3() -> None:
    m = CausalConv3d(2, 2, kernel_size=3)
    _assert_causal(m, (1, 2, 6, 4, 4), perturb_from=3)


def test_causal_conv_temporal_causality_kernel_7() -> None:
    m = CausalConv3d(2, 2, kernel_size=7)
    _assert_causal(m, (1, 2, 8, 4, 4), perturb_from=4)


def test_causal_conv_temporal_causality_tuple_772() -> None:
    m = CausalConv3d(3, 6, kernel_size=(7, 7, 2))
    _assert_causal(m, (1, 3, 6, 8, 8), perturb_from=3)


def test_causal_conv_causality_holds_with_dilation_2() -> None:
    m = CausalConv3d(2, 2, kernel_size=3, dilation=2)
    # effective kernel = 5, need larger T to see causality boundary
    _assert_causal(m, (1, 2, 8, 4, 4), perturb_from=5)


def test_causal_conv_handles_groups_and_no_bias() -> None:
    m = CausalConv3d(4, 4, kernel_size=3, groups=4, bias=False)
    assert m.bias is None
    x = torch.randn(1, 4, 4, 4, 4)
    y = m(x)
    assert y.shape == x.shape
    _assert_causal(m, (1, 4, 4, 4, 4), perturb_from=2)


def test_causal_conv_exposes_weight_and_bias() -> None:
    m = CausalConv3d(2, 4, kernel_size=3)
    assert m.weight is m.conv.weight
    assert m.bias is m.conv.bias
    assert m.weight.shape == (4, 2, 3, 3, 3)
    # mutating via properties affects inner conv
    with torch.no_grad():
        torch.nn.init.zeros_(m.weight)
        assert torch.all(m.conv.weight == 0)


def test_causal_conv_t1_edge_preserves_shape_and_finite() -> None:
    m = CausalConv3d(2, 2, kernel_size=3)
    x = torch.randn(1, 2, 1, 4, 4)
    y = m(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_causal_conv_gradient_flows() -> None:
    m = CausalConv3d(2, 2, kernel_size=3)
    x = torch.randn(1, 2, 4, 4, 4, requires_grad=True)
    y = m(x)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert m.weight.grad is not None and torch.isfinite(m.weight.grad).all()


# ── CausalGroupNorm ─────────────────────────────────────────────────────────


def test_causal_groupnorm_preserves_shape_5d() -> None:
    m = CausalGroupNorm(num_groups=2, num_channels=4)
    x = torch.randn(2, 4, 3, 4, 4)
    assert m(x).shape == x.shape


def test_causal_groupnorm_4d_fallback_behaves_like_gn() -> None:
    torch.manual_seed(0)
    g = nn.GroupNorm(2, 4)
    c = CausalGroupNorm(2, 4)
    c.load_state_dict(g.state_dict())
    x = torch.randn(2, 4, 4, 4)
    torch.testing.assert_close(c(x), g(x))


def test_causal_groupnorm_temporal_causality() -> None:
    m = CausalGroupNorm(num_groups=2, num_channels=4)
    _assert_causal(m, (1, 4, 6, 4, 4), perturb_from=3)


def test_causal_groupnorm_future_perturb_does_not_affect_past_vs_vanilla_leaks() -> None:
    # vanilla GroupNorm leaks across T; causal does not
    torch.manual_seed(1)
    gn = nn.GroupNorm(2, 4)
    cg = CausalGroupNorm(2, 4)
    cg.load_state_dict(gn.state_dict())
    x1 = torch.randn(1, 4, 4, 4, 4)
    x2 = x1.clone()
    x2[:, :, 2:, :, :] += 5.0
    with torch.no_grad():
        y_gn1, y_gn2 = gn(x1), gn(x2)
        y_cg1, y_cg2 = cg(x1), cg(x2)
    # vanilla leaks: past differs
    assert (y_gn1[:, :, 0] - y_gn2[:, :, 0]).abs().max() > 1e-3
    # causal preserves past
    torch.testing.assert_close(y_cg1[:, :, 0], y_cg2[:, :, 0])


def test_causal_groupnorm_state_dict_roundtrip() -> None:
    m = CausalGroupNorm(2, 4)
    sd = m.state_dict()
    assert "weight" in sd and "bias" in sd
    m2 = CausalGroupNorm(2, 4)
    m2.load_state_dict(sd)
    x = torch.randn(1, 4, 2, 4, 4)
    torch.testing.assert_close(m(x), m2(x))


def test_causal_groupnorm_gradient_flows() -> None:
    m = CausalGroupNorm(2, 4)
    x = torch.randn(1, 4, 2, 3, 3, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert m.weight.grad is not None


# ── Integration: modules are end-to-end causal ─────────────────────────────


def test_stem_is_causal_with_norm() -> None:
    m = ConvGamerStem(in_channels=4, use_softmax=False, use_norm=True)
    _assert_causal(m, (1, 4, 6, 8, 8), perturb_from=3)


def test_stem_is_causal_without_norm() -> None:
    m = ConvGamerStem(in_channels=4, use_softmax=False, use_norm=False)
    _assert_causal(m, (1, 4, 6, 8, 8), perturb_from=3)


def test_downsampler_is_causal() -> None:
    m = LearnedSpatialTemporalDownsampler(
        in_channels=3,
        intermediate_channels=6,
        out_factor=2,
        target_size=(8, 8),
        temporal_reduction_factor=1,
    )
    _assert_causal(m, (1, 3, 6, 8, 8), perturb_from=3)


def test_downsampler_causal_with_temporal_reduction() -> None:
    m = LearnedSpatialTemporalDownsampler(
        in_channels=3,
        intermediate_channels=6,
        out_factor=2,
        target_size=(8, 8),
        temporal_reduction_factor=2,
    )
    # T=6 -> target_t=3, perturb future source frames >=4 should not affect output t=0
    torch.manual_seed(0)
    x1 = torch.randn(1, 3, 6, 16, 16)
    x2 = x1.clone()
    x2[:, :, 4:, :, :] += 5.0
    m.eval()
    with torch.no_grad():
        y1, y2 = m(x1), m(x2)
    torch.testing.assert_close(y1[:, :, 0], y2[:, :, 0])
