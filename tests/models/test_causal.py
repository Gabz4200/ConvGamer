"""Behavior tests for causal primitives and end-to-end causality."""

import pytest
import torch
import torch.nn as nn

from convgamer.models.convgamer.causal import (
    CausalConv3d,
    CausalLayerNorm,
    CausalTemporalMixer,
)
from convgamer.models.convgamer.downsampler import LearnedSpatialTemporalDownsampler
from convgamer.models.convgamer.stem import ConvGamerStem


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


@pytest.mark.parametrize(
    "kernel_size,in_ch,out_ch,shape",
    [
        (1, 2, 2, (2, 2, 4, 8, 8)),
        (2, 2, 2, (1, 2, 3, 5, 5)),
        (3, 2, 2, (2, 2, 4, 8, 8)),
        (7, 2, 2, (2, 2, 4, 8, 8)),
        ((7, 7, 2), 3, 6, (1, 3, 4, 8, 8)),
    ],
)
def test_causal_conv_preserves_shape(kernel_size, in_ch, out_ch, shape) -> None:
    """Causal padding preserves B/T/H/W for odd, even, and tuple kernels."""
    m = CausalConv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size)
    x = torch.randn(*shape)
    out = m(x)
    assert out.shape[:1] + out.shape[2:] == x.shape[:1] + x.shape[2:]
    assert out.shape[1] == out_ch
    assert torch.isfinite(out).all()


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


# ── CausalLayerNorm ─────────────────────────────────────────────────────────


def test_causal_layernorm_preserves_shape_5d() -> None:
    m = CausalLayerNorm(4)
    x = torch.randn(2, 4, 3, 4, 4)
    assert m(x).shape == x.shape


def test_causal_layernorm_matches_per_location_layernorm() -> None:
    torch.manual_seed(0)
    c = CausalLayerNorm(4)
    x = torch.randn(2, 4, 4, 4)
    ref = nn.LayerNorm(4)(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    torch.testing.assert_close(c(x), ref, rtol=1e-4, atol=1e-5)


def test_causal_layernorm_temporal_causality() -> None:
    m = CausalLayerNorm(4)
    _assert_causal(m, (1, 4, 6, 4, 4), perturb_from=3)


def test_causal_layernorm_spatial_location_does_not_leak() -> None:
    torch.manual_seed(1)
    m = CausalLayerNorm(4)
    x1 = torch.randn(1, 4, 4, 4, 4)
    x2 = x1.clone()
    x2[:, :, 2:, :, :] += 5.0
    m.eval()
    with torch.no_grad():
        y1, y2 = m(x1), m(x2)
    torch.testing.assert_close(y1[:, :, 0], y2[:, :, 0])


def test_causal_layernorm_state_dict_roundtrip() -> None:
    m = CausalLayerNorm(4)
    sd = m.state_dict()
    assert "weight" in sd and "bias" in sd
    m2 = CausalLayerNorm(4)
    m2.load_state_dict(sd)
    x = torch.randn(1, 4, 2, 4, 4)
    torch.testing.assert_close(m(x), m2(x))


def test_causal_layernorm_gradient_flows() -> None:
    m = CausalLayerNorm(4)
    x = torch.randn(1, 4, 2, 3, 3, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert m.weight.grad is not None


# ── Integration: modules are end-to-end causal ─────────────────────────────


def test_temporal_layers_are_stateless() -> None:
    """No per-layer streaming seam: no init_state/step methods, no history buffers."""
    from convgamer.models.convgamer.stem import ConvGamerStem

    modules = [
        CausalConv3d(3, 6, kernel_size=3),
        CausalTemporalMixer(channels=4, dilations=(1, 2)),
        LearnedSpatialTemporalDownsampler(in_channels=3),
        ConvGamerStem(in_channels=4),
    ]
    for m in modules:
        assert not hasattr(m, "init_state"), type(m).__name__
        assert not hasattr(m, "step"), type(m).__name__
        assert not list(m.buffers()), type(m).__name__


def test_stem_is_causal_with_norm() -> None:
    m = ConvGamerStem(in_channels=4, use_norm=True)
    _assert_causal(m, (1, 4, 6, 8, 8), perturb_from=3)


def test_stem_is_causal_without_norm() -> None:
    m = ConvGamerStem(in_channels=4, use_norm=False)
    _assert_causal(m, (1, 4, 6, 8, 8), perturb_from=3)


def test_downsampler_is_causal() -> None:
    m = LearnedSpatialTemporalDownsampler(
        in_channels=3,
        expansion=2,
        out_factor=2,
    )
    _assert_causal(m, (1, 3, 6, 8, 8), perturb_from=3)


def test_causal_temporal_mixer_forward_shape() -> None:
    """forward maps (B, F, T, H, W) -> (B, Hid, T, H, W) with Hid==channels."""
    m = CausalTemporalMixer(channels=4, dilations=(1,))
    m.eval()
    x = torch.randn(2, 4, 5, 8, 8)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (2, 4, 5, 8, 8)
