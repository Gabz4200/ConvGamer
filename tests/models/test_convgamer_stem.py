"""Behavior tests for ConvGamerStem multi-scale 3D stem.

Contracts:
- (B, C, T, H, W) -> (B, 4*C, T, H, W) same resolution, 4x channel expansion
- default in_channels=24 -> out 96, custom values preserved
- spatial/temporal dims unchanged across sizes
- output finite, normed, gradient flows
- alias new_f_conv <-> fuse_conv and old checkpoint compat
"""

import pytest
import torch

from convgamer.models.convgamer.blocks import ConvGamerStem


def test_default_expands_24_to_96_preserves_resolution() -> None:
    m = ConvGamerStem(in_channels=24)
    x = torch.randn(2, 24, 8, 64, 64)
    y = m(x)
    assert y.shape == torch.Size([2, 96, 8, 64, 64])
    assert m.out_channels == 96
    assert m.in_channels == 24


@pytest.mark.parametrize("in_channels", [3, 8, 16, 24, 32])
def test_channel_expansion_is_fourfold(in_channels: int) -> None:
    m = ConvGamerStem(in_channels=in_channels)
    x = torch.randn(1, in_channels, 4, 16, 16)
    y = m(x)
    assert y.shape[1] == in_channels * 4
    assert m.out_channels == in_channels * 4
    assert m.norm.num_channels == m.out_channels


@pytest.mark.parametrize(
    "t,h,w",
    [(2, 8, 8), (4, 16, 16), (8, 32, 32), (7, 7, 7)],
)
def test_resolution_preserved_across_sizes(t: int, h: int, w: int) -> None:
    m = ConvGamerStem(in_channels=8)
    x = torch.randn(1, 8, t, h, w)
    y = m(x)
    assert y.shape[2] == t
    assert y.shape[3] == h
    assert y.shape[4] == w


def test_batch_dimension_preserved() -> None:
    m = ConvGamerStem(in_channels=16)
    x = torch.randn(4, 16, 4, 8, 8)
    y = m(x)
    assert y.shape[0] == 4


def test_output_finite() -> None:
    m = ConvGamerStem(in_channels=24)
    x = torch.randn(2, 24, 4, 16, 16)
    y = m(x)
    assert torch.isfinite(y).all()


def test_zero_input_finite_through_norm() -> None:
    m = ConvGamerStem(in_channels=8)
    x = torch.zeros(1, 8, 4, 8, 8)
    y = m(x)
    assert torch.isfinite(y).all()


def test_gradient_flows_to_input_and_params() -> None:
    m = ConvGamerStem(in_channels=8)
    x = torch.randn(2, 8, 4, 8, 8, requires_grad=True)
    y = m(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for n, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {n}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {n}"


def test_alias_new_f_conv_is_fuse_conv() -> None:
    m = ConvGamerStem(in_channels=12)
    assert hasattr(m, "fuse_conv")
    assert m.new_f_conv is m.fuse_conv
    # mutating via alias mutates fuse_conv
    new = torch.nn.Conv3d(12 * 6, 12 * 3, kernel_size=1)
    m.new_f_conv = new
    assert m.fuse_conv is new
    assert m.new_f_conv is new
    assert "fuse_conv.weight" in m.state_dict()
    assert "new_f_conv.weight" not in m.state_dict()


def test_old_checkpoint_with_new_f_conv_keys_loads() -> None:
    old = ConvGamerStem(in_channels=16)
    sd = old.state_dict()
    legacy = {}
    for k, v in sd.items():
        if k.startswith("fuse_conv."):
            legacy[k.replace("fuse_conv.", "new_f_conv.")] = v
        else:
            legacy[k] = v
    assert "new_f_conv.weight" in legacy
    new_model = ConvGamerStem(in_channels=16)
    new_model.load_state_dict(legacy)
    torch.testing.assert_close(new_model.fuse_conv.weight, old.fuse_conv.weight)
    torch.testing.assert_close(new_model.fuse_conv.bias, old.fuse_conv.bias)
    # forward after legacy load stays correct shape
    x = torch.randn(1, 16, 4, 8, 8)
    y = new_model(x)
    assert y.shape == torch.Size([1, 64, 4, 8, 8])


def test_state_dict_roundtrip_preserves_output() -> None:
    m = ConvGamerStem(in_channels=8)
    x = torch.randn(1, 8, 4, 8, 8)
    with torch.no_grad():
        y_before = m(x)
    sd = m.state_dict()
    m2 = ConvGamerStem(in_channels=8)
    m2.load_state_dict(sd)
    with torch.no_grad():
        y_after = m2(x)
    torch.testing.assert_close(y_before, y_after)
