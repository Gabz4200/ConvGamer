"""Behavior tests for ConvGamerStem multi-scale 3D stem.

Contracts:
- (B, C, T, H, W) -> (B, 4*C, T, H, W) same resolution, 4x channel expansion
- default in_channels=24 -> out 96, custom values preserved
- spatial/temporal dims unchanged across sizes
- output finite, normed, gradient flows
"""

import pytest
import torch

from convgamer.models.convgamer.stem import ConvGamerStem


@pytest.mark.parametrize("in_channels", [3, 8, 16, 24, 32])
def test_channel_expansion_is_fourfold(in_channels: int) -> None:
    m = ConvGamerStem(in_channels=in_channels, use_softmax=False, use_norm=True)
    x = torch.randn(1, in_channels, 4, 16, 16)
    y = m(x)
    assert y.shape[1] == in_channels * 4
    assert m.out_channels == in_channels * 4
    assert m.norm.num_channels == m.out_channels


def test_softmax_concatenation_doubles_channels() -> None:
    m = ConvGamerStem(in_channels=8, use_softmax=True, use_norm=True)
    x = torch.randn(1, 8, 4, 16, 16)
    y = m(x)
    assert y.shape[1] == 8 * 8
    assert m.out_channels == 8 * 8
    assert m.norm.num_channels == m.out_channels


def test_norm_is_identity_by_default() -> None:
    assert isinstance(ConvGamerStem(in_channels=8).norm, torch.nn.Identity)


def test_batch_and_resolution_preserved() -> None:
    m = ConvGamerStem(in_channels=16)
    x = torch.randn(4, 16, 4, 8, 8)
    y = m(x)
    assert y.shape == (4, 64, 4, 8, 8)


def test_output_finite_including_zero_input() -> None:
    m = ConvGamerStem(in_channels=24)
    assert torch.isfinite(m(torch.randn(2, 24, 4, 16, 16))).all()
    assert torch.isfinite(m(torch.zeros(2, 24, 4, 16, 16))).all()


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
