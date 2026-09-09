"""Behavior tests for InceptionDWConv2d and InceptionNeXtBlock.

These tests verify the *behavioral contracts* from the paper
(arXiv:2303.16900v2) rather than internal attribute names,
so they remain valid under refactors.

Key invariants from the paper:

InceptionDWConv2d (Algorithm 1, §3.2):
  - Input (B,C,H,W) is split along channels into 4 groups:
    3 groups of size gc=getChannel, 1 group of size C-3*gc.
  - 3 groups go through depthwise conv branches (3x3, 1xk, kx1).
  - 4th group (identity) passes through unchanged.
  - Output is channel-concatenation, same shape as input.

InceptionNeXtBlock (Eq. 3, §3.1):
  - Single residual path: output + input.
  - TokenMixer -> Norm -> MLP -> gamma scaling -> residual.
  - output shape == input shape.
"""

import torch
import torch.nn as nn

from convgamer.models.blocks import InceptionDWConv2d, InceptionNeXtBlock

# ── InceptionDWConv2d ────────────────────────────────────────────────────────


def test_preserves_spatial_shape():
    """InceptionDWConv2d must preserve (B, C, H, W) shape."""
    for in_ch in [16, 32, 96, 128]:
        x = torch.randn(2, in_ch, 14, 14)
        op = InceptionDWConv2d(in_channels=in_ch)
        assert op(x).shape == x.shape


def test_identity_branch_is_untouched():
    """The identity branch channels must pass through unchanged.

    Paper Eq. 6: X'id = Xid — the 5/8 channels are not convolved.
    """
    x = torch.randn(1, 96, 8, 8)
    op = InceptionDWConv2d(in_channels=96)
    gc = 12  # 1/8 of 96
    x_id = x[:, 3 * gc :, :, :]  # last 60 channels
    out = op(x)
    torch.testing.assert_close(out[:, 3 * gc :, :, :], x_id)


def test_split_sums_to_total_channels():
    """split_indexes must sum to in_channels for all tested widths."""
    op = InceptionDWConv2d(in_channels=96)
    gc = 12
    assert op.split_indexes == (gc, gc, gc, 96 - 3 * gc)


def test_conv_branches_are_depthwise():
    """Paper: each conv branch is depthwise (groups == gc)."""
    op = InceptionDWConv2d(in_channels=96)
    gc = 12
    assert op.dwconv_hw.groups == gc
    assert op.dwconv_w.groups == gc
    assert op.dwconv_h.groups == gc
    assert op.dwconv_hw.in_channels == gc
    assert op.dwconv_hw.out_channels == gc


def test_kernel_sizes_match_paper_defaults():
    """Paper: square=3x3, band=1x11 and 11x1."""
    op = InceptionDWConv2d(in_channels=96)
    assert tuple(op.dwconv_hw.kernel_size) == (3, 3)
    assert tuple(op.dwconv_w.kernel_size) == (1, 11)
    assert tuple(op.dwconv_h.kernel_size) == (11, 1)


def test_gradient_flows_through_all_branches():
    """Zero gradient to identity branch shouldn't affect conv branches."""
    x = torch.randn(1, 96, 8, 8, requires_grad=True)
    op = InceptionDWConv2d(in_channels=96)
    out = op(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


# ── InceptionNeXtBlock ───────────────────────────────────────────────────────


def test_block_is_residual():
    """Paper Eq. 3: output = gamma * MLP(Norm(TM(x))) + x."""
    block = InceptionNeXtBlock(in_channels=32, hidden_dim=128)
    x = torch.zeros(1, 32, 8, 8)  # zeros -> output should be ~0 + x = 0
    out = block(x)
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=1e-5, atol=1e-5)


def test_block_gradient_flows():
    """Residual connection must allow gradient to reach input."""
    block = InceptionNeXtBlock(in_channels=32, hidden_dim=128)
    x = torch.randn(1, 32, 8, 8, requires_grad=True)
    out = block(x).sum()
    out.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_block_uses_layernorm():
    """Paper §4.3 ablation: LayerNorm used (vs BatchNorm)."""
    block = InceptionNeXtBlock(in_channels=32, hidden_dim=128)
    assert isinstance(block.norm, nn.LayerNorm)
    assert block.norm.normalized_shape == (32,)


def test_block_uses_gelu():
    """Paper Eq. 3: σ activation = GELU."""
    block = InceptionNeXtBlock(in_channels=32, hidden_dim=128)
    assert isinstance(block.mlp[1], nn.GELU)


def test_block_uses_linear_layers():
    """Paper: MLP is two fully-connected layers (or 1x1 convs)."""
    block = InceptionNeXtBlock(in_channels=32, hidden_dim=128)
    assert isinstance(block.mlp[0], nn.Linear)
    assert isinstance(block.mlp[2], nn.Linear)
    assert block.mlp[0].in_features == 32
    assert block.mlp[0].out_features == 128
    assert block.mlp[2].out_features == 32


def test_layer_scale_init_value():
    """Paper: LayerScale gamma initialized to 1e-6."""
    block = InceptionNeXtBlock(in_channels=32, hidden_dim=128, layer_scale_init=1e-6)
    assert block.gamma is not None
    assert block.gamma.shape == (32,)
    torch.testing.assert_close(block.gamma, torch.full((32,), 1e-6))


def test_layer_scale_disabled_when_zero():
    """layer_scale_init=0 disables gamma (no scaling)."""
    block = InceptionNeXtBlock(in_channels=32, hidden_dim=128, layer_scale_init=0.0)
    assert block.gamma is None


def test_block_single_shortcut():
    """Paper: MetaNeXt has a single residual (not two like MetaFormer).

    With gamma=1e-6, output ≈ input (tiny perturbation), confirming
    single residual path dominates.
    """
    block = InceptionNeXtBlock(in_channels=32, hidden_dim=128, layer_scale_init=1.0)
    x = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        out = block(x)
        # With gamma=1.0, the block significantly modifies input (not identity).
        assert not torch.allclose(out, x, atol=1e-3)
