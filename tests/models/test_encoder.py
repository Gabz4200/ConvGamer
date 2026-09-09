"""Behavior tests for the InceptionNeXt Encoder (§3.3).

Tests verify the *behavioral contracts* from the paper:
  - Stem downsamples 4x with Conv2d(4, stride=4) + LayerNorm.
  - 4 stages with channel doubling (C, 2C, 4C, 8C).
  - Stage transition uses 1x1 conv (no pooling/strided conv).
  - MLP ratio 4 for stages 1-3, 3 for stage 4.
  - Global average pooling before classifier.
  - Residual connections throughout.
  - Configurable per-stage layer counts.
"""

import torch
import torch.nn as nn

# Encoder is auto-registered via convgamer.models.__init__ -> inception_next.encoder
from convgamer.models.registry import get_model

# ── Shape / architecture ─────────────────────────────────────────────────────


def test_encoder_produces_logits():
    net = get_model("encoder", input_dim=3, hidden_dim=96, num_layers=1, num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    out = net(x)
    assert out.shape == (2, 10)


def test_stem_downsamples_4x():
    """Paper Table 3: stem is Conv2d(3, C, kernel=4, stride=4)."""
    net = get_model("encoder", input_dim=3, hidden_dim=96, num_layers=1, num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    stem_out = net.stem(x)
    assert stem_out.shape == (1, 96, 56, 56)  # 224/4 = 56


def test_stem_has_layernorm():
    """Paper: LayerNorm after stem conv (ablation §4.3 uses LayerNorm)."""
    net = get_model("encoder", input_dim=3, hidden_dim=96, num_classes=10)
    assert isinstance(net.stem_norm, nn.LayerNorm)
    assert net.stem_norm.normalized_shape == (96,)


def test_four_stages_with_channel_doubling():
    """Paper Table 3: channels double per stage (96, 192, 384, 768)."""
    net = get_model("encoder", input_dim=3, hidden_dim=96, num_layers=1, num_classes=10)

    def first_block(stage):
        return stage[0] if not isinstance(stage[0], nn.Conv2d) else stage[1]

    assert first_block(net.stages[0]).norm.normalized_shape == (96,)
    assert first_block(net.stages[1]).norm.normalized_shape == (192,)
    assert first_block(net.stages[2]).norm.normalized_shape == (384,)
    assert first_block(net.stages[3]).norm.normalized_shape == (768,)


def test_stage_transitions_use_1x1_conv():
    """Paper: stage downsampling is 1x1 Conv2d (not pooling/strided)."""
    net = get_model("encoder", input_dim=3, hidden_dim=96, num_layers=1, num_classes=10)
    for stage_idx in range(1, 4):
        stage = net.stages[stage_idx]
        assert isinstance(stage[0], nn.Conv2d)
        assert stage[0].kernel_size == (1, 1)


def test_stage0_has_no_downsample():
    """Paper: first stage has no transition (stem already downsampled 4x)."""
    net = get_model("encoder", input_dim=3, hidden_dim=96, num_layers=1, num_classes=10)
    assert not isinstance(net.stages[0][0], nn.Conv2d)


# ── MLP ratio ─────────────────────────────────────────────────────────────────


def test_mlp_ratio_stage4_is_3():
    """Paper §3.3: MLP ratio is 3 in stage 4."""
    net = get_model("encoder", input_dim=3, hidden_dim=96, num_layers=1, num_classes=10)
    # Stage 3 (index 3): 768 channels, ratio 3 -> 2304
    stage3 = net.stages[3]
    block = stage3[1] if isinstance(stage3[0], nn.Conv2d) else stage3[0]
    assert block.mlp[0].out_features == 768 * 3
    assert block.mlp[2].out_features == 768


def test_mlp_ratio_stages_1_to_3_is_4():
    """Paper §3.3: MLP ratio is 4 in stages 1-3."""
    net = get_model("encoder", input_dim=3, hidden_dim=96, num_layers=1, num_classes=10)
    assert net.stages[0][0].mlp[0].out_features == 96 * 4


def test_custom_mlp_ratios():
    """User can override mlp_ratios per stage."""
    net = get_model(
        "encoder", input_dim=3, hidden_dim=32, num_layers=1, num_classes=10, mlp_ratios=(2, 3, 4, 2)
    )
    assert net.stages[0][0].mlp[0].out_features == 32 * 2
    stage3 = net.stages[3]
    block = stage3[1] if isinstance(stage3[0], nn.Conv2d) else stage3[0]
    assert block.mlp[0].out_features == 256 * 2


# ── Per-stage layer counts ────────────────────────────────────────────────────


def test_per_stage_layer_counts_int():
    """Int num_layers broadcasts to all stages."""
    net = get_model("encoder", input_dim=3, hidden_dim=32, num_layers=2, num_classes=10)

    def block_count(stage):
        return len(stage) - 1 if isinstance(stage[0], nn.Conv2d) else len(stage)

    counts = [block_count(s) for s in net.stages]
    assert counts == [2, 2, 2, 2]


def test_per_stage_layer_counts_tuple():
    """Tuple num_layers gives per-stage counts (paper §3.3: [3,3,9,3])."""
    net = get_model("encoder", input_dim=3, hidden_dim=32, num_layers=(1, 1, 2, 1), num_classes=10)

    def block_count(stage):
        return len(stage) - 1 if isinstance(stage[0], nn.Conv2d) else len(stage)

    counts = [block_count(s) for s in net.stages]
    assert counts == [1, 1, 2, 1]


# ── Pooling + head ────────────────────────────────────────────────────────────


def test_global_average_pooling_before_head():
    """Paper: global average pool -> Linear head."""
    net = get_model("encoder", input_dim=3, hidden_dim=96, num_layers=1, num_classes=1000)
    # Head input should be hidden_dim * 8 = 768
    assert net.head.in_features == 768
    assert net.head.out_features == 1000


def test_no_head_for_feature_extractor():
    """num_classes=0 returns features (mean-pooled) with no classification head."""
    net = get_model("encoder", input_dim=3, hidden_dim=96, num_layers=1, num_classes=0)
    assert isinstance(net.head, nn.Identity)
    x = torch.randn(1, 3, 32, 32)
    out = net(x)
    assert out.shape == (1, 768)


# ── Residual connections ──────────────────────────────────────────────────────


def test_forward_features_matches_forward():
    """forward_features should return same pre-head features as forward."""
    net = get_model("encoder", input_dim=3, hidden_dim=96, num_layers=1, num_classes=0)
    x = torch.randn(1, 3, 224, 224)
    feats = net.forward_features(x)
    direct = net(x)
    torch.testing.assert_close(feats, direct)


def test_layer_scale_gamma_present():
    """Paper: LayerScale with init 1e-6."""
    net = get_model("encoder", input_dim=3, hidden_dim=96, num_layers=1, num_classes=10)
    block = net.stages[0][0]
    assert block.gamma is not None
    torch.testing.assert_close(block.gamma, torch.full((96,), 1e-6))
