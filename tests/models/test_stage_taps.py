"""Tests for true multi-level (stage-tap) JEPA supervision (V-JEPA 2.1 §2.3.2).

The four encoder levels are distinct feature modalities tapped from the four
InceptionNeXt stages — never copies of the final map.
"""

import torch

from convgamer.models.convgamer.encoder import ConvGamerEncoder
from convgamer.models.inception_next.encoder import InceptionNeXtEncoder


def _enc() -> ConvGamerEncoder:
    return ConvGamerEncoder(input_dim=3, hidden_dim=16, num_layers=1, level_dim=32)


def test_stage_maps_are_four_distinct_levels() -> None:
    """forward_stage_maps returns 4 maps with doubling channels."""
    enc = InceptionNeXtEncoder(input_dim=8, hidden_dim=16, num_layers=1, num_classes=0)
    maps = enc.forward_stage_maps(torch.randn(2, 8, 16, 16))
    assert len(maps) == 4
    assert [m.shape[1] for m in maps] == [16, 32, 64, 128]
    for m in maps:
        assert m.shape[0] == 2 and m.shape[2:] == (4, 4)


def test_forward_pyramid_stacks_four_levels() -> None:
    """forward_feature_maps returns (B, 4, P, T, H, W) with uniform level dim."""
    enc = _enc()
    enc.eval()
    with torch.no_grad():
        pyr = enc.forward_feature_maps(torch.randn(1, 3, 4, 32, 32))
    assert pyr.shape[:3] == (1, 4, 32)
    assert pyr.shape[3] == 4  # T preserved


def test_pyramid_levels_differ() -> None:
    """Levels must not be copies of one another."""
    enc = _enc()
    enc.eval()
    with torch.no_grad():
        pyr = enc.forward_feature_maps(torch.randn(1, 3, 4, 32, 32))
    assert not torch.allclose(pyr[:, 0], pyr[:, 1])
    assert not torch.allclose(pyr[:, 2], pyr[:, 3])


def test_fuse_pyramid_matches_forward_features() -> None:
    """fuse + spatial mean reproduces forward_features exactly."""
    enc = _enc()
    enc.eval()
    video = torch.randn(1, 3, 4, 32, 32)
    with torch.no_grad():
        pyr = enc.forward_feature_maps(video)
        fused = enc.fuse_pyramid(pyr)
        feats = enc.forward_features(video)
    assert fused.shape[:2] == (1, 32)
    assert fused.shape[2] == 4
    torch.testing.assert_close(fused.mean(dim=[-2, -1]), feats)


def test_pyramid_streaming_parity() -> None:
    """step() features match forward_features; state advances per frame."""
    enc = _enc()
    enc.eval()
    video = torch.randn(1, 3, 4, 32, 32)
    with torch.no_grad():
        feats = enc.forward_features(video)
        state = enc.init_state(1, 32, 32)
        outs = []
        for t in range(4):
            out = enc.step(video[:, :, t], state)
            outs.append(out.features)
            state = out.new_state
        rolled = torch.cat(outs, dim=2)
    torch.testing.assert_close(rolled, feats, atol=1e-5, rtol=1e-4)


def test_cache_is_fixed_size_history() -> None:
    """Cache starts empty and is bounded by each stage's window: O(1) in T."""
    enc = _enc()
    state = enc.init_state(1, 32, 32)
    assert state.base.shape[1] == 6  # post-patchify channels (3*out_factor)
    assert state.down.shape[1] == 6  # downsampled channels
    assert state.levels.shape[1:3] == (4, 32)  # levels, level dim
    assert state.base.shape[2] == 0
    assert state.down.shape[2] == 0
    assert state.levels.shape[3] == 0
    assert state.count == 0


def test_cache_shape_stable_across_steps() -> None:
    """Cache windows grow to their bound, then stay fixed: O(1) memory in T."""
    enc = _enc()
    enc.eval()
    n = 2 * enc.history_frames
    video = torch.randn(1, 3, n, 32, 32)
    state = enc.init_state(1, 32, 32)
    with torch.no_grad():
        for t in range(n):
            out = enc.step(video[:, :, t], state)
            state = out.new_state
            assert state.base.shape[2] <= enc._correction_history
            assert state.down.shape[2] <= enc._stem_history
            assert state.levels.shape[3] <= enc.history_frames
            assert state.count == t + 1
    # Windows saturate at their bounds, never beyond.
    assert state.base.shape[2] == enc._correction_history
    assert state.down.shape[2] == enc._stem_history
    assert state.levels.shape[3] == enc.history_frames


def test_native_16_by_9_clip_keeps_aspect_end_to_end() -> None:
    """Native game resolution flows through the full encoder undistorted."""
    enc = _enc()
    enc.eval()
    with torch.no_grad():
        pyr = enc.forward_feature_maps(torch.randn(1, 3, 4, 144, 256))
        feats = enc.forward_features(torch.randn(1, 3, 4, 144, 256))
    assert pyr.shape[3:] == (4, 4, 8)  # T, H, W: /8 patch, /4 stem
    assert pyr.shape[5] == 2 * pyr.shape[4]  # 16:9 aspect preserved
    assert feats.shape == (1, 32, 4)
