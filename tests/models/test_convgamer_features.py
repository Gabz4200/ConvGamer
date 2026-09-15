"""Behavior tests for ConvGamerEncoder/ConvGamerModel feature extraction.

Seams:
- ConvGamerEncoder.forward_features: returns (B, F, T) feature sequence with
  NO head, NO causal pooling. Pure backbone output.
- ConvGamerEncoder.forward: forward_features -> causal cumulative-mean pool
  over past frames -> head -> logits/class sequence.
- ConvGamerModel.forward_features: delegates to encoder.forward_features.
"""

import torch
from omegaconf import DictConfig, OmegaConf

from convgamer.models.convgamer.encoder import ConvGamerEncoder
from convgamer.modules.lightning_module import ConvGamerModel


def _cfg(num_classes: int = 4) -> DictConfig:
    cfg = OmegaConf.create(
        {
            "model": {
                "input_dim": 3,
                "hidden_dim": 16,
                "num_layers": 1,
                "num_classes": num_classes,
                "layer_scale_init": 1e-6,
                "mlp_ratios": [4, 4, 4, 3],
                "out_factor": 2,
                "use_softmax": False,
                "temporal_dilations": [1, 2],
            },
            "optimizer": {"lr": 1e-3, "weight_decay": 0.0},
        }
    )
    assert isinstance(cfg, DictConfig)
    return cfg


def _enc(num_classes: int | None = 4) -> ConvGamerEncoder:
    return ConvGamerEncoder(
        input_dim=3,
        hidden_dim=16,
        num_layers=1,
        num_classes=num_classes,
        layer_scale_init=1e-6,
        mlp_ratios=(4, 4, 4, 3),
        out_factor=2,
        use_softmax=False,
        temporal_dilations=(1, 2),
    )


def test_forward_features_returns_sequence_no_head() -> None:
    """forward_features returns (B, F, T) features; no classification head applied."""
    enc: ConvGamerEncoder = _enc(num_classes=4)
    enc.eval()
    with torch.no_grad():
        feats = enc.forward_features(torch.randn(2, 3, 8, 32, 32))
    assert feats.shape[:2] == (2, enc.frame_encoder.feature_dim)
    assert feats.shape[2] == 8  # temporal preserved, no subsampling


def test_forward_features_pure_backbone_no_pooling() -> None:
    """forward_features must NOT do causal cumsum pooling — that lives in forward."""
    enc: ConvGamerEncoder = _enc(num_classes=4)
    enc.eval()
    video = torch.randn(1, 3, 8, 32, 32)
    # forward_features must not raise and must return a tensor (B, F, T)
    with torch.no_grad():
        feats = enc.forward_features(video)
    assert feats.ndim == 3
    # forward still produces pooled logits
    with torch.no_grad():
        logits = enc(video)
    assert logits.shape == (1, 4)


def test_forward_matches_head_on_pooled_features() -> None:
    """forward applies head to causal-pooled forward_features output."""
    enc: ConvGamerEncoder = _enc(num_classes=4)
    enc.eval()
    video = torch.randn(1, 3, 8, 32, 32)
    with torch.no_grad():
        feats: torch.Tensor = enc.forward_features(video)
        logits: torch.Tensor = enc(video)
    t = feats.shape[2]
    pooled = torch.cumsum(feats, dim=2) / torch.arange(
        1, t + 1, device=feats.device, dtype=feats.dtype
    )
    expected = enc.head(enc.norm(pooled[:, :, -1]))
    torch.testing.assert_close(logits.flatten(), expected.flatten())


def test_forward_features_streaming_parity() -> None:
    """step() one frame at a time must match forward_features over full clip."""
    enc: ConvGamerEncoder = _enc(num_classes=0)
    enc.eval()
    video = torch.randn(1, 3, 8, 32, 32)
    with torch.no_grad():
        feats_par: torch.Tensor = enc.forward_features(video)
        state = enc.init_state(1, 32, 32)
        outs = [enc.step(video[:, :, t], state)[0] for t in range(video.shape[2])]
        feats_step = torch.cat(outs, dim=2)  # (B, F, T)
    assert feats_step.shape == feats_par.shape
    torch.testing.assert_close(feats_step, feats_par, atol=1e-5, rtol=1e-4)


def test_step_returns_features_and_logits() -> None:
    """step() returns (features, logits) tuple when head exists."""
    enc: ConvGamerEncoder = _enc(num_classes=4)
    enc.eval()
    video = torch.randn(1, 3, 8, 32, 32)
    with torch.no_grad():
        state = enc.init_state(1, 32, 32)
        feat, logits = enc.step(video[:, :, 0], state)
    assert feat.shape == (1, enc.frame_encoder.feature_dim, 1)
    assert logits.shape[0] == 1


def test_step_logits_match_forward_causal_pooling() -> None:
    """Streaming cumulative-mean logits match forward() output at last frame."""
    enc: ConvGamerEncoder = _enc(num_classes=4)
    enc.eval()
    video = torch.randn(1, 3, 6, 32, 32)
    with torch.no_grad():
        logits_par = enc(video)  # causal cumsum pool + head
        state = enc.init_state(1, 32, 32)
        last_logits: torch.Tensor = enc.step(video[:, :, 0], state)[1]
        for t in range(1, video.shape[2]):
            last_logits = enc.step(video[:, :, t], state)[1]
    torch.testing.assert_close(last_logits.flatten(), logits_par.flatten(), atol=1e-5, rtol=1e-4)


def test_forward_sequence_returns_per_frame_logits() -> None:
    """return_sequence=True emits per-frame logits (B, T, K)."""
    enc: ConvGamerEncoder = _enc(num_classes=4)
    enc.eval()
    with torch.no_grad():
        seq = enc(torch.randn(1, 3, 8, 32, 32), return_sequence=True)
    assert seq.shape[-1] == 4
    assert seq.shape[1] == 8  # temporal preserved


def test_foundation_seam_headless_then_attachable() -> None:
    """Default encoder is headless; head attaches later for downstream tasks."""
    import warnings

    enc: ConvGamerEncoder = _enc(num_classes=None)
    assert isinstance(enc.head, torch.nn.Identity)
    with torch.no_grad():
        pooled = enc(torch.randn(1, 3, 4, 32, 32))
    assert pooled.shape[1] == enc.frame_encoder.feature_dim
    enc.add_classification_head(4)
    with torch.no_grad():
        logits = enc(torch.randn(1, 3, 4, 32, 32))
    assert logits.shape == (1, 4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _enc(num_classes=4)
    assert any(issubclass(w.category, FutureWarning) for w in caught)


def test_module_forward_features_delegates_to_encoder() -> None:
    """ConvGamerModel.forward_features calls encoder.forward_features."""
    mod: ConvGamerModel = ConvGamerModel(_cfg(num_classes=4))
    mod.eval()
    with torch.no_grad():
        feats = mod.forward_features(torch.randn(1, 3, 8, 32, 32))
    assert feats.shape[0] == 1
    assert feats.shape[2] == 8  # temporal preserved
