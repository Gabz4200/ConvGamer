"""Behavior tests for ConvGamerModel video Lightning module."""

import torch
from omegaconf import DictConfig, OmegaConf

from convgamer.modules.lightning_module import ConvGamerModel


def _video_cfg() -> DictConfig:
    cfg = OmegaConf.create(
        {
            "model": {
                "input_dim": 3,
                "hidden_dim": 16,
                "num_layers": 1,
                "num_classes": 4,
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


def test_video_module_forward_produces_logits() -> None:
    m = ConvGamerModel(_video_cfg())
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 8, 32, 32))
    assert out.shape == (1, 4)


def test_video_module_steps_run() -> None:
    m = ConvGamerModel(_video_cfg())
    batch = (torch.randn(2, 3, 8, 32, 32), torch.randint(0, 4, (2,)))
    for step in (m.training_step, m.validation_step, m.test_step):
        assert torch.isfinite(step(batch, 0))


def test_video_module_optimizer_builds() -> None:
    m = ConvGamerModel(_video_cfg())
    opt = m.configure_optimizers()
    assert isinstance(opt, torch.optim.AdamW)


def test_video_temporal_mix_operates_on_spatial_maps() -> None:
    """Feature maps keep H,W > 1; logits collapse to per-class vector."""
    from convgamer.models import ConvGamerEncoder

    enc = ConvGamerEncoder(
        input_dim=3, hidden_dim=8, num_layers=1, num_classes=4, temporal_dilations=(1,)
    )
    enc.eval()
    with torch.no_grad():
        maps = enc.forward_feature_maps(torch.randn(1, 3, 4, 32, 32))
        logits = enc(torch.randn(1, 3, 4, 32, 32))
    assert maps.ndim == 5 and maps.shape[3] > 1 and maps.shape[4] > 1
    assert logits.shape == (1, 4)
