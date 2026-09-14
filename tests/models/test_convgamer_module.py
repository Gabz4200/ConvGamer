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
    for stage in ("train", "val", "test"):
        loss = m._step(batch, stage)
        assert torch.isfinite(loss)


def test_video_module_optimizer_builds() -> None:
    m = ConvGamerModel(_video_cfg())
    opt = m.configure_optimizers()
    assert isinstance(opt, torch.optim.AdamW)
