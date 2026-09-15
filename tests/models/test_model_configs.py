"""Behavior tests for InceptionNeXt/ConvGamer model-size configs.

Seams tested:
- ``configs/model/inception_tiny.yaml`` and ``inception_small.yaml`` build
  InceptionNeXtEncoder instances with the expected stage layouts.
- ``configs/model/convgamer_tiny.yaml`` and ``convgamer_small.yaml`` build
  ConvGamerEncoder instances with the expected stage layouts and no head.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from convgamer.models.convgamer.encoder import ConvGamerEncoder
from convgamer.models.inception_next.encoder import InceptionNeXtEncoder


def _load_cfg(name: str) -> dict[str, Any]:
    path = Path("configs/model") / f"{name}.yaml"
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]


def _compose_cfg(overrides: list[str]) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
        composed = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(composed, dict)
        return {str(k): v for k, v in composed.items()}


def test_inception_tiny_config_builds() -> None:
    cfg = _load_cfg("inception_tiny")
    enc = InceptionNeXtEncoder(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_classes=cfg["num_classes"],
        layer_scale_init=cfg["layer_scale_init"],
        mlp_ratios=tuple(cfg["mlp_ratios"]),
    )
    assert enc.feature_dim == 768  # 96 * 2^3
    assert isinstance(enc.head, torch.nn.Linear)
    assert enc.head.out_features == 10
    x = torch.randn(1, 3, 32, 32)
    out = enc.forward_feature_map(x)
    assert out.shape[0] == 1
    assert out.shape[1] == 768


def test_inception_small_config_builds() -> None:
    cfg = _load_cfg("inception_small")
    enc = InceptionNeXtEncoder(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_classes=cfg["num_classes"],
        layer_scale_init=cfg["layer_scale_init"],
        mlp_ratios=tuple(cfg["mlp_ratios"]),
        widths=tuple(cfg["widths"]),
    )
    assert enc.feature_dim == 1024  # last stage width
    x = torch.randn(1, 3, 32, 32)
    out = enc.forward_feature_map(x)
    assert out.shape[0] == 1
    assert out.shape[1] == 1024


def test_convgamer_tiny_config_builds() -> None:
    cfg = _load_cfg("convgamer_tiny")
    enc = ConvGamerEncoder(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_classes=cfg["num_classes"],
        layer_scale_init=cfg["layer_scale_init"],
        mlp_ratios=tuple(cfg["mlp_ratios"]),
        out_factor=cfg["out_factor"],
        use_softmax=cfg["use_softmax"],
        temporal_dilations=tuple(cfg["temporal_dilations"]),
    )
    assert enc.frame_encoder.feature_dim == 768
    assert isinstance(enc.head, torch.nn.Identity)
    x = torch.randn(1, 3, 4, 32, 32)
    out = enc.forward_feature_maps(x)
    assert out.shape[0] == 1


def test_convgamer_small_config_builds() -> None:
    cfg = _load_cfg("convgamer_small")
    enc = ConvGamerEncoder(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_classes=cfg["num_classes"],
        layer_scale_init=cfg["layer_scale_init"],
        mlp_ratios=tuple(cfg["mlp_ratios"]),
        out_factor=cfg["out_factor"],
        use_softmax=cfg["use_softmax"],
        temporal_dilations=tuple(cfg["temporal_dilations"]),
        widths=tuple(cfg["widths"]),
    )
    assert enc.frame_encoder.feature_dim == 1024
    assert isinstance(enc.head, torch.nn.Identity)
    x = torch.randn(1, 3, 4, 32, 32)
    out = enc.forward_feature_maps(x)
    assert out.shape[0] == 1


def test_jepa_tiny_config_composition() -> None:
    """Tiny JEPA stack resolves with matching feature dims and Appendix A values."""
    overrides = [
        "experiment=jepa_tiny",
        "model=jepa_tiny",
        "data=jepa_tiny",
        "trainer=jepa_tiny",
        "optimizer=jepa",
    ]
    cfg = _compose_cfg(overrides)
    assert cfg["model"]["target"] == "convgamer.models.jepa"
    assert cfg["model"]["encoder"]["num_classes"] is None
    assert cfg["model"]["encoder"]["hidden_dim"] == 96
    assert cfg["model"]["encoder"]["num_layers"] == [3, 3, 9, 3]
    assert cfg["model"]["predictor"]["feature_dim"] == 768
    assert cfg["model"]["loss"]["feature_dim"] == 768
    assert cfg["model"]["loss"]["lambda_base"] == 0.5
    assert cfg["model"]["loss"]["lambda_image"] == 0.7
    assert cfg["model"]["ema_decay"] == 0.99925
    assert cfg["model"]["lr"] == 5.25e-4
    assert cfg["optimizer"]["lr"] == 5.25e-4
    assert cfg["optimizer"]["weight_decay"] == 0.04
    assert cfg["data"]["num_frames"] == 12
    assert cfg["data"]["to_oklab"] is True
    assert len(cfg["data"]["video_dataset_ids"]) == 7
    assert len(cfg["data"]["archive_dataset_ids"]) == 4
    assert len(cfg["data"]["image_dataset_ids"]) == 2
    assert len(cfg["data"]["regularization_dataset_ids"]) == 1
    assert cfg["trainer"]["max_steps"] == 135000
    assert cfg["trainer"]["precision"] == "16-mixed"
