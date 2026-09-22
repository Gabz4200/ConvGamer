"""Regression tests for checkpoint loading of the Lightning modules.

Covers three concerns that previously failed:
- raw checkpoint metadata is intact (``model`` hparams carry ``target``);
- plain ``load_from_checkpoint`` rebuilds the backbone instead of storing a dict;
- injected ``model=backbone`` preserves ``hparams["model"]["num_layers"]``.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch

from convgamer.models.convgamer.encoder import ConvGamerEncoder
from convgamer.models.inception_next.encoder import InceptionNeXtEncoder
from convgamer.modules.lightning_module import ConvGamerModel, InceptionNeXtModule

_INCEPTION_TARGET = "convgamer.models.inception_next.encoder.InceptionNeXtEncoder"
_CONVGAMER_TARGET = "convgamer.models.convgamer.encoder.ConvGamerEncoder"


def _save(module, path) -> None:
    """Write a Lightning-format checkpoint with the version tag Lightning expects."""
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": module.hparams,
            "pytorch-lightning_version": pl.__version__,
        },
        str(path),
    )


def _inception_module():
    encoder = InceptionNeXtEncoder(hidden_dim=16, num_layers=(1, 1, 1, 1), num_classes=4)
    return InceptionNeXtModule(
        encoder,
        lr=1e-3,
        weight_decay=0.0,
        model_kwargs={
            "target": _INCEPTION_TARGET,
            "hidden_dim": 16,
            "num_layers": (1, 1, 1, 1),
            "num_classes": 4,
        },
    )


def _video_module():
    encoder = ConvGamerEncoder(
        input_dim=3,
        hidden_dim=16,
        num_layers=1,
        num_classes=None,
        target_size=(8, 8),
        temporal_dilations=(1,),
    )
    return ConvGamerModel(
        encoder,
        lr=1e-3,
        weight_decay=0.0,
        model_kwargs={
            "target": _CONVGAMER_TARGET,
            "hidden_dim": 16,
            "num_layers": 1,
            "num_classes": None,
        },
    )


def test_checkpoint_metadata_intact(tmp_path) -> None:
    """The saved hyper-parameter metadata survives the round trip."""
    ckpt = tmp_path / "model.ckpt"
    _save(_inception_module(), ckpt)
    raw = torch.load(str(ckpt), weights_only=False)
    model_cfg = raw["hyper_parameters"]["model"]
    assert model_cfg["target"] == _INCEPTION_TARGET
    assert list(model_cfg["num_layers"]) == [1, 1, 1, 1]


def test_plain_load_rebuilds_backbone(tmp_path) -> None:
    """Plain load must produce a real encoder, not a dict."""
    ckpt = tmp_path / "model.ckpt"
    _save(_inception_module(), ckpt)
    restored = InceptionNeXtModule.load_from_checkpoint(str(ckpt), strict=False)
    assert isinstance(restored.model, InceptionNeXtEncoder)
    assert restored.hparams["model"]["target"] == _INCEPTION_TARGET
    assert list(restored.hparams["model"]["num_layers"]) == [1, 1, 1, 1]


def test_injected_backbone_preserves_metadata(tmp_path) -> None:
    """Injection must preserve the saved ``num_layers`` metadata."""
    ckpt = tmp_path / "model.ckpt"
    _save(_inception_module(), ckpt)
    backbone = InceptionNeXtEncoder(hidden_dim=16, num_layers=(1, 1, 1, 1), num_classes=4)
    injected = InceptionNeXtModule.load_from_checkpoint(str(ckpt), model=backbone, strict=False)
    assert isinstance(injected.model, InceptionNeXtEncoder)
    assert list(injected.hparams["model"]["num_layers"]) == [1, 1, 1, 1]


def test_video_plain_load_rebuilds_backbone(tmp_path) -> None:
    """ConvGamer checkpoint plain-load rebuilds a ConvGamerEncoder."""
    ckpt = tmp_path / "model.ckpt"
    _save(_video_module(), ckpt)
    restored = ConvGamerModel.load_from_checkpoint(str(ckpt), strict=False)
    assert isinstance(restored.model, ConvGamerEncoder)
    assert restored.hparams["model"]["num_layers"] == 1
