"""Behavior tests for wiring seams: script dispatch, trainer factory, datamodules."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from convgamer.callbacks.ema_update import EMAUpdateCallback
from convgamer.data.datamemodule import ConvGamerDataModule
from convgamer.data.jepa_datamodule import VJEPAGamingDataModule
from convgamer.integrations.transformers import ConvGamerConfig, ConvGamerModel
from convgamer.modules.jepa_module import ConvGamerVJEPAModel
from convgamer.modules.lightning_module import ConvGamerModel as PLVideoModel
from convgamer.modules.lightning_module import InceptionNeXtModule
from convgamer.scripts.common import (
    backbone_kwargs,
    build_backbone,
    build_model,
    system_for_backbone,
)
from convgamer.training.engine import create_trainer


def _target_cfg(target: str):
    return OmegaConf.create({"model": {"target": target}})


def _load_full_model_cfg(name: str):
    """Load a real model config file (configs/model/<name>.yaml)."""
    path = Path("configs/model") / f"{name}.yaml"
    return OmegaConf.create({"model": OmegaConf.to_container(OmegaConf.load(path), resolve=True)})


def test_when_jepa_marker_target_then_backbone_build_refuses() -> None:
    """The JEPA marker names a pretraining stack, not a backbone class.

    ``backbone_kwargs`` separates the ``target`` marker from the backbone
    arguments so ``target`` can never leak into a backbone constructor; the
    classification path only accepts the two registered backbone types.
    """
    target, params = backbone_kwargs(_target_cfg("convgamer.models.jepa"))
    assert target == "convgamer.models.jepa"
    assert params == {}
    with pytest.raises(TypeError, match="pretraining stack"):
        build_backbone(_target_cfg("convgamer.models.jepa"))


def test_when_inception_module_cfg_then_image_backbone_constructs() -> None:
    """The full inception model config builds an InceptionNeXt backbone."""
    from hydra.utils import instantiate

    cfg = _load_full_model_cfg("inception_next")
    target, params = backbone_kwargs(cfg)
    assert target == "convgamer.models.inception_next.encoder.InceptionNeXtEncoder"
    backbone = instantiate({"_target_": target, **params})
    assert system_for_backbone(backbone) is InceptionNeXtModule


def test_when_video_module_cfg_then_video_backbone_constructs() -> None:
    """The full convgamer model config builds a ConvGamer backbone."""
    from hydra.utils import instantiate

    cfg = _load_full_model_cfg("convgamer")
    target, params = backbone_kwargs(cfg)
    assert target == "convgamer.models.convgamer.encoder.ConvGamerEncoder"
    backbone = instantiate({"_target_": target, **params})
    assert system_for_backbone(backbone) is PLVideoModel


def test_when_unknown_backbone_then_system_selection_fails() -> None:
    """A backbone without a registered system fails with the type named."""
    with pytest.raises(ValueError, match="object"):
        system_for_backbone(object())


def test_when_jepa_config_then_build_model_runs() -> None:
    """JEPA config instantiates encoder/predictor/loss end to end."""
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=jepa",
                "data=jepa",
                "trainer=jepa",
                "optimizer=jepa",
            ],
        )
    model = build_model(cfg)
    assert isinstance(model, ConvGamerVJEPAModel)
    assert model.ema_decay == 0.99925


def test_when_jepa_trainer_cfg_then_ema_callback_builds() -> None:
    """JEPA trainer config attaches the target-encoder update callback."""
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
        cfg = compose(config_name="config", overrides=["trainer=jepa"])
    cfg.trainer.accelerator = "cpu"
    cfg.trainer.devices = 1
    trainer = create_trainer(cfg)
    callbacks = trainer.callbacks  # type: ignore[attr-defined]
    assert any(isinstance(callback, EMAUpdateCallback) for callback in callbacks)


def test_when_default_trainer_cfg_then_trainer_builds() -> None:
    """Default trainer config builds a PL Trainer without launching training."""
    cfg = OmegaConf.create(
        {
            "trainer": {
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "log_every_n_steps": 10,
                "max_epochs": 1,
                "callbacks": ["model_checkpoint"],
            },
            "fast_dev_run": True,
        }
    )
    trainer = create_trainer(cfg)
    assert trainer.max_epochs == 1


def test_when_synthetic_datamodule_then_loaders_yield_batches() -> None:
    """Synthetic JEPA datamodule yields (x, y, mask) batches after setup."""
    dm = VJEPAGamingDataModule(
        batch_size=2,
        num_workers=0,
        num_frames=4,
        height=16,
        width=16,
        num_synthetic_samples=4,
        mode="synthetic",
    )
    dm.setup()
    x, y, mask = next(iter(dm.train_dataloader()))
    assert x.shape == y.shape == (2, 3, 4, 16, 16)
    assert mask.shape == (2, 4, 16, 16)
    assert mask.dtype == torch.bool
    assert dm.val_dataloader() is not None


def test_when_classification_datamodule_not_setup_then_loaders_fail() -> None:
    """Classification loaders fail loudly before dataset setup."""
    dm = ConvGamerDataModule()
    with pytest.raises(RuntimeError, match="train_ds"):
        dm.train_dataloader()
    with pytest.raises(RuntimeError, match="val_ds"):
        dm.val_dataloader()
    with pytest.raises(RuntimeError, match="test_ds/val_ds"):
        dm.test_dataloader()


def test_when_classification_datamodule_setup_then_test_aliases_val() -> None:
    """setup(None) keeps the evaluation test loader aliased to validation."""
    dm = ConvGamerDataModule()
    dm.setup(None)
    assert dm.test_ds is dm.val_ds


def test_when_jepa_datamodule_not_setup_then_train_loader_fails() -> None:
    """JEPA training loader fails loudly before dataset setup."""
    dm = VJEPAGamingDataModule(mode="synthetic")
    with pytest.raises(RuntimeError, match="train_ds"):
        dm.train_dataloader()


def test_when_invalid_jepa_mode_then_raises() -> None:
    """Unknown JEPA datamodule modes fail at setup with the bad mode named."""
    dm = VJEPAGamingDataModule(mode="no-such-mode")
    with pytest.raises(ValueError, match="no-such-mode"):
        dm.setup()


def test_when_mixed_mode_then_video_and_image_loaders_stay_separate(tmp_path) -> None:
    """Mixed mode returns modality-homogeneous loaders; no T=12/T=1 collate mix."""
    from torchvision.io import write_png

    video_dir = tmp_path / "videos"
    image_dir = tmp_path / "images"
    video_dir.mkdir()
    image_dir.mkdir()
    for i in range(2):
        write_png((torch.rand(3, 16, 16) * 255).to(torch.uint8), str(image_dir / f"img{i}.png"))
    dm = VJEPAGamingDataModule(
        batch_size=1,
        num_workers=0,
        num_frames=4,
        height=16,
        width=16,
        video_dataset_ids=[str(video_dir / "*.mp4")],
        image_dataset_ids=[str(image_dir / "*.png")],
        mode="mixed",
    )
    dm.setup()
    loaders = dm.train_dataloader()
    assert isinstance(loaders, list) and len(loaders) == 2
    _, y_img, mask_img = next(iter(loaders[1]))
    assert y_img.shape == (1, 3, 1, 16, 16)
    assert mask_img.shape == (1, 1, 16, 16)


def test_when_invalid_modality_then_raises() -> None:
    """Classification datamodule rejects modalities outside image/video."""
    with pytest.raises(ValueError, match="modality"):
        ConvGamerDataModule(modality="audio")


def test_when_video_modality_then_loader_shapes_match() -> None:
    """Video classification datamodule yields (B, C, T, H, W) batches."""
    dm = ConvGamerDataModule(
        batch_size=2, num_samples=8, modality="video", num_frames=4, image_size=16
    )
    dm.setup()
    x, y = next(iter(dm.train_dataloader()))
    assert x.shape == (2, 3, 4, 16, 16)
    assert y.shape == (2,)


def test_when_hf_wrapper_missing_inputs_then_raises() -> None:
    """HF wrapper requires one of input_ids/pixel_values/inputs_embeds."""
    wrapper = ConvGamerModel(ConvGamerConfig(hidden_dim=16, num_layers=1, num_classes=4))
    with pytest.raises(ValueError, match="Specify input_ids"):
        wrapper.forward()


def test_when_hf_wrapper_pixel_values_then_returns_dict() -> None:
    """HF wrapper maps pixel_values through the native backbone to a dict."""
    wrapper = ConvGamerModel(ConvGamerConfig(hidden_dim=16, num_layers=1, num_classes=4))
    wrapper.eval()
    with torch.no_grad():
        out = wrapper(pixel_values=torch.randn(1, 3, 32, 32))
    assert out.last_hidden_state.shape == (1, 4)
