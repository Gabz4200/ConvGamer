"""Behavior tests for wiring seams: script dispatch, trainer factory, datamodules."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from convgamer.data.datamemodule import ConvGamerDataModule
from convgamer.data.jepa_datamodule import VJEPAGamingDataModule
from convgamer.integrations.transformers import ConvGamerConfig, ConvGamerModel
from convgamer.modules.jepa_module import ConvGamerVJEPAModel
from convgamer.modules.lightning_module import ConvGamerModel as PLVideoModel
from convgamer.modules.lightning_module import InceptionNeXtModule
from convgamer.scripts.common import build_model, model_class
from convgamer.training.engine import create_trainer


def _target_cfg(target: str):
    return OmegaConf.create({"model": {"target": target}})


def test_when_jepa_target_then_jepa_module_class() -> None:
    """Targets containing 'jepa' dispatch to the V-JEPA Lightning module."""
    assert model_class(_target_cfg("convgamer.models.jepa")) is ConvGamerVJEPAModel


def test_when_inception_target_then_image_module_class() -> None:
    """Inception targets dispatch to the image classifier module."""
    assert model_class(_target_cfg("convgamer.models.inception_next.encoder")) is (
        InceptionNeXtModule
    )


def test_when_video_target_then_video_module_class() -> None:
    """Anything else dispatches to the causal video classifier module."""
    assert model_class(_target_cfg("convgamer.models.convgamer.encoder")) is PLVideoModel


def test_when_jepa_tiny_config_then_build_model_runs() -> None:
    """JEPA tiny config instantiates encoder/predictor/loss end to end."""
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=jepa_tiny",
                "model=jepa_tiny",
                "data=jepa_tiny",
                "trainer=jepa_tiny",
                "optimizer=jepa",
            ],
        )
    model = build_model(cfg)
    assert isinstance(model, ConvGamerVJEPAModel)
    assert model.ema_decay == 0.99925


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
                "callbacks": ["logger"],
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
