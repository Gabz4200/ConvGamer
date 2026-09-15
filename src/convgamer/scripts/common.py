"""Shared Hydra/CLI helpers for the train and eval entrypoints."""

from __future__ import annotations

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig

from convgamer.modules.jepa_module import ConvGamerVJEPAModel
from convgamer.modules.lightning_module import ConvGamerModel, InceptionNeXtModule

# Absolute config path so Hydra finds configs regardless of CWD or __main__.
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"


def model_class(
    cfg: DictConfig,
) -> type[InceptionNeXtModule] | type[ConvGamerModel] | type[ConvGamerVJEPAModel]:
    """Select the Lightning module class from the configured model target."""
    target = str(cfg.model.get("target", ""))
    if "jepa" in target.lower():
        return ConvGamerVJEPAModel
    if "inception" in target:
        return InceptionNeXtModule
    return ConvGamerModel


def _build_jepa_model(cfg: DictConfig) -> ConvGamerVJEPAModel:
    """Build a JEPA model from config sub-objects (encoder/predictor/loss)."""
    encoder = instantiate(cfg.model.encoder)
    predictor = instantiate(cfg.model.predictor)
    loss_fn = instantiate(cfg.model.loss)
    return ConvGamerVJEPAModel(
        encoder=encoder,
        predictor=predictor,
        loss=loss_fn,
        ema_decay=cfg.model.get("ema_decay", 0.99925),
        lr=cfg.model.get("lr", 5.25e-4),
        weight_decay=cfg.optimizer.weight_decay,
        warmup_steps=cfg.model.get("warmup_steps", 12000),
        total_steps=cfg.model.get("total_steps", 135000),
    )


def build_model(
    cfg: DictConfig,
) -> InceptionNeXtModule | ConvGamerModel | ConvGamerVJEPAModel:
    """Instantiate the Lightning module selected by ``model_class``.

    For JEPA modules, instantiate encoder/predictor/loss as sub-objects
    from the config, then pass them to the module constructor.
    """
    target = str(cfg.model.get("target", ""))
    if "jepa" in target.lower():
        return _build_jepa_model(cfg)
    non_jepa_cls: type[InceptionNeXtModule] | type[ConvGamerModel] = model_class(cfg)  # type: ignore[assignment]
    return non_jepa_cls(cfg)
