"""Shared Hydra/CLI helpers for the train and eval entrypoints."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from convgamer.modules.lightning_module import ConvGamerModel, InceptionNeXtModule

# Absolute config path so Hydra finds configs regardless of CWD or __main__.
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"


def model_class(cfg: DictConfig) -> type[InceptionNeXtModule] | type[ConvGamerModel]:
    """Select the Lightning module from the configured model target."""
    if "inception" in str(cfg.model.get("target", "")):
        return InceptionNeXtModule
    return ConvGamerModel


def build_model(cfg: DictConfig) -> InceptionNeXtModule | ConvGamerModel:
    """Instantiate the Lightning module selected by ``model_class``."""
    return model_class(cfg)(cfg)
