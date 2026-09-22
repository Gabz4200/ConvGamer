"""Shared Hydra/CLI helpers for the train and eval entrypoints.

Composition root: configs describe *what* to build, and this module builds the
pure backbone with ``hydra.utils.instantiate`` and wraps it in its training
system.  Backbones never see a config object; adding one means registering a
model class and picking it in a model YAML — nothing here needs to change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from convgamer.data.datamemodule import ConvGamerDataModule
from convgamer.data.jepa_datamodule import VJEPAGamingDataModule
from convgamer.models.convgamer.encoder import ConvGamerEncoder
from convgamer.models.inception_next.encoder import InceptionNeXtEncoder
from convgamer.modules.jepa_module import ConvGamerVJEPAModel
from convgamer.modules.lightning_module import (
    ClassificationLightningModule,
    ConvGamerModel,
    InceptionNeXtModule,
)

# Absolute config path so Hydra finds configs regardless of CWD or __main__.
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"

# Marker target for the JEPA pretraining stack: it does not name a backbone
# class, so it is kept out of the ``instantiate`` path and handled explicitly.
_JEPA_TARGET = "convgamer.models.jepa"


def _backbone_kwargs_for_checkpoint(target: str, params: dict[str, Any]) -> dict[str, Any]:
    """Metadata the system records under its ``model`` hyperparameters.

    Saved checkpoints carry the backbone's construction parameters (resolved,
    config-type free) so ``export-hf`` can rebuild the HF config from any
    checkpoint — including ones written before the shell refactor.  The config
    value of ``num_layers`` is normalized because Hydra hands us a plain list.
    """
    kwargs = {"target": target, **params}
    if isinstance(kwargs.get("num_layers"), list):
        kwargs["num_layers"] = tuple(kwargs["num_layers"])
    if isinstance(kwargs.get("temporal_dilations"), list):
        kwargs["temporal_dilations"] = tuple(kwargs["temporal_dilations"])
    if isinstance(kwargs.get("mlp_ratios"), list):
        kwargs["mlp_ratios"] = tuple(kwargs["mlp_ratios"])
    if kwargs.get("target_size") is not None and not isinstance(kwargs["target_size"], tuple):
        kwargs["target_size"] = tuple(kwargs["target_size"])
    return kwargs


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


def backbone_kwargs(cfg: DictConfig) -> tuple[str, dict[str, Any]]:
    """Split ``cfg.model`` into (target string, backbone kwargs).

    The ``target`` key is composition metadata, never a backbone argument; so
    are the JEPA sub-object keys (``encoder``/``predictor``/``loss``), which the
    JEPA branch of :func:`build_model` consumes directly.
    """
    container = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(container, dict):
        raise TypeError(f"model config must be a dict, got {type(container).__name__}")
    params = {str(k): v for k, v in container.items()}
    try:
        target = str(params.pop("target"))
    except KeyError:
        raise ValueError(f"model config needs a 'target' key, got {sorted(params)}") from None
    return target, params


def system_for_backbone(
    backbone: object,
) -> type[InceptionNeXtModule] | type[ConvGamerModel]:
    """Pick the classification system by what the backbone *is*, not by strings."""
    if isinstance(backbone, InceptionNeXtEncoder):
        return InceptionNeXtModule
    if isinstance(backbone, ConvGamerEncoder):
        return ConvGamerModel
    raise ValueError(
        f"No classification system for backbone type {type(backbone).__name__}; "
        "register it in system_for_backbone."
    )


def build_backbone(cfg: DictConfig) -> InceptionNeXtEncoder | ConvGamerEncoder:
    """Instantiate the pure backbone from ``cfg.model`` (JEPA targets excluded)."""
    target, params = backbone_kwargs(cfg)
    if target == _JEPA_TARGET:
        raise TypeError(
            f"model target {target!r} names the JEPA pretraining stack, not a "
            "backbone class; build the JEPA system with build_model() instead"
        )
    backbone = instantiate({"_target_": target, **params})
    if not isinstance(backbone, (InceptionNeXtEncoder, ConvGamerEncoder)):
        raise TypeError(
            f"model target {target!r} must instantiate an InceptionNeXtEncoder or "
            f"ConvGamerEncoder, got {type(backbone).__name__}"
        )
    return backbone


def build_model(
    cfg: DictConfig,
) -> ClassificationLightningModule:
    """Instantiate the training system selected by ``cfg.model.target``.

    Classification backbones are built with ``instantiate`` and injected into
    their system wrapper.  JEPA modules instantiate encoder/predictor/loss as
    sub-objects from the config, then receive them the same way.
    """
    target, params = backbone_kwargs(cfg)
    if target == _JEPA_TARGET:
        return cast(ClassificationLightningModule, _build_jepa_model(cfg))
    backbone = cast(Any, build_backbone(cfg))
    system_cls = system_for_backbone(backbone)
    optimizer = cfg.get("optimizer", {})
    system: ClassificationLightningModule = system_cls(  # type: ignore[call-overload]
        backbone,
        lr=float(optimizer.get("lr", 1e-3)),
        weight_decay=float(optimizer.get("weight_decay", 0.0)),
        model_kwargs=_backbone_kwargs_for_checkpoint(target, params),
    )
    return system


def build_datamodule(
    cfg: DictConfig, *, num_workers: int | None = None
) -> ConvGamerDataModule | VJEPAGamingDataModule:
    """Single JEPA-vs-classic datamodule selector shared by train and eval."""
    if str(cfg.model.get("target", "")) != _JEPA_TARGET:
        return ConvGamerDataModule(**cfg.data)
    mode = cfg.data.get("mode", "synthetic")
    workers = cfg.data.num_workers if num_workers is None else num_workers
    shared = {
        "batch_size": cfg.data.batch_size,
        "num_workers": workers,
        "num_frames": cfg.data.num_frames,
        "height": cfg.data.height,
        "width": cfg.data.width,
        "mask_ratio": cfg.data.mask_ratio,
        "to_oklab": cfg.data.to_oklab,
    }
    if mode == "synthetic":
        return VJEPAGamingDataModule(
            **shared,
            mode=mode,
            num_synthetic_samples=cfg.data.get("num_synthetic_samples", 64),
        )
    return VJEPAGamingDataModule(
        **shared,
        mode=mode,
        image_mask_ratio=cfg.data.get("image_mask_ratio", 0.1),
        video_dataset_ids=cfg.data.get("video_dataset_ids", []),
        image_dataset_ids=cfg.data.get("image_dataset_ids", []),
        regularization_dataset_ids=cfg.data.get("regularization_dataset_ids", []),
        data_dir=cfg.data.get("data_dir", "./data/jepa"),
        sample_stride=cfg.data.get("sample_stride", 1),
        max_frames=cfg.data.get("max_frames", 10_000),
    )
