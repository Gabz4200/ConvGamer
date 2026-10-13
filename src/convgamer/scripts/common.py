"""Shared Hydra/CLI helpers for the train and eval entrypoints.

Composition root: configs describe *what* to build, and this module builds the
pure backbone with ``hydra.utils.instantiate`` and wraps it in its training
system.  Backbones never see a config object; adding one means registering a
model class and picking it in a model YAML — nothing here needs to change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from convgamer.data.datamemodule import ConvGamerDataModule
from convgamer.data.jepa_datamodule import VJEPAGamingDataModule
from convgamer.models.convgamer.encoder import ConvGamerEncoder
from convgamer.models.inception_next.encoder import InceptionNeXtEncoder
from convgamer.modules.jepa_module import ConvGamerVJEPAModel
from convgamer.modules.lightning_module import BackboneClassifier

# Absolute config path so Hydra finds configs regardless of CWD or __main__.
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"

# Marker target for the JEPA pretraining stack: it does not name a backbone
# class, so it is kept out of the ``instantiate`` path and handled explicitly.
_JEPA_TARGET = "convgamer.models.jepa"


def _backbone_kwargs_for_checkpoint(target: str, params: dict[str, Any]) -> dict[str, Any]:
    """Metadata the system records under its ``model`` hyperparameters.

    Saved checkpoints carry the backbone's construction parameters (resolved,
    config-type free) so ``export-hf`` can rebuild the HF config. Plain lists
    from Hydra are normalized to tuples.
    """
    kwargs = {"target": target, **params}
    for key in ("num_layers", "temporal_dilations", "mlp_ratios", "widths"):
        if isinstance(kwargs.get(key), list):
            kwargs[key] = tuple(kwargs[key])
    return kwargs


def _build_jepa_model(cfg: DictConfig) -> ConvGamerVJEPAModel:
    """Build a JEPA model from config sub-objects (encoder/predictor/loss)."""
    encoder = instantiate(cfg.model.encoder)
    predictor = instantiate(cfg.model.predictor)
    loss_fn = instantiate(cfg.model.loss)
    betas = tuple(cfg.optimizer.get("betas", [0.9, 0.999]))
    return ConvGamerVJEPAModel(
        encoder=encoder,
        predictor=predictor,
        loss=loss_fn,
        ema_decay=cfg.model.get("ema_decay", 0.99925),
        lr=cfg.model.get("lr", 5.25e-4),
        weight_decay=cfg.optimizer.weight_decay,
        warmup_steps=cfg.model.get("warmup_steps", 12000),
        total_steps=cfg.model.get("total_steps", 135000),
        betas=(float(betas[0]), float(betas[1])),
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
) -> BackboneClassifier | ConvGamerVJEPAModel:
    """Instantiate the training system selected by ``cfg.model.target``.

    Classification backbones are built with ``instantiate`` and injected into
    their system wrapper.  JEPA modules instantiate encoder/predictor/loss as
    sub-objects from the config, then receive them the same way.
    """
    target, params = backbone_kwargs(cfg)
    if target == _JEPA_TARGET:
        return _build_jepa_model(cfg)
    backbone = build_backbone(cfg)
    optimizer = cfg.get("optimizer", {})
    system = BackboneClassifier(
        backbone,
        lr=float(optimizer.get("lr", 1e-3)),
        weight_decay=float(optimizer.get("weight_decay", 0.0)),
        model_kwargs=_backbone_kwargs_for_checkpoint(target, params),
    )
    return system


def build_datamodule(
    cfg: DictConfig, *, num_workers: int | None = None
) -> ConvGamerDataModule | VJEPAGamingDataModule:
    """Single JEPA-vs-classic datamodule selector shared by train and eval.

    The model target and the datamodule must agree: the JEPA stack consumes
    ``(context, target, mask)`` triplets while the classification shell
    consumes ``(x, y)`` pairs. Mismatches fail here with a clear message
    instead of a cryptic collate crash mid-training. Trainer/optimizer
    groups stay freely composable.
    """
    is_jepa = str(cfg.model.get("target", "")) == _JEPA_TARGET
    if is_jepa:
        if "mode" not in cfg.data:
            raise ValueError(
                "model target is the JEPA stack but data config has no 'mode'; "
                "use data=jepa with model=jepa, or model=<backbone> with data=default."
            )
        return _build_jepa_datamodule(cfg, num_workers=num_workers)
    if "mode" in cfg.data:
        raise ValueError(
            "model target is a classification backbone but data config is a JEPA "
            "triplet config (has 'mode'); use data=default with classification "
            "models, or model=jepa with data=jepa."
        )
    return ConvGamerDataModule(**cfg.data)


def _build_jepa_datamodule(
    cfg: DictConfig, *, num_workers: int | None = None
) -> VJEPAGamingDataModule:
    data = cfg.data
    mode = data.get("mode", "fixture")
    workers = data.get("num_workers", 4) if num_workers is None else num_workers
    shared = {
        "batch_size": data.get("batch_size", 8),
        "num_workers": workers,
        "num_frames": data.get("num_frames", 16),
        "height": data.get("height", 144),
        "width": data.get("width", 256),
        "mask_ratio": data.get("mask_ratio", 0.25),
    }
    if mode == "fixture":
        return VJEPAGamingDataModule(**shared, mode=mode)
    return VJEPAGamingDataModule(
        **shared,
        mode=mode,
        image_mask_ratio=cfg.data.get("image_mask_ratio", 0.1),
        video_dataset_ids=cfg.data.get("video_dataset_ids", []),
        image_dataset_ids=cfg.data.get("image_dataset_ids", []),
        data_dir=cfg.data.get("data_dir", "./data/jepa"),
        sample_stride=cfg.data.get("sample_stride", 1),
        max_frames=cfg.data.get("max_frames", 10_000),
    )
