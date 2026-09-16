"""Shared Hydra/CLI helpers for the train and eval entrypoints."""

from __future__ import annotations

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig

from convgamer.data.datamemodule import ConvGamerDataModule
from convgamer.data.jepa_datamodule import VJEPAGamingDataModule
from convgamer.modules.jepa_module import ConvGamerVJEPAModel
from convgamer.modules.lightning_module import ConvGamerModel, InceptionNeXtModule

# Absolute config path so Hydra finds configs regardless of CWD or __main__.
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"

# Exact target strings from configs/model/*.yaml. Substring matching
# misroutes future names (e.g. "jepalike"), so dispatch is exact.
_JEPA_TARGET = "convgamer.models.jepa"
_MODEL_CLASS_BY_TARGET: dict[str, type] = {
    _JEPA_TARGET: ConvGamerVJEPAModel,
    "convgamer.models.inception_next.encoder": InceptionNeXtModule,
    "convgamer.models.inception_next.encoder.InceptionNeXtEncoder": InceptionNeXtModule,
    "convgamer.models.convgamer.encoder": ConvGamerModel,
    "convgamer.models.convgamer.encoder.ConvGamerEncoder": ConvGamerModel,
}


def model_class(
    cfg: DictConfig,
) -> type[InceptionNeXtModule] | type[ConvGamerModel] | type[ConvGamerVJEPAModel]:
    """Select the Lightning module class from the configured model target."""
    target = str(cfg.model.get("target", ""))
    try:
        return _MODEL_CLASS_BY_TARGET[target]
    except KeyError:
        valid = sorted(_MODEL_CLASS_BY_TARGET)
        raise ValueError(f"Unknown model target {target!r}. Valid: {valid}") from None


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
    if target == _JEPA_TARGET:
        return _build_jepa_model(cfg)
    non_jepa_cls: type[InceptionNeXtModule] | type[ConvGamerModel] = model_class(cfg)  # type: ignore[assignment]
    return non_jepa_cls(cfg)


def build_datamodule(
    cfg: DictConfig, *, num_workers: int | None = None
) -> ConvGamerDataModule | VJEPAGamingDataModule:
    """Single JEPA-vs-classic datamodule selector shared by train and eval."""
    if str(cfg.model.get("target", "")) != _JEPA_TARGET:
        return ConvGamerDataModule(**cfg.data)
    mode = cfg.data.get("mode", "synthetic")
    workers = cfg.data.num_workers if num_workers is None else num_workers
    if mode == "synthetic":
        return VJEPAGamingDataModule(
            batch_size=cfg.data.batch_size,
            num_workers=workers,
            num_frames=cfg.data.num_frames,
            height=cfg.data.height,
            width=cfg.data.width,
            mask_ratio=cfg.data.mask_ratio,
            to_oklab=cfg.data.to_oklab,
            mode=mode,
            num_synthetic_samples=cfg.data.get("num_synthetic_samples", 64),
        )
    return VJEPAGamingDataModule(
        batch_size=cfg.data.batch_size,
        num_workers=workers,
        num_frames=cfg.data.num_frames,
        height=cfg.data.height,
        width=cfg.data.width,
        mask_ratio=cfg.data.mask_ratio,
        image_mask_ratio=cfg.data.get("image_mask_ratio", 0.1),
        to_oklab=cfg.data.to_oklab,
        video_dataset_ids=cfg.data.get("video_dataset_ids", []),
        image_dataset_ids=cfg.data.get("image_dataset_ids", []),
        regularization_dataset_ids=cfg.data.get("regularization_dataset_ids", []),
        data_dir=cfg.data.get("data_dir", "./data/jepa"),
        mode=mode,
        sample_stride=cfg.data.get("sample_stride", 1),
        max_frames=cfg.data.get("max_frames", 10_000),
    )
