"""Lightning modules for InceptionNeXt (arXiv:2303.16900) and ConvGamer.

``InceptionNeXtModule`` trains the 2D image backbone on (B, C, H, W).
``ConvGamerModel`` trains the causal video encoder on (B, C, T, H, W).
Both share ``ClassificationLightningModule`` for the step/optimizer logic.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from convgamer.models.convgamer.encoder import ConvGamerEncoder
from convgamer.models.inception_next import InceptionNeXtEncoder


def _optimizer_from_cfg(parameters, cfg) -> torch.optim.AdamW:
    opt = cfg.optimizer if isinstance(cfg, DictConfig) else cfg["optimizer"]
    if not isinstance(opt, DictConfig):
        assert isinstance(opt, dict)
        return torch.optim.AdamW(parameters, lr=opt["lr"], weight_decay=opt["weight_decay"])
    opt = OmegaConf.to_container(opt, resolve=True)
    assert isinstance(opt, dict)
    return torch.optim.AdamW(parameters, lr=opt["lr"], weight_decay=opt["weight_decay"])


def _cls_metrics(logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    loss = nn.functional.cross_entropy(logits, y)
    acc = (logits.argmax(dim=1) == y).float().mean()
    return loss, acc


class ClassificationLightningModule(pl.LightningModule):
    """Shared train/val/test step and AdamW wiring for classifiers."""

    def _step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch
        loss, acc = _cls_metrics(self(x), y)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def test_step(self, batch, _):
        return self._step(batch, "test")

    def configure_optimizers(self):
        return _optimizer_from_cfg(self.parameters(), self.hparams)


class InceptionNeXtModule(ClassificationLightningModule):
    """Pure InceptionNeXt LightningModule — matches paper §4.1.

    AdamW optimiser, CrossEntropyLoss on classification logits.  The
    optimiser hyper-parameters (``lr``, ``weight_decay``) come from the
    ``optimizer`` section of *cfg*, mirroring the paper's recipe:

    - lr = 0.001 x batchsize/1024  (paper §4.1)
    - weight_decay = 0.05
    - cosine decay schedule (configured on the trainer side)
    - 300 epochs (paper); reduced for dev via ``trainer.max_epochs``

    This module depends on the InceptionNeXt implementation only — it does
    not import or reference any ConvGamer-specific code.
    """

    model: InceptionNeXtEncoder

    def __init__(self, cfg: DictConfig):
        super().__init__()
        container = (
            OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
        )
        self.save_hyperparameters(container)
        mlp_ratios = tuple(cfg.model.get("mlp_ratios", (4, 4, 4, 3)))
        self.model = InceptionNeXtEncoder(
            input_dim=cfg.model.input_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            num_classes=cfg.model.num_classes,
            layer_scale_init=cfg.model.layer_scale_init,
            mlp_ratios=mlp_ratios,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ConvGamerModel(ClassificationLightningModule):
    """ConvGamer video LightningModule — trains ``ConvGamerEncoder`` end to end.

    Batch is ``(video, label)`` with video shaped (B, C, T, H, W).
    Fully causal: pooled logits at frame ``t`` see only frames ``<= t``.
    """

    model: ConvGamerEncoder

    def __init__(self, cfg: DictConfig):
        super().__init__()
        container = (
            OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
        )
        self.save_hyperparameters(container)
        model_cfg = cfg.model
        target_size = model_cfg.get("target_size", None)
        if target_size is not None:
            target_size = tuple(target_size)
        temporal_dilations = tuple(model_cfg.get("temporal_dilations", (1, 2, 4)))
        mlp_ratios = tuple(model_cfg.get("mlp_ratios", (4, 4, 4, 3)))
        self.model = ConvGamerEncoder(
            input_dim=model_cfg.input_dim,
            hidden_dim=model_cfg.hidden_dim,
            num_layers=model_cfg.num_layers,
            num_classes=model_cfg.num_classes,
            layer_scale_init=model_cfg.layer_scale_init,
            mlp_ratios=mlp_ratios,
            out_factor=model_cfg.get("out_factor", 2),
            use_softmax=model_cfg.get("use_softmax", False),
            target_size=target_size,
            temporal_dilations=temporal_dilations,
        )

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        return self.model(x, return_sequence=return_sequence)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Delegate to the encoder's feature extractor."""
        return self.model.forward_features(x)

    def init_state(self, batch_size: int = 1, height: int = 32, width: int = 32) -> dict:
        """Delegate to the encoder's streaming state."""
        return self.model.init_state(batch_size, height, width)

    def step(
        self, x_t: torch.Tensor, state: dict | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stream one frame through the full video pipeline -> (features, logits)."""
        return self.model.step(x_t, state)
