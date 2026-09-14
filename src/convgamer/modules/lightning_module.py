"""Lightning modules for InceptionNeXt (arXiv:2303.16900) and ConvGamer.

``InceptionNeXtModule`` is the pure training module — it imports the
InceptionNeXt backbone directly from ``convgamer.models.inception_next`` and
knows nothing about ConvGamer-specific geometry ops.

``ConvGamerModel`` extends it to wire in the ConvGamer geometry-op backend
(via ``cfg.ops.backend``) in later stages.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from convgamer.models.inception_next import InceptionNeXtEncoder


class InceptionNeXtModule(pl.LightningModule):
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

    def __init__(self, cfg: DictConfig):
        super().__init__()
        container = (
            OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
        )  # type: ignore[arg-type]
        self.save_hyperparameters(container)
        self.model = InceptionNeXtEncoder(
            input_dim=cfg.model.input_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            num_classes=cfg.model.num_classes,
            layer_scale_init=cfg.model.layer_scale_init,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # Paper §4.1: AdamW with lr = 0.001 × batchsize/1024
        opt_cfg = (
            self.hparams["optimizer"]
            if "optimizer" in self.hparams
            else self.hparams.get("optimizer", {})
        )  # type: ignore[attr-defined]
        # OmegaConf container is plain dict after save_hyperparameters conversion
        if isinstance(opt_cfg, DictConfig):
            opt_cfg = OmegaConf.to_container(opt_cfg, resolve=True)  # type: ignore[assignment]
        lr = opt_cfg["lr"] if isinstance(opt_cfg, dict) else opt_cfg.lr  # type: ignore[union-attr]
        weight_decay = (
            opt_cfg["weight_decay"] if isinstance(opt_cfg, dict) else opt_cfg.weight_decay
        )  # type: ignore[union-attr]
        return torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )


class ConvGamerModel(InceptionNeXtModule):
    """ConvGamer LightningModule — extends InceptionNeXtModule.

    Placeholder for future geometry-op composition (``cfg.ops.backend``).
    Currently trains identically to ``InceptionNeXtModule`` until ops are wired
    into forward. Kept for HF/export compatibility.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
