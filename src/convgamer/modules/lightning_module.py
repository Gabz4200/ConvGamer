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
from omegaconf import DictConfig
from torch import nn

from convgamer.models.inception_next import Encoder


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
        self.save_hyperparameters(cfg)
        self.model = Encoder(
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
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Paper §4.1: AdamW with lr = 0.001 × batchsize/1024
        lr = self.hparams["optimizer"]["lr"]
        weight_decay = self.hparams["optimizer"]["weight_decay"]
        return torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )


class ConvGamerModel(InceptionNeXtModule):
    """ConvGamer LightningModule — extends InceptionNeXtModule.

    Adds the ConvGamer geometry-op integration (wired in via ``cfg.ops.backend``)
    applied in later stages.  Until the geometry ops are composed into the
    forward pass this module trains identically to ``InceptionNeXtModule``.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
