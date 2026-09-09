from __future__ import annotations

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import nn


class ConvGamerModel(pl.LightningModule):
    """Lightning module wrapping the InceptionNeXt encoder.

    Follows the paper's §4.1 setup: AdamW optimizer, CrossEntropyLoss
    on classification logits.  The geometry-op integration (ConvGamer-specific)
    is wired in via ``cfg.ops.backend`` and applied in later stages.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        from convgamer.models.registry import get_model

        self.model = get_model(
            cfg.model.name,
            input_dim=cfg.model.input_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            num_classes=cfg.model.num_classes,
            layer_scale_init=cfg.model.layer_scale_init,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
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
        weight_decay = self.hparams["optimizer"].get("weight_decay", 0.05)
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
