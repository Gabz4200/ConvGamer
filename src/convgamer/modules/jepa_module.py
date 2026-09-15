"""V-JEPA 2.1 ConvGamer LightningModule.

Implements the dense predictive loss from V-JEPA 2.1 (arXiv:2603.14482)
using ConvGamer as the backbone instead of Vision Transformer.

Architecture (§2.3, §4):
    - x-encoder: ConvGamerEncoder (online, trainable)
    - y-encoder: EMA shadow copy of x-encoder (target, stop-grad)
    - predictor: VJEPAPredictor (maps context + mask tokens -> predictions)
    - loss: JEPALoss (L_predict + L_ctx, distance-weighted)

The encoder processes (B, C, T, H, W) video directly via 3D convolutions,
naturally handling both video (T>1) and static frames (T=1).
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn

from convgamer.models.convgamer.encoder import ConvGamerEncoder
from convgamer.models.jepa import EMAEncoder, JEPALoss, VJEPAPredictor


class ConvGamerVJEPAModel(pl.LightningModule):
    """LightningModule for V-JEPA 2.1 pretraining on ConvGamer.

    Args:
        encoder: ConvGamerEncoder (online, trainable). Classification head
            should be ``nn.Identity`` (foundation model seam).
        predictor: VJEPAPredictor for dense mask-token prediction.
        loss: JEPALoss computing L_predict + L_ctx.
        ema_decay: EMA coefficient for target encoder (default 0.99925).
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Linear warmup steps.
        total_steps: Total training steps for constant schedule.
        use_amp: Mixed precision (FP16 for T4).
    """

    def __init__(
        self,
        encoder: ConvGamerEncoder,
        predictor: VJEPAPredictor,
        loss: JEPALoss,
        ema_decay: float = 0.99925,
        lr: float = 5.25e-4,
        weight_decay: float = 0.04,
        warmup_steps: int = 12_000,
        total_steps: int = 135_000,
        use_amp: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.loss_fn = loss
        self.ema_decay = ema_decay
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.use_amp = use_amp

        # EMA target encoder (shadow copy of encoder)
        self.ema_encoder = EMAEncoder(encoder, decay=ema_decay)

        # Save hyperparameters (encoder is a module, not a config dict)
        # We don't save encoder/predictor/loss as hparams since they are
        # nn.Module instances, not config values.
        self.save_hyperparameters(ignore=["encoder", "predictor", "loss", "ema_encoder"])

    def encode_x(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the x-view (context encoder on masked input).

        Args:
            x: ``(B, C, T, H, W)`` masked video in oklab space.
            mask: ``(B, T, H, W)`` boolean mask.

        Returns:
            Encoder features ``(B, F, T', H', W')``.
        """
        return self.encoder.forward_feature_maps(x)

    @torch.no_grad()
    def encode_y(self, y: torch.Tensor) -> torch.Tensor:
        """Encode the y-view (target encoder on clean input, no grad).

        Args:
            y: ``(B, C, T, H, W)`` clean video in oklab space.

        Returns:
            Target features ``(B, F, T', H', W')``.
        """
        from convgamer.models.convgamer.encoder import ConvGamerEncoder

        shadow = self.ema_encoder.shadow_module
        if isinstance(shadow, ConvGamerEncoder):
            return shadow.forward_feature_maps(y)
        raise TypeError(f"EMA shadow module is not a ConvGamerEncoder: {type(shadow).__name__}")

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Single V-JEPA 2.1 training step.

        Returns scalar L_dense = L_predict + L_ctx.
        """
        # x-encoder: context features on masked input
        x_feat = self.encode_x(x)

        # y-encoder: target features on clean input (stop-grad)
        y_feat = self.encode_y(y)

        # Predictor: predict target features from context + mask
        pred = self.predictor(x_feat, mask)

        # Resize pred to match y_feat spatial dims if needed
        if pred.shape != y_feat.shape:
            pred = nn.functional.interpolate(
                pred,
                size=y_feat.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        # Dense predictive loss (L_predict + L_ctx)
        return self.loss_fn(pred, y_feat, mask)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        x, y, mask = batch
        loss = self(x, y, mask)
        if self._trainer is not None:
            self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0) -> None:  # noqa: ANN001
        """Update EMA after the optimizer step (V-JEPA 2.1 protocol)."""
        self.ema_encoder.update()

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        x, y, mask = batch
        loss = self(x, y, mask)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        x, y, mask = batch
        loss = self(x, y, mask)
        self.log("test/loss", loss, sync_dist=True)
        return loss

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["ema_state_dict"] = self.ema_encoder.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state = checkpoint.get("ema_state_dict")
        if state is not None:
            self.ema_encoder.load_state_dict(state)

    def configure_optimizers(self):
        """AdamW with linear warmup + constant schedule (Appendix A)."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            return 1.0  # constant after warmup

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [scheduler]

    def setup(self, stage: str | None = None) -> None:
        """Ensure EMA shadow matches encoder at the start of training.

        Skipped on resume: ``on_load_checkpoint`` restores the shadow then.
        """
        trainer = self._trainer
        resumed = trainer is not None and getattr(trainer, "ckpt_path", None) not in (None, "")
        if stage == "fit" and not resumed:
            self.ema_encoder = EMAEncoder(self.encoder, decay=self.ema_decay)
