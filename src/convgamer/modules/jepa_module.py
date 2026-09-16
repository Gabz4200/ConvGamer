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

from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch

from convgamer.models.jepa import EMAEncoder, JEPALoss, VJEPAPredictor

if TYPE_CHECKING:
    from convgamer.models.convgamer.encoder import ConvGamerEncoder


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
        shadow = self.ema_encoder.shadow_module
        forward_maps = getattr(shadow, "forward_feature_maps", None)
        if not callable(forward_maps):
            raise TypeError(
                "EMA shadow module must expose forward_feature_maps(y); "
                f"got {type(shadow).__name__}"
            )
        out = forward_maps(y)
        if not isinstance(out, torch.Tensor):
            raise TypeError(
                f"EMA shadow forward_feature_maps must return a Tensor; got {type(out).__name__}"
            )
        return out

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

        if pred.shape != y_feat.shape:
            raise ValueError(
                f"Predictor output {tuple(pred.shape)} must match target {tuple(y_feat.shape)}"
            )

        # Dense predictive loss (L_predict + L_ctx)
        return self.loss_fn(pred, y_feat, mask)

    def training_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:  # noqa: ARG002
        x, y, mask = batch
        loss = self(x, y, mask)
        # V-JEPA 2.1 ramps lambda over early epochs: global_step survives
        # resume via the checkpoint, unlike a manual Python counter.
        self.loss_fn.set_step(int(self.global_step) + 1)
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
        self._sync_ema_shadow()

    def configure_optimizers(self):  # type: ignore[override]
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
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def _sync_ema_shadow(self) -> None:
        """Move the non-module EMA shadow onto the online encoder's device/dtype."""
        shadow = self.ema_encoder.shadow_module
        ref = next(self.encoder.parameters(), None)
        if shadow is None or ref is None:
            return
        if next(shadow.parameters(), None) is None:
            return
        shadow.to(device=ref.device, dtype=ref.dtype)

    def setup(self, stage: str | None = None) -> None:
        """Ensure EMA shadow matches encoder at the start of training.

        Skipped on resume: ``on_load_checkpoint`` restores the shadow then.
        """
        trainer = self._trainer
        resumed = trainer is not None and getattr(trainer, "ckpt_path", None) not in (None, "")
        if stage == "fit" and not resumed:
            self.ema_encoder = EMAEncoder(self.encoder, decay=self.ema_decay)
        self._sync_ema_shadow()
