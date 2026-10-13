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

Target-encoder maintenance (EMA update) is not this module's job: attach
``convgamer.callbacks.ema_update.EMAUpdateCallback`` to the trainer and it
drives ``self.ema_encoder.update()`` after every optimizer step.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch

from convgamer.models.jepa import JEPALoss, VJEPAPredictor
from convgamer.models.protocols import DenseFeatureEncoder

from .ema import EMAEncoder


class ConvGamerVJEPAModel(pl.LightningModule):
    """LightningModule for V-JEPA 2.1 pretraining on ConvGamer.

    Args:
        encoder: ConvGamerEncoder (online, trainable). Classification head
            should be ``nn.Identity`` (foundation model seam).
        predictor: VJEPAPredictor for dense mask-token prediction.
        loss: JEPALoss computing L_predict + L_ctx.
        ema_decay: EMA coefficient for target encoder (default 0.99925).
        lr: Peak learning rate.
        weight_decay: AdamW weight decay (applied to 2D+ params only;
            norms/bias/mask token are excluded).
        warmup_steps: Linear warmup steps.
        total_steps: Kept for config compat; the schedule is warmup-then-
            constant so this is currently unused.
    """

    def __init__(
        self,
        encoder: DenseFeatureEncoder,
        predictor: VJEPAPredictor,
        loss: JEPALoss,
        ema_decay: float = 0.99925,
        lr: float = 5.25e-4,
        weight_decay: float = 0.04,
        warmup_steps: int = 12_000,
        total_steps: int = 135_000,
        betas: tuple[float, float] = (0.9, 0.999),
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.loss_fn = loss
        self.ema_decay = ema_decay
        self.lr: float = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.betas = (float(betas[0]), float(betas[1]))
        self._ema_initialized = False
        enc_dim = getattr(encoder, "level_dim", None)
        if enc_dim is not None and predictor.feature_dim != enc_dim:
            raise ValueError(
                f"predictor feature_dim={predictor.feature_dim} must match "
                f"encoder level_dim={enc_dim}"
            )
        if predictor.num_outputs != 4:
            raise ValueError(
                f"predictor num_outputs={predictor.num_outputs} must equal encoder levels (4)"
            )

        # EMA target encoder (shadow copy of encoder)
        self.ema_encoder = EMAEncoder(encoder, decay=ema_decay)

        # Modules aren't config values, so exclude them from hparams.
        self.save_hyperparameters(ignore=["encoder", "predictor", "loss", "ema_encoder"])

    def encode_x(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the x-view (context encoder on masked input).

        Args:
            x: ``(B, C, T, H, W)`` masked standardized-RGB video.

        Returns:
            Encoder level pyramid ``(B, 4, P, T', H', W')``.
        """
        return self.encoder.forward_feature_maps(x)

    @torch.no_grad()
    def encode_y(self, y: torch.Tensor) -> torch.Tensor:
        """Encode the y-view (target encoder on clean input, no grad).

        The shadow is a deepcopy of the injected encoder, so the checker
        already knows it satisfies the dense-feature seam — no runtime probe.

        Args:
            y: ``(B, C, T, H, W)`` clean standardized-RGB video.

        Returns:
            Target level pyramid ``(B, 4, P, T', H', W')``.
        """
        return self.ema_encoder.shadow_module.forward_feature_maps(y)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Single V-JEPA 2.1 training step.

        Returns scalar L_dense = L_predict + L_ctx, averaged over levels.
        """
        # x-encoder: level pyramid on masked input, fused for the predictor
        x_pyramid = self.encode_x(x)
        x_fused = self.encoder.fuse_pyramid(x_pyramid)

        # y-encoder: target level pyramid on clean input (stop-grad)
        y_pyramid = self.encode_y(y)

        # Predictor: per-level predictions from fused context + mask
        pred = self.predictor(x_fused, mask)

        # Dense predictive loss (L_predict + L_ctx), averaged over levels
        return self.loss_fn(pred, y_pyramid, mask)

    def training_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:  # noqa: ARG002
        x, y, mask = batch
        loss = self(x, y, mask)
        # V-JEPA 2.1 ramps lambda over early epochs: global_step survives
        # resume via the checkpoint, unlike a manual Python counter.
        self.loss_fn.set_step(int(self.global_step) + 1)
        if self._trainer is not None:
            self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def _val_test_loss(self, batch) -> torch.Tensor:
        """Loss with the warmup schedule frozen so val/test are comparable."""
        x, y, mask = batch
        prev = int(self.loss_fn._step_int)
        try:
            self.loss_fn.set_step(max(prev, self.loss_fn.lambda_warmup_steps))
            return self(x, y, mask)
        finally:
            self.loss_fn.set_step(prev)

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        loss = self._val_test_loss(batch)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        loss = self._val_test_loss(batch)
        self.log("test/loss", loss, sync_dist=True)
        return loss

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["ema_state_dict"] = self.ema_encoder.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state = checkpoint.get("ema_state_dict")
        if state is not None:
            self.ema_encoder.load_state_dict(state)
        self._ema_initialized = True
        self._sync_ema_shadow()

    def configure_optimizers(self):  # type: ignore[override]
        """AdamW with linear warmup + constant schedule (Appendix A)."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            # 1-D params (norms/biases) and the zero-init mask token get no decay.
            if p.ndim < 2 or name.endswith("mask_token"):
                no_decay.append(p)
            else:
                decay.append(p)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.lr,
            betas=self.betas,
        )

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return (step + 1) / max(1, self.warmup_steps)
            return 1.0  # constant after warmup

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int = 0):
        out = super().transfer_batch_to_device(batch, device, dataloader_idx)
        self._sync_ema_shadow()
        return out

    def _sync_ema_shadow(self) -> None:
        """Move the non-module EMA shadow onto the online encoder's device.

        Dtype is left alone: the EMA target stays fp32 even under AMP.
        """
        shadow = self.ema_encoder.shadow_module
        ref = next(self.encoder.parameters(), None)
        if shadow is None or ref is None or next(shadow.parameters(), None) is None:
            return
        if next(shadow.parameters()).device != ref.device:
            shadow.to(device=ref.device)

    def setup(self, stage: str | None = None) -> None:
        """Ensure EMA shadow matches encoder at the start of training.

        Never wipes an existing shadow on re-fit; resume restores it via
        ``on_load_checkpoint``.
        """
        if stage == "fit" and not self._ema_initialized:
            # First fit only: seed the shadow from the online weights.
            self.ema_encoder = EMAEncoder(self.encoder, decay=self.ema_decay)
            self._ema_initialized = True
        self._sync_ema_shadow()
