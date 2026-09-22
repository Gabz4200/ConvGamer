"""Lightning modules for InceptionNeXt (arXiv:2303.16900) and ConvGamer.

``InceptionNeXtModule`` trains the 2D image backbone on (B, C, H, W).
``ConvGamerModel`` trains the causal video encoder on (B, C, T, H, W).
Both share ``ClassificationLightningModule`` for the step/optimizer logic.

Dependency injection: each wrapper *receives* its backbone as an argument.
Nothing here reads a config object; the composition root in
``convgamer.scripts.common`` builds backbones from the config and injects them.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn

from convgamer.models.convgamer.encoder import ConvGamerEncoder
from convgamer.models.inception_next import InceptionNeXtEncoder
from convgamer.models.io import StepOutput, StreamingState


def _cls_metrics(logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    loss = nn.functional.cross_entropy(logits, y)
    acc = (logits.argmax(dim=1) == y).float().mean()
    return loss, acc


class ClassificationLightningModule(pl.LightningModule):
    """Shared train/val/test step and AdamW wiring for classifiers."""

    lr: float
    weight_decay: float

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
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        *args: Any,
        map_location: Any = None,
        hparams_file: Any = None,
        strict: bool | None = None,
        weights_only: bool | None = None,
        **kwargs: Any,
    ):
        """Reconstruct the backbone from checkpoint metadata when needed.

        Checkpoints store the backbone's construction parameters under the
        ``model`` hyper-parameter key (with ``target`` selecting the encoder
        class).  Lightning passes that dict straight to the constructor as
        ``model``, which normally yields a dict instead of a real encoder;
        and an injected ``model=backbone`` overwrites that hyper-parameter so
        the constructor records an empty ``model`` mapping and the saved
        metadata is lost.

        This override normalizes both paths at the composition boundary only:
        build the backbone from the stored metadata when none is supplied, and
        forward ``model_kwargs`` so the constructor repopulates ``hparams["model"]``
        even when a backbone was injected.  The checkpoint format is unchanged.
        """
        # Checkpoints carry non-tensor hparams (AttributeDict), so the default
        # weights_only=True torch.load path cannot read them.
        if weights_only is None:
            weights_only = False
        model_hparams = _load_model_metadata(checkpoint_path)
        if isinstance(kwargs.get("model"), nn.Module) and kwargs.get("model_kwargs") is None:
            kwargs["model_kwargs"] = dict(model_hparams)
        elif "model" not in kwargs and model_hparams:
            target = model_hparams["target"]
            params = {k: v for k, v in model_hparams.items() if k != "target"}
            kwargs["model"] = _build_backbone(target, params)
            kwargs["model_kwargs"] = dict(model_hparams)
        return super().load_from_checkpoint(
            checkpoint_path,
            *args,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            weights_only=weights_only,
            **kwargs,
        )


def _load_model_metadata(checkpoint_path: Any) -> dict[str, Any]:
    """Return a mutable copy of the ``model`` hyper-parameter mapping."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hyper_parameters = checkpoint.get("hyper_parameters") or {}
    model_kwargs = hyper_parameters.get("model", {}) if isinstance(hyper_parameters, dict) else {}
    return dict(model_kwargs)


def _build_backbone(target: str, params: dict[str, Any]) -> nn.Module:
    """Instantiate a backbone from a class-path ``target`` and kwargs.

    ``get_model`` is keyed on the short registry name, so resolve the dotted
    path to its class and instantiate directly.
    """
    from importlib import import_module

    module_name, _, cls_name = target.rpartition(".")
    cls = getattr(import_module(module_name), cls_name)
    return cls(**params)


class InceptionNeXtModule(ClassificationLightningModule):
    """Pure InceptionNeXt LightningModule — matches paper §4.1.

    AdamW optimiser, CrossEntropyLoss on classification logits.  The
    optimiser hyper-parameters (``lr``, ``weight_decay``) are plain typed
    values — the paper's recipe (mirroring §4.1):

    - lr = 0.001 x batchsize/1024
    - weight_decay = 0.05
    - cosine decay schedule (configured on the trainer side)
    - 300 epochs (paper); reduced for dev via ``trainer.max_epochs``

    This module depends on the InceptionNeXt implementation only — it does
    not import or reference any ConvGamer-specific code.

    ``model_kwargs`` records the backbone's construction parameters so
    checkpoints stay inspectable and ``export-hf`` can rebuild the HF config;
    it is metadata, not a config object, and no config type crosses this
    boundary.
    """

    model: InceptionNeXtEncoder

    def __init__(
        self,
        model: InceptionNeXtEncoder,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        model_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.model = model
        self.lr: float = lr
        self.weight_decay: float = weight_decay
        self.save_hyperparameters(
            {"lr": lr, "weight_decay": weight_decay, "model": dict(model_kwargs or {})}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ConvGamerModel(ClassificationLightningModule):
    """ConvGamer video LightningModule — trains ``ConvGamerEncoder`` end to end.

    Batch is ``(video, label)`` with video shaped (B, C, T, H, W).
    Fully causal: pooled logits at frame ``t`` see only frames ``<= t``.
    See :class:`InceptionNeXtModule` for the ``model_kwargs`` contract.
    """

    model: ConvGamerEncoder

    def __init__(
        self,
        model: ConvGamerEncoder,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        model_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.model = model
        self.lr: float = lr
        self.weight_decay: float = weight_decay
        self.save_hyperparameters(
            {"lr": lr, "weight_decay": weight_decay, "model": dict(model_kwargs or {})}
        )

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        return self.model(x, return_sequence=return_sequence)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Delegate to the encoder's feature extractor."""
        return self.model.forward_features(x)

    def init_state(self, batch_size: int = 1, height: int = 32, width: int = 32) -> StreamingState:
        """Delegate to the encoder's streaming state."""
        return self.model.init_state(batch_size, height, width)

    def step(self, x_t: torch.Tensor, state: StreamingState) -> StepOutput:
        """Stream one frame through the full video pipeline."""
        return self.model.step(x_t, state)
