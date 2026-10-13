"""Lightning modules: supervised classification shell around the backbones.

``ClassificationLightningModule`` holds the shared train/val/test step and
AdamW wiring. ``BackboneClassifier`` is the single supervised shell for both
backbones: the 2D image backbone on (B, C, H, W) and the causal video encoder
on (B, C, T, H, W). One class because the training logic is identical — only
the backbone (injected, never built here) differs.

Dependency injection: the wrapper *receives* its backbone as an argument.
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
from convgamer.models.io import StepOutput, StreamCache


def _cls_metrics(logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    loss = nn.functional.cross_entropy(logits, y)
    acc = (logits.argmax(dim=1) == y).float().mean()
    return loss, acc


class ClassificationLightningModule(pl.LightningModule):
    """Shared train/val/test step and AdamW wiring for classifiers."""

    lr: float
    weight_decay: float

    def __init__(
        self,
        model: nn.Module,
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

    def _step(self, batch, stage: str) -> torch.Tensor:
        try:
            x, y = batch
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"BackboneClassifier expects (x, y) pairs; got batch of type "
                f"{type(batch).__name__} (for (context, target, mask) triplets use "
                f"ConvGamerVJEPAModel): {e}"
            ) from e
        loss, acc = _cls_metrics(self(x), y)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
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
    """Return a mutable copy of the ``model`` hyper-parameter mapping.

    Trusted checkpoints only: Lightning stores non-tensor hparams that
    require ``weights_only=False`` to read, so load once that way.
    """
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

    try:
        module_name, _, cls_name = target.rpartition(".")
        cls = getattr(import_module(module_name), cls_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"cannot resolve backbone target {target!r}: {e}") from e
    try:
        return cls(**params)
    except TypeError as e:
        raise ValueError(f"cannot build backbone {target!r} with {sorted(params)}: {e}") from e


class BackboneClassifier(ClassificationLightningModule):
    """Single supervised shell for the image and video backbones.

    Batch is ``(x, y)`` with ``x`` shaped (B, C, H, W) for
    :class:`InceptionNeXtEncoder` or (B, C, T, H, W) for
    :class:`ConvGamerEncoder`. Extra forward kwargs (e.g.
    ``return_sequence``) pass straight through to the backbone.

    The video-only delegates (``forward_features``/``init_state``/``step``)
    require a :class:`ConvGamerEncoder` and fail loudly otherwise. The
    classification head always lives on the backbone (foundation models ship
    headless and attach one via ``add_classification_head``) — never here.
    """

    model: InceptionNeXtEncoder | ConvGamerEncoder  # type: ignore[bad-override-mutable-attribute]

    def __init__(
        self,
        model: InceptionNeXtEncoder | ConvGamerEncoder,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        model_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(model, lr=lr, weight_decay=weight_decay, model_kwargs=model_kwargs)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.model(x, **kwargs)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Delegate to the backbone's feature extractor."""
        return self.model.forward_features(x)

    def _video_model(self) -> ConvGamerEncoder:
        model = self.model
        if not isinstance(model, ConvGamerEncoder):
            raise TypeError(f"streaming requires ConvGamerEncoder, got {type(model).__name__}")
        return model

    def init_state(self, batch_size: int = 1, height: int = 32, width: int = 32) -> StreamCache:
        """Delegate to the video encoder's streaming cache."""
        return self._video_model().init_state(batch_size, height, width)

    def step(self, x_t: torch.Tensor, state: StreamCache) -> StepOutput:
        """Stream one frame through the full video pipeline."""
        return self._video_model().step(x_t, state)
