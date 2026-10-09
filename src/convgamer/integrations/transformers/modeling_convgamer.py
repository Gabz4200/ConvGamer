"""HuggingFace wrapper around the ConvGamer *image* backbone.

This wraps :class:`InceptionNeXtEncoder` (single images), not the video
foundation encoder — the export scripts always transfer frame-encoder
weights, hence the image-explicit name.
"""

from __future__ import annotations

import torch
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel

from convgamer.models.registry import get_model

from .configuration_convgamer import ConvGamerImageConfig


class ConvGamerImageModel(PreTrainedModel):
    config_class = ConvGamerImageConfig
    base_model_prefix = "convgamer"

    def __init__(self, config: ConvGamerImageConfig):
        super().__init__(config)
        self.native = get_model(
            "InceptionNeXtEncoder",
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
            layer_scale_init=config.layer_scale_init,
            mlp_ratios=tuple(config.mlp_ratios),
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> BaseModelOutput | tuple:
        """Encode images; returns logits with a head, pooled features headless."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        hidden = (
            inputs_embeds
            if inputs_embeds is not None
            else (pixel_values if pixel_values is not None else input_ids)
        )
        if hidden is None:
            raise ValueError("Specify input_ids, pixel_values, or inputs_embeds")
        out = self.native(hidden)
        if return_dict:
            return BaseModelOutput(last_hidden_state=out)
        return (out,)
