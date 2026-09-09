from __future__ import annotations

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel

from convgamer.models.registry import get_model

from .configuration_convgamer import ConvGamerConfig


class ConvGamerModel(PreTrainedModel):
    config_class = ConvGamerConfig
    base_model_prefix = "convgamer"

    def __init__(self, config: ConvGamerConfig):
        super().__init__(config)
        self.native = get_model(
            "encoder",
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
            layer_scale_init=config.layer_scale_init,
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> BaseModelOutput | tuple:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        hidden = inputs_embeds if inputs_embeds is not None else input_ids
        if hidden is None:
            raise ValueError("Specify input_ids or inputs_embeds")
        out = self.native(hidden)
        if return_dict:
            return BaseModelOutput(last_hidden_state=out)
        return (out,)

    def get_input_embeddings(self) -> nn.Module:
        return nn.Identity()
