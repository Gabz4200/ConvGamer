"""HuggingFace ``PretrainedConfig`` for ConvGamer (InceptionNeXt).

``model_type`` is ``convgamer`` — verified non-colliding with existing
HF model types.
"""

from transformers import PretrainedConfig


class ConvGamerConfig(PretrainedConfig):
    model_type = "convgamer"
    config_class = "ConvGamerConfig"

    def __init__(
        self,
        hidden_dim: int = 96,
        num_layers: int | list[int] = 3,
        num_classes: int = 1000,
        input_dim: int = 3,
        layer_scale_init: float = 1e-6,
        mlp_ratios: tuple[int, int, int, int] | list[int] = (4, 4, 4, 3),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.layer_scale_init = layer_scale_init
        self.mlp_ratios = mlp_ratios
