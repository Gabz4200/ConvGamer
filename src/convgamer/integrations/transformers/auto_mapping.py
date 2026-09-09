from transformers import AutoConfig, AutoModel

from .configuration_convgamer import ConvGamerConfig
from .modeling_convgamer import ConvGamerModel


def register_auto_classes() -> None:
    AutoConfig.register("convgamer", ConvGamerConfig)
    AutoModel.register(ConvGamerConfig, ConvGamerModel)


register_auto_classes()
