import contextlib

from transformers import AutoConfig, AutoModel

from .configuration_convgamer import ConvGamerConfig
from .modeling_convgamer import ConvGamerModel


def register_auto_classes() -> None:
    with contextlib.suppress(ValueError, AttributeError):
        AutoConfig.register("convgamer", ConvGamerConfig)
    with contextlib.suppress(ValueError, AttributeError):
        AutoModel.register(ConvGamerConfig, ConvGamerModel)


register_auto_classes()
