from .encoder import Encoder  # noqa: F401 — triggers @register_model
from .registry import get_model, register_model

__all__ = ["register_model", "get_model", "Encoder"]
