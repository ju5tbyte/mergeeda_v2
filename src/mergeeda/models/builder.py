"""Model factory for building model instances from configuration.

This module provides a registry pattern for instantiating models based on
configuration. New models should be registered in MODEL_REGISTRY.
"""

from typing import Union

from omegaconf import DictConfig

from .qwen_vl_model import QwenVLModel
from .qwen_vl_finetuned_model import QwenVLFinetunedModel

MODEL_REGISTRY = {
    "QwenVLModel": QwenVLModel,
    "QwenVLFinetunedModel": QwenVLFinetunedModel,
}


def build_model(
    model_cfg: DictConfig,
) -> Union[QwenVLModel, QwenVLFinetunedModel]:
    """Build and return a model instance from configuration."""
    model_name = model_cfg.name
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry")

    return MODEL_REGISTRY[model_name](**model_cfg.params)
