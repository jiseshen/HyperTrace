from .base import BaseLM, GenerationConfig
from .openai_model import OpenAIModel
from .openrouter_model import OpenRouterModel
from typing import Optional


def load_model(backend: str, default_cfg: Optional[GenerationConfig] = None) -> BaseLM:
    if backend == "openai":
        return OpenAIModel(default_cfg=default_cfg)
    elif backend == "openrouter":
        return OpenRouterModel(default_cfg=default_cfg)
    else:
        raise ValueError(f"Unsupported backend '{backend}'. Supported backends: openai, openrouter.")
