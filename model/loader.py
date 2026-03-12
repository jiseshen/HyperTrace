from .base import BaseLM, GenerationConfig
from .openai_model import OpenAIModel
from typing import Optional


def load_model(backend: str, default_cfg: Optional[GenerationConfig] = None) -> BaseLM:
    if backend == "openai":
        return OpenAIModel(default_cfg=default_cfg)
    # TODO: Add vLLM and Gemini backends with the same generate/async_generate contract.
    else:
        raise ValueError(f"Unsupported backend '{backend}'. Supported backends: openai.")
