from base import BaseLM, GenerationConfig
from openai_model import OpenAIModel
from typing import Optional


def load_model(backend: str, generation_cfg: Optional[GenerationConfig] = None) -> BaseLM:
    if backend == "openai":
        return OpenAIModel(default_cfg=generation_cfg)
    # TODO: vLLM Chat Completion API w/ formatting & Gemini API
    else:
        raise ValueError(f"Unsupported backend: {backend}")