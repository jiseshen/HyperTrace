from .base import PromptSet, base_prompts
from .personamem_adapter import personamem_prompts
from .prism_adapter import prism_prompts


def load_prompt_adapter(name: str) -> PromptSet:
    normalized = name.lower()
    if normalized in {"prism", "base"}:
        return prism_prompts()
    if normalized in {"personamem_v2", "personamem-v2", "personamem"}:
        return personamem_prompts()
    raise ValueError(f"Unsupported prompt adapter: {name}")


__all__ = ["PromptSet", "base_prompts", "prism_prompts", "personamem_prompts", "load_prompt_adapter"]
