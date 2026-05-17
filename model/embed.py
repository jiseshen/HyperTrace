import numpy as np
from dataclasses import dataclass
import time
from typing import List

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

@dataclass
class EmbedConfig:
    backend: str = "openai"
    model: str = "text-embedding-3-small"
    dim: int = 1536
    base_url: str | None = None
    max_retries: int = 3
    retry_delay: float = 0.5


_client, _backend = None, None

def get_client(backend: str = "openai", base_url: str | None = None):
    global _client, _backend
    cache_key = f"{backend}:{base_url or ''}"
    if _client is None or _backend != cache_key:
        if backend == "openai":
            from openai import OpenAI
            _client = OpenAI()
            _backend = cache_key
        elif backend == "openrouter":
            import os
            from openai import OpenAI
            _client = OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY", "empty"),
                base_url=base_url or os.getenv("OPENROUTER_API_BASE", OPENROUTER_BASE_URL),
            )
            _backend = cache_key
        elif backend == "gemini":
            from google import genai
            _client = genai.Client()
            _backend = cache_key
    return _client

_model = None

def get_encoder(model_name: str):
    global _model
    if _model is None or _model.model_card_data.model_id != model_name:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(model_name)
    return _model

def embed(
    text: str | List[str],
    *,
    embed_cfg: EmbedConfig
) -> np.ndarray:
    """
    Embed an experience into a vector.

    Args:
        text: String or list of strings to embed.
        embed_cfg: Configuration for the embedding model.

    Returns:
        2D numpy array of shape (N, D), normalized.
    """
    if isinstance(text, str):
        text = [text]
    if len(text) == 0:
        return np.empty((0, embed_cfg.dim), dtype=np.float32)
    if embed_cfg.backend in ("openai", "openrouter"):
        from openai import APIError, RateLimitError

        client = get_client(embed_cfg.backend, embed_cfg.base_url)
        last_error = None
        for attempt in range(embed_cfg.max_retries):
            try:
                response = client.embeddings.create(
                    model=embed_cfg.model,
                    input=text
                )
                if not response.data:
                    raise ValueError("No embedding data received")
                if len(response.data) != len(text):
                    raise ValueError(
                        "Embedding response length mismatch "
                        f"(inputs={len(text)}, outputs={len(response.data)})"
                )
                break
            except (APIError, RateLimitError, ValueError) as exc:
                last_error = exc
                if attempt == embed_cfg.max_retries - 1:
                    raise RuntimeError(
                        "Embedding request failed after retries "
                        f"(backend={embed_cfg.backend}, model={embed_cfg.model}, "
                        f"inputs={len(text)}, max_retries={embed_cfg.max_retries}, "
                        f"texts={text!r})"
                    ) from last_error
                time.sleep(embed_cfg.retry_delay)
        vec = np.array([d.embedding for d in sorted(response.data, key=lambda x: x.index)], dtype=np.float32)
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-14)
        return vec

    elif "transformer" in embed_cfg.backend:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(embed_cfg.model)
        vec = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec

    elif embed_cfg.backend == "gemini":
        client = get_client(embed_cfg.backend)
        resp = client.models.embed_content(
            model=embed_cfg.model,
            contents=text
        )
        vec = np.stack([np.array(item.values, dtype=np.float32) for item in resp.embeddings])
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-14)
        return vec
        
    else:
        raise ValueError(f"Unknown embedding backend: {embed_cfg.backend}")
