import numpy as np
from dataclasses import dataclass
import time
import os
import threading
from .credentials import require_api_key
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


_local = threading.local()


def get_client(backend: str = "openai", base_url: str | None = None):
    """Keep each worker's clients separate when different backends run concurrently."""
    if backend not in ("openai", "openrouter", "gemini"):
        raise ValueError(f"Unsupported embedding client backend: {backend}")
    if not hasattr(_local, "clients"):
        _local.clients = {}
    cache_key = (backend, base_url)
    if cache_key not in _local.clients:
        if backend in ("openai", "openrouter"):
            from openai import OpenAI
            variable = "OPENAI_API_KEY" if backend == "openai" else "OPENROUTER_API_KEY"
            default_url = "https://api.openai.com/v1" if backend == "openai" else OPENROUTER_BASE_URL
            url_variable = "OPENAI_API_BASE" if backend == "openai" else "OPENROUTER_API_BASE"
            client = OpenAI(
                api_key=require_api_key(variable),
                base_url=base_url or os.getenv(url_variable, default_url),
            )
        else:
            from google import genai
            key = os.getenv("GOOGLE_API_KEY") or require_api_key("GEMINI_API_KEY")
            client = genai.Client(api_key=key)
        _local.clients[cache_key] = client
    return _local.clients[cache_key]


def close_embedding_clients():
    for client in getattr(_local, "clients", {}).values():
        client.close()
    _local.clients = {}


_encoder = None
_encoder_name = None
_encoder_lock = threading.Lock()


def get_encoder(model_name: str):
    global _encoder, _encoder_name
    with _encoder_lock:
        if _encoder is None or _encoder_name != model_name:
            from sentence_transformers import SentenceTransformer
            _encoder = SentenceTransformer(model_name)
            _encoder_name = model_name
        return _encoder


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
    if embed_cfg.backend not in ("openai", "openrouter", "gemini", "transformer"):
        raise ValueError(f"Unknown embedding backend: {embed_cfg.backend}")
    if embed_cfg.max_retries < 1:
        raise ValueError("Embedding max_retries must be positive")
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
                        f"inputs={len(text)}, max_retries={embed_cfg.max_retries})"
                    ) from last_error
                time.sleep(embed_cfg.retry_delay)
        vec = np.array([d.embedding for d in sorted(response.data, key=lambda x: x.index)], dtype=np.float32)
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-14)
        return vec

    elif embed_cfg.backend == "transformer":
        model = get_encoder(embed_cfg.model)
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
