import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class EmbedConfig:
    backend: str = "openai"
    model: str = "text-embedding-3-small"
    dim: int = 1536


_client, _backend = None, None

def get_client(backend: str = "openai"):
    global _client, _backend
    if _client is None or _backend != backend:
        if backend == "openai":
            from openai import OpenAI
            _client = OpenAI()
            _backend = backend
        elif backend == "gemini":
            from google import genai
            _client = genai.Client()
            _backend = backend
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
    if embed_cfg.backend == "openai":
        client = get_client(embed_cfg.backend)
        response = client.embeddings.create(
            model=embed_cfg.model,
            input=text
        )
        vec = np.array([d.embedding for d in sorted(response.data, key=lambda x: x.index)], dtype=np.float32)
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
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
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
        return vec
        
    else:
        raise ValueError(f"Unknown embedding backend: {embed_cfg.backend}")
