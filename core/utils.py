from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from model import BaseLM, GenerationConfig
from .hypothesis_set import HypothesisSet, WorkingBelief

_client = None

def get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI()
    return _client

_tokenizer, _encoder = None, None

def get_encoder(model_name: str):
    global _tokenizer, _encoder
    if _tokenizer is None or _encoder is None or _tokenizer.name_or_path != model_name:
        from transformers import AutoTokenizer, AutoModel
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _encoder = AutoModel.from_pretrained(model_name)
    return _tokenizer, _encoder

@dataclass
class TracerConfig:
    n_hypotheses: int = 5
    consolidate_alpha: float = 0.5        # The fraction of old priors to retain when consolidating
    similarity_threshold: float = 0.8     # Threshold for clustering hypotheses based on semantic similarity
    perturb_alpha: float = 0.3            # The fraction of total weight to split among new hypotheses when perturbing a cluster
    summary_threshold: float = 0.1
    max_history_turns: int = 3
    profile_top_p: float = 0.8
    

@dataclass
class TracerContext:
    model: BaseLM
    hypothesis_set: HypothesisSet
    current_belief: Optional[WorkingBelief] = None
    tracer_config: TracerConfig = TracerConfig()
    generation_config: Optional[GenerationConfig] = GenerationConfig()
    
    @property
    def belief(self) -> WorkingBelief:
        assert self.current_belief is not None, "Current belief is not initialized"
        return self.current_belief
    
    def update_belief(self, new_belief: WorkingBelief):
        self.current_belief = new_belief
        


def embed(
    text: str | List[str],
    *,
    backend: str = "openai",
    model: str = "text-embedding-3-small",
) -> np.ndarray:
    """
    Embed an experience into a vector.

    Args:
        text: String or list of strings to embed.
        backend: "openai" or "transformer".
        model: Embedding model name.

    Returns:
        2D numpy array of shape (N, D), normalized.
    """
    if isinstance(text, str):
        text = [text]
    if backend == "openai":
        client = get_client()
        response = client.embeddings.create(
            model=model,
            input=text
        )
        vec = np.array([d.embedding for d in sorted(response.data, key=lambda x: x.index)], dtype=np.float32)
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
        return vec

    elif "transformer" in backend:
        import torch
        tokenizer, encoder = get_encoder(model)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.inference_mode():
            outputs = encoder(**inputs)

        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        vec = pooled.cpu().numpy().astype(np.float32)
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
        return vec

    else:
        raise ValueError(f"Unknown embedding backend: {backend}")