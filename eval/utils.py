from model import embed, EmbedConfig
import numpy as np
from typing import List

def text_similarity(text1: str, text2: str, embed_cfg: EmbedConfig, embedding_prompt: str | None = None) -> float:
    if embedding_prompt:
        text1 = f"{embedding_prompt}\n{text1}"
        text2 = f"{embedding_prompt}\n{text2}"
    vec = embed([text1, text2], embed_cfg=embed_cfg)
    return np.dot(vec[0], vec[1])

def relative_similarity_score(adapted: str, candidates: List[str], chosen_idx: int, embed_cfg: EmbedConfig, embedding_prompt: str | None = None) -> float:
    if embedding_prompt:
        adapted = f"{embedding_prompt}\n{adapted}"
        candidates = [f"{embedding_prompt}\n{c}" for c in candidates]
    vec = embed([adapted] + candidates, embed_cfg=embed_cfg)
    adapted_vec = vec[0]
    candidate_vecs = vec[1:]
    similarities = np.dot(candidate_vecs, adapted_vec)
    chosen_similarity = similarities[chosen_idx]
    rejected_similarities = np.delete(similarities, chosen_idx)
    return chosen_similarity - np.max(rejected_similarities)
