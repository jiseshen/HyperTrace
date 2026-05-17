from model import embed, EmbedConfig
import numpy as np
from typing import List

EMBEDDING_CONTEXT_TOKEN_LIMIT = 8000


class EmbeddingInputTooLongError(ValueError):
    pass


def _check_embedding_inputs(texts: List[str]) -> None:
    too_long = [
        (idx, len(text.split()))
        for idx, text in enumerate(texts)
        if len(text.split()) > EMBEDDING_CONTEXT_TOKEN_LIMIT
    ]
    if too_long:
        details = ", ".join(f"input[{idx}]={n} tokens" for idx, n in too_long)
        raise EmbeddingInputTooLongError(
            f"Skipping embedding eval because input exceeds {EMBEDDING_CONTEXT_TOKEN_LIMIT} tokens: {details}"
        )


def text_similarity(text1: str, text2: str, embed_cfg: EmbedConfig, embedding_prompt: str | None = None) -> float:
    if embedding_prompt:
        text1 = f"{embedding_prompt}\n{text1}"
        text2 = f"{embedding_prompt}\n{text2}"
    _check_embedding_inputs([text1, text2])
    vec = embed([text1, text2], embed_cfg=embed_cfg)
    return float(np.dot(vec[0], vec[1]).item())


def candidate_similarity_scores(adapted: str, candidates: List[str], embed_cfg: EmbedConfig, embedding_prompt: str | None = None) -> List[float]:
    if embedding_prompt:
        adapted = f"{embedding_prompt}\n{adapted}"
        candidates = [f"{embedding_prompt}\n{c}" for c in candidates]
    texts = [adapted] + candidates
    _check_embedding_inputs(texts)
    vec = embed(texts, embed_cfg=embed_cfg)
    adapted_vec = vec[0]
    candidate_vecs = vec[1:]
    similarities = np.dot(candidate_vecs, adapted_vec)
    return [float(score.item()) for score in similarities]


def relative_similarity_score(adapted: str, candidates: List[str], chosen_idx: int, embed_cfg: EmbedConfig, embedding_prompt: str | None = None) -> float:
    similarities = np.array(candidate_similarity_scores(adapted, candidates, embed_cfg, embedding_prompt))
    chosen_similarity = similarities[chosen_idx]
    rejected_similarities = np.delete(similarities, chosen_idx)
    return float((chosen_similarity - np.max(rejected_similarities)).item())
