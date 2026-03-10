from core import embed
import numpy as np
from typing import List


def text_similarity(text1: str, text2: str) -> float:
    vec = embed([text1, text2])
    return np.dot(vec[0], vec[1])

def relative_similarity_score(adapted: str, candidates: List[str], chosen_idx: int) -> float:
    vec = embed([adapted] + candidates)
    adapted_vec = vec[0]
    candidate_vecs = vec[1:]
    similarities = np.dot(candidate_vecs, adapted_vec)
    chosen_similarity = similarities[chosen_idx]
    rejected_similarities = np.delete(similarities, chosen_idx)
    return chosen_similarity - np.max(rejected_similarities)
