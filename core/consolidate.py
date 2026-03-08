import numpy as np
from .utils import TracerContext, compute_importance
from .hypothesis_set import Hypothesis, WorkingBelief


CONSOLIDATE_PROMPT = """
Given the 
"""


def compute_importance(conversation_length: int, entropy: float) -> float:
    g = 1 - np.exp(-conversation_length)
    h = 1 - entropy
    return g * h


