from dataclasses import dataclass
from typing import Optional
import numpy as np
from model import BaseLM, GenerationConfig
from .hypothesis_set import HypothesisSet, WorkingBelief


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
    
    
def compute_importance(conversation_length: int, entropy: float) -> float:
    g = 1 - np.exp(-conversation_length)
    h = 1 - entropy
    return g * h