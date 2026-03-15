from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional
from model import BaseLM, GenerationConfig

if TYPE_CHECKING:
    from .hypothesis_set import HypothesisSet, WorkingBelief

@dataclass
class TracerConfig:
    hierarchical: bool = True
    allow_expand: bool = False
    allow_skip: bool = True
    n_hypotheses: int = 5
    consolidate_alpha: float = 1        # The fraction of old priors to retain when consolidating
    similarity_threshold: float = 0.8     # Threshold for clustering hypotheses based on semantic similarity
    perturb_alpha: float = 0.3            # The fraction of total weight to split among new hypotheses when perturbing a cluster
    bradley_terry_temp: float = 1.0       # Temperature parameter for Bradley-Terry model when updating belief
    max_history_turns: int = 3
    profile_top_p: float = 0.8

@dataclass
class TracerContext:
    model: BaseLM
    hypothesis_set: "HypothesisSet"
    current_belief: Optional["WorkingBelief"] = None
    tracer_config: TracerConfig = field(default_factory=TracerConfig)
    generation_config: Optional[GenerationConfig] = field(default_factory=GenerationConfig)
    
    @property
    def belief(self) -> "WorkingBelief":
        assert self.current_belief is not None, "Current belief is not initialized"
        return self.current_belief
    
    def update_belief(self, new_belief: "WorkingBelief"):
        self.current_belief = new_belief