from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional
from model import BaseLM, GenerationConfig
from model.base import GenerationOverrides
from prompt import PromptSet, prism_prompts

if TYPE_CHECKING:
    from .hypothesis_set import HypothesisSet, WorkingBelief

@dataclass
class TaskOverride:
    model: Optional[str] = None
    reasoning_effort: Optional[str] = None
    max_tokens_extra: Optional[int] = None


@dataclass
class OverrideConfig:
    initialize_override: TaskOverride = field(default_factory=TaskOverride)
    branch_override: TaskOverride = field(default_factory=TaskOverride)
    perturb_override: TaskOverride = field(default_factory=TaskOverride)
    response_override: TaskOverride = field(default_factory=TaskOverride)
    skip_override: TaskOverride = field(default_factory=TaskOverride)
    preprocess_override: TaskOverride = field(default_factory=TaskOverride)
    filter_override: TaskOverride = field(default_factory=TaskOverride)
    axis_override: TaskOverride = field(default_factory=TaskOverride)
    merge_override: TaskOverride = field(default_factory=TaskOverride)
    summary_override: TaskOverride = field(default_factory=TaskOverride)
    profile_override: TaskOverride = field(default_factory=TaskOverride)
    prediction_override: TaskOverride = field(default_factory=TaskOverride)

    def __post_init__(self):
        for name in (
            "initialize_override",
            "branch_override",
            "perturb_override",
            "response_override",
            "skip_override",
            "preprocess_override",
            "filter_override",
            "axis_override",
            "merge_override",
            "summary_override",
            "profile_override",
            "prediction_override",
        ):
            value = getattr(self, name)
            if isinstance(value, dict):
                setattr(self, name, TaskOverride(**value))

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
    override: OverrideConfig = field(default_factory=OverrideConfig)

@dataclass
class TracerContext:
    model: BaseLM
    hypothesis_set: "HypothesisSet"
    current_belief: Optional["WorkingBelief"] = None
    tracer_config: TracerConfig = field(default_factory=TracerConfig)
    generation_config: Optional[GenerationConfig] = field(default_factory=GenerationConfig)
    prompts: PromptSet = field(default_factory=prism_prompts)
    
    @property
    def belief(self) -> "WorkingBelief":
        assert self.current_belief is not None, "Current belief is not initialized"
        return self.current_belief
    
    def update_belief(self, new_belief: "WorkingBelief"):
        self.current_belief = new_belief

    def get_generation_overrides(self, task_name: str) -> GenerationOverrides:
        task_override = getattr(self.tracer_config.override, task_name, None)
        overrides: GenerationOverrides = {}
        if task_override is None:
            return overrides
        if task_override.model:
            overrides["model"] = task_override.model
        if task_override.reasoning_effort:
            overrides["reasoning_effort"] = task_override.reasoning_effort
        if task_override.max_tokens_extra is not None:
            overrides["max_tokens_extra"] = task_override.max_tokens_extra
        return overrides
