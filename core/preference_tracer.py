from .utils import TracerConfig, TracerContext
from .hypothesis_set import Hypothesis, HypothesisSet, WorkingBelief, RepoConfig
from data import Conversation, Turn, UserData
from model import BaseModel, GenerationConfig
from .initialize import initialize_hypothesis
from .branch import branch_hypotheses

class PreferenceTracer:
    def __init__(
        self, 
        model: BaseModel, 
        generation_cfg: GenerationConfig, 
        tracer_cfg: TracerConfig, 
        repo_cfg: RepoConfig
    ):
        self.model = model
        self.base_generation_config = generation_cfg
        self.tracer_config = tracer_cfg
        self.hypothesis_set = HypothesisSet(repo_config=repo_cfg)
        self.context = TracerContext(
            model=model,
            hypothesis_set=self.hypothesis_set,
            tracer_config=tracer_cfg,
            generation_config=generation_cfg
        )
    
    def trace(self, user_data: UserData):
        for conversation in user_data.conversations:
            initialized = False
            conversation_history = []
            for turn in conversation.turns:
                skip = True
                conversation_history.append(turn)
                if not initialized:
                    belief = initialize_hypothesis(conversation_history=conversation_history, context=self.context)
                    if belief is not None:
                        self.context.current_belief = belief
                        initialized = True
                        skip = False
                else:
                    updated_belief = branch_hypotheses(conversation_history=conversation_history, context=self.context)
                    if updated_belief is not None:
                        self.context.current_belief = updated_belief
                        skip = False