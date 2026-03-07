from .utils import TracerConfig, TracerContext
from .hypothesis_set import Hypothesis, HypothesisSet, WorkingBelief, RepoConfig
from data import Conversation, Turn, UserData
from model import BaseModel, GenerationConfig
from .preprocess import preprocess_candidates
from .initialize import initialize_hypothesis
from .branch import branch_hypotheses
from .filter import weight_hypothesis
from .perturb import perturb_hypotheses
from eval import predict_choice, profile_score, evaluate_generation

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
        self.results = []
    
    def trace(self, user_data: UserData):
        for conversation in user_data.conversations:
            results = []
            initialized = False
            conversation_history = []
            for turn in conversation.turns:
                conversation_history.append(turn)
                # Online Evaluation
                choice = predict_choice(conversation_history=conversation_history, current_turn=turn, context=self.context)
                generation = evaluate_generation(conversation_history=conversation_history, current_turn=turn, context=self.context)

                # Online Update
                candidates = preprocess_candidates(conversation_history=conversation_history, current_turn=turn, context=self.context)
                if candidates is None:
                    results.append({"turn": turn, "skipped": True})
                    continue
                if not initialized:
                    initialize_hypothesis(conversation_history=conversation_history, candidates=candidates, context=self.context)
                    initialized = True
                else:
                    branch_hypotheses(conversation_history=conversation_history, candidates=candidates, context=self.context)
                weight_hypothesis(conversation_history=conversation_history, candidates=candidates, context=self.context)
                if self.context.belief.ess() < self.tracer_config.n_hypotheses / 2:
                    similar_groups = self.context.belief.resample()
                    perturb_hypotheses(conversation_history=conversation_history, candidates=candidates, similar_groups=similar_groups, context=self.context)
                else:
                    similar_groups = self.context.belief.get_similarity_groups(threshold=self.tracer_config.similarity_threshold)
                    perturb_hypotheses(conversation_history=conversation_history, candidates=candidates, similar_groups=similar_groups, context=self.context)
                    