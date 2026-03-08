from typing import Optional
from .utils import TracerConfig, TracerContext
from .hypothesis_set import Hypothesis, HypothesisSet, WorkingBelief, RepoConfig
from data import Conversation, Turn, UserData
from model import BaseLM, GenerationConfig


from .preprocess import preprocess_candidates
from .initialize import initialize_hypothesis
from .branch import branch_hypotheses
from .filter import weight_hypothesis
from .perturb import perturb_hypotheses
from .summary import summarize_hypotheses, summarize_profile
from eval import predict_choice, profile_score, evaluate_generation


class PreferenceTracer:
    def __init__(
        self, 
        model: BaseLM,
        generation_cfg: GenerationConfig, 
        tracer_cfg: TracerConfig, 
        repo_cfg: RepoConfig,
        evaluation_model: Optional[BaseLM] = None,
        evaluation_cfg: Optional[GenerationConfig] = None
    ):
        self.model = model
        self.evaluation_model = evaluation_model or model
        self.base_generation_config = generation_cfg
        self.evaluation_config = evaluation_cfg or generation_cfg
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
                working_profile = summarize_hypotheses(conversation_history, self.context)
                # Online Evaluation
                choice = predict_choice(
                    model=self.evaluation_model, 
                    conversation_history=conversation_history, 
                    profile=working_profile, 
                    generation_cfg=self.evaluation_config
                )
                generation = evaluate_generation(
                    model=self.evaluation_model, 
                    conversation_history=conversation_history, 
                    profile=working_profile,
                    context=self.context
                )

                # Online Update
                candidates = preprocess_candidates(conversation_history, self.context)
                if candidates is None:
                    results.append({"turn": turn, "skipped": True})
                    continue
                if not initialized:
                    initialize_hypothesis(conversation_history, candidates, self.context)
                    initialized = True
                else:
                    branch_hypotheses(conversation_history, candidates, self.context)
                weight_hypothesis(conversation_history, candidates, self.context)
                if self.context.belief.ess() < self.tracer_config.n_hypotheses / 2:
                    similar_groups = self.context.belief.resample()
                    perturb_hypotheses(conversation_history, candidates, similar_groups, self.context)
                else:
                    similar_groups = self.context.belief.get_similarity_groups(threshold=self.tracer_config.similarity_threshold)
                    perturb_hypotheses(conversation_history, candidates, similar_groups, self.context)
        
        # Evaluate profile alignment    
        profile = summarize_profile(self.context)
        profile_alignment = profile_score(self.evaluation_model, profile, user_data.gt_profile, self.evaluation_config)
        