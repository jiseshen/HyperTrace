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
from .consolidate import consolidate_hypotheses

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
    
    def trace(self, user_data: UserData):
        records = {"user": user_data.user_id, "turns": []}
        for conversation in user_data.conversations:
            initialized = False
            conversation_history = []
            for turn in conversation.turns:
                turn_record = {}
                conversation_history.append(turn)
                working_profile = summarize_hypotheses(conversation_history, self.context)
                turn_record["summary"] = working_profile
                # Online Evaluation
                turn_record["choice_metrics"] = predict_choice(
                    model=self.evaluation_model, 
                    conversation_history=conversation_history, 
                    profile=working_profile, 
                    generation_cfg=self.evaluation_config
                )
                
                turn_record["generation_metrics"] = evaluate_generation(
                    model=self.evaluation_model, 
                    conversation_history=conversation_history, 
                    profile=working_profile,
                    context=self.context
                )

                # Online Update
                candidates, preprocess_status = preprocess_candidates(conversation_history, self.context)
                turn_record["preprocess"] = preprocess_status
                if not preprocess_status["success"] or preprocess_status["skip"]:
                    records["turns"].append(turn_record)
                    continue
                if not initialized:
                    initialize_record = initialize_hypothesis(conversation_history, candidates, self.context)
                    turn_record["initialize"] = initialize_record
                    if not (initialized := initialize_record["success"]):
                        records["turns"].append(turn_record)
                        continue
                else:
                    branch_status = branch_hypotheses(conversation_history, candidates, self.context)
                    turn_record["branch"] = branch_status
                weight_status = weight_hypothesis(conversation_history, candidates, self.context)
                turn_record["weight"] = weight_status
                if (ess := self.context.belief.ess()) < self.tracer_config.n_hypotheses / 2:
                    similar_groups = self.context.belief.resample()
                else:
                    similar_groups = self.context.belief.get_similarity_groups(threshold=self.tracer_config.similarity_threshold)
                turn_record["perturb"] = perturb_hypotheses(conversation_history, candidates, similar_groups, self.context)
                turn_record["perturb"]["ess"] = ess
                records["turns"].append(turn_record)
            records["turns"][-1]["consolidate"] = consolidate_hypotheses(conversation_history, self.context)
        # Evaluate profile alignment    
        profile = summarize_profile(self.context)
        records["profile_metrics"] = profile_score(self.evaluation_model, profile, user_data.gt_profile, self.evaluation_config)
        
        return records
        
        