from typing import Optional
from .utils import TracerConfig, TracerContext, EmbedConfig
from .hypothesis_set import Hypothesis, HypothesisSet, WorkingBelief
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
        tracer_cfg: TracerConfig, 
        model: BaseLM,
        generation_cfg: GenerationConfig,
        embed_cfg: EmbedConfig,
        evaluation_model: Optional[BaseLM] = None,
        evaluation_cfg: Optional[GenerationConfig] = None
    ):
        self.model = model
        self.evaluation_model = evaluation_model or model
        self.base_generation_config = generation_cfg
        self.embed_config = embed_cfg
        self.evaluation_config = evaluation_cfg or generation_cfg
        self.tracer_config = tracer_cfg
    
    def trace(self, user_data: UserData):
        hypothesis_set = HypothesisSet(self.embed_config)
        context = TracerContext(
            model=self.model,
            hypothesis_set=hypothesis_set,
            tracer_config=self.tracer_config,
            generation_config=self.base_generation_config
        )
        records = {"user": user_data.user_id, "turns": []}
        for conversation in user_data.conversations:
            initialized = False
            conversation_history = []
            for turn in conversation.turns:
                turn_record = {}
                conversation_history.append(turn)
                working_profile = summarize_hypotheses(conversation_history, context)
                turn_record["summary"] = working_profile
                # Online Evaluation
                turn_record["choice_metrics"] = predict_choice(
                    model=self.model, 
                    conversation_history=conversation_history, 
                    profile=working_profile, 
                    generation_cfg=self.base_generation_config
                )
                
                turn_record["generation_metrics"] = evaluate_generation(
                    model=self.evaluation_model, 
                    conversation_history=conversation_history, 
                    profile=working_profile,
                    embed_cfg=self.embed_config,
                    generation_cfg=self.base_generation_config,
                    eval_model=self.evaluation_model,
                    evaluation_cfg=self.evaluation_config
                )

                # Online Update
                candidates, preprocess_status = preprocess_candidates(conversation_history, context)
                turn_record["preprocess"] = preprocess_status
                if not preprocess_status["success"] or preprocess_status["skip"]:
                    records["turns"].append(turn_record)
                    continue
                if not initialized:
                    initialize_record = initialize_hypothesis(conversation_history, candidates, context)
                    turn_record["initialize"] = initialize_record
                    if not (initialized := initialize_record["success"]):
                        records["turns"].append(turn_record)
                        continue
                else:
                    branch_status = branch_hypotheses(conversation_history, candidates, context)
                    turn_record["branch"] = branch_status
                weight_status = weight_hypothesis(conversation_history, candidates, context)
                turn_record["weight"] = weight_status
                if (ess := context.belief.ess()) < self.tracer_config.n_hypotheses / 2:
                    similar_groups = context.belief.resample()
                else:
                    similar_groups = context.belief.get_similarity_groups(threshold=self.tracer_config.similarity_threshold)
                turn_record["perturb"] = perturb_hypotheses(conversation_history, candidates, similar_groups, context)
                turn_record["perturb"]["ess"] = ess
                turn_record["hypotheses"] = context.belief.log_dict()
                records["turns"].append(turn_record)
            records["turns"][-1]["consolidate"] = consolidate_hypotheses(conversation_history, context)
        # Evaluate profile alignment    
        profile = summarize_profile(context)
        records["profile_metrics"] = profile_score(self.evaluation_model, profile, user_data.gt_profile, self.embed_config, self.evaluation_config)
        return records
        
        