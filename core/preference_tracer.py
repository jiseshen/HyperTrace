import json
from typing import Optional, TypedDict
from .utils import TracerConfig, TracerContext
from .hypothesis_set import HypothesisSet
from data import UserData
from model import BaseLM, GenerationConfig, EmbedConfig

from .preprocess import preprocess_candidates
from .initialize import initialize_hypothesis
from .branch import branch_hypotheses
from .filter import weight_hypothesis
from .perturb import perturb_hypotheses
from .summary import summarize_hypotheses, summarize_profile
from .consolidate import consolidate_hypotheses
from .response import generate_adapted_response

class Records(TypedDict):
    user: str
    turns: list[dict]
    general_profile: Optional[str]

class PreferenceTracer:
    def __init__(
        self,
        tracer_cfg: TracerConfig, 
        model: BaseLM,
        generation_cfg: GenerationConfig,
        embed_cfg: EmbedConfig,
    ):
        self.model = model
        self.base_generation_config = generation_cfg
        self.embed_config = embed_cfg
        self.tracer_config = tracer_cfg
    
    def trace(self, user_data: UserData):
        hypothesis_set = HypothesisSet(n_hypotheses=self.tracer_config.n_hypotheses, embed_config=self.embed_config)
        context = TracerContext(
            model=self.model,
            hypothesis_set=hypothesis_set,
            tracer_config=self.tracer_config,
            generation_config=self.base_generation_config
        )
        working_profile = ""
        records: Records = {"user": user_data.user_id, "turns": [], "general_profile": None}
        for conversation in user_data.conversations:
            initialized = False
            conversation_history = []
            for turn in conversation.turns:
                turn_record = {}
                conversation_history.append(turn)

                turn_record["adapted"] = generate_adapted_response(
                    conversation_history=conversation_history, 
                    profile=working_profile,
                    context=context
                )

                # Online Update
                structured_candidates, preprocess_status = preprocess_candidates(conversation_history, context)
                turn_record["preprocess"] = preprocess_status
                if not preprocess_status["success"] or preprocess_status["skip"]:
                    records["turns"].append(turn_record)
                    continue
                candidates_with_choice = "[CandidateSet]\n" + json.dumps(structured_candidates, ensure_ascii=False, indent=2)
                candidates_for_filter = "[CandidateSet]\n" + json.dumps(
                    [{"i": item["i"], "summary": item["summary"], "content": item["content"]} for item in structured_candidates],
                    ensure_ascii=False,
                    indent=2,
                )
                if not initialized:
                    initialize_record = initialize_hypothesis(conversation_history, candidates_with_choice, context)
                    turn_record["initialize"] = initialize_record
                    if not (initialized := initialize_record["success"]):
                        records["turns"].append(turn_record)
                        continue
                else:
                    branch_status = branch_hypotheses(conversation_history, candidates_with_choice, context)
                    turn_record["branch"] = branch_status
                weight_status = weight_hypothesis(conversation_history, candidates_for_filter, context)
                turn_record["weight"] = weight_status
                working_profile = summarize_hypotheses(context)
                if (ess := context.belief.ess()) < self.tracer_config.n_hypotheses / 2:
                    similar_groups = context.belief.resample()
                else:
                    similar_groups = context.belief.get_similarity_groups(threshold=self.tracer_config.similarity_threshold)
                turn_record["perturb"] = perturb_hypotheses(conversation_history, candidates_with_choice, similar_groups, context)
                turn_record["perturb"]["ess"] = ess
                turn_record["hypotheses"] = context.belief.log_dict()
                turn_record["summary"] = working_profile
                records["turns"].append(turn_record)
            
            # Consolidate at the end of conversation
            if conversation_history and initialized:
                records["turns"][-1]["consolidate"] = consolidate_hypotheses(conversation_history, context)
                
        # Export final profile for offline evaluation
        if context.current_belief is not None:
            records["final_profile"] = summarize_profile(context)
        return records
        
        
