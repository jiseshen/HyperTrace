import json
from typing import Any, Callable, Optional, TypedDict
from .utils import TracerConfig, TracerContext
from .hypothesis_set import HypothesisSet, WorkingBelief
from data import UserData
from model import BaseLM, GenerationConfig, EmbedConfig
from prompt import PromptSet, prism_prompts

from .preprocess import preprocess_candidates
from .initialize import initialize_hypothesis
from .branch import branch_hypotheses
from .filter import weight_hypothesis
from .perturb import perturb_hypotheses
from .summary import retrieve_hypothesis_items, retrieve_inference_profile, summarize_hypotheses, summarize_profile, summarize_retrieved_hypotheses
from .consolidate import consolidate_hypotheses
from .response import generate_adapted_response

class Records(TypedDict):
    user: str
    turns: list[dict]
    general_profile: Optional[str]

ProgressHook = Callable[[str, int, int], None]

NON_CONSOLIDATING_MODES = {"flat5", "retrieve_replace"}


def retrieve_belief_for_update(
    query: str,
    context: TracerContext,
    retrieved_items: Optional[list[dict[str, Any]]] = None,
    source: str = "belief_retrieval",
) -> dict[str, Any]:
    if retrieved_items is None:
        pool_k = max(context.tracer_config.belief_retrieve_pool_k, context.tracer_config.belief_retrieve_top_k)
        retrieved_items = retrieve_hypothesis_items(query, context, top_k=pool_k)
    selected = retrieved_items[:context.tracer_config.belief_retrieve_top_k]
    if not selected:
        return {
            "success": False,
            "source": source,
            "query": query,
            "retrieved": [],
            "pool_count": len(retrieved_items),
        }
    ids = [item["id"] for item in selected]
    selected_priors = [item["prior"] for item in selected]
    if sum(selected_priors) <= 0:
        selected_priors = [1.0 for _ in selected_priors]
    context.update_belief(WorkingBelief(
        ids=ids,
        priors=selected_priors,
        repo=context.hypothesis_set,
    ))
    if not context.tracer_config.use_hypothesis_topics:
        for item in selected:
            item.pop("category", None)
    return {
        "success": True,
        "source": source,
        "query": query,
        "retrieved": selected,
        "pool_count": len(retrieved_items),
    }


def apply_belief_retrieval(status: dict[str, Any], context: TracerContext) -> bool:
    if not status.get("success"):
        return False
    selected = status.get("retrieved", [])
    ids = [item["id"] for item in selected]
    selected_priors = [item["prior"] for item in selected]
    if not ids:
        return False
    if sum(selected_priors) <= 0:
        selected_priors = [1.0 for _ in selected_priors]
    context.update_belief(WorkingBelief(
        ids=ids,
        priors=selected_priors,
        repo=context.hypothesis_set,
    ))
    return True


def replace_global_priors_from_current_belief(context: TracerContext) -> None:
    context.hypothesis_set.replace_belief_priors(context.belief.ids, context.belief.weights)

class PreferenceTracer:
    def __init__(
        self,
        tracer_cfg: TracerConfig, 
        model: BaseLM,
        generation_cfg: GenerationConfig,
        embed_cfg: EmbedConfig,
        prompts: PromptSet = None,
        progress_hook: Optional[ProgressHook] = None,
    ):
        self.model = model
        self.base_generation_config = generation_cfg
        self.embed_config = embed_cfg
        self.tracer_config = tracer_cfg
        self.prompts = prompts or prism_prompts()
        self.progress_hook = progress_hook
    
    def trace(self, user_data: UserData):
        hypothesis_set = HypothesisSet(
            n_hypotheses=self.tracer_config.n_hypotheses,
            embed_config=self.embed_config,
            use_topics=self.tracer_config.use_hypothesis_topics,
        )
        context = TracerContext(
            model=self.model,
            hypothesis_set=hypothesis_set,
            tracer_config=self.tracer_config,
            generation_config=self.base_generation_config,
            prompts=self.prompts,
        )
        working_profile = ""
        records: Records = {"user": user_data.user_id, "turns": [], "general_profile": None}
        total_turns = sum(len(conversation.turns) for conversation in user_data.conversations)
        turn_index = 0
        initialized_once = False
        update_mode = self.tracer_config.hypothesis_update_mode
        for conversation in user_data.conversations:
            initialized = initialized_once if update_mode in NON_CONSOLIDATING_MODES else False
            conversation_history = []
            for turn in conversation.turns:
                turn_index += 1
                if self.progress_hook:
                    self.progress_hook(user_data.user_id, turn_index, total_turns)
                turn_record = {}
                conversation_history.append(turn)
                if (
                    self.tracer_config.inference_profile_source == "working"
                    and update_mode == "hybrid"
                    and
                    len(conversation_history) == 1
                    and len(context.hypothesis_set.hypotheses) > context.tracer_config.n_hypotheses
                ):
                    try:
                        retrieved_profile = summarize_retrieved_hypotheses(turn.user_message, context)
                    except Exception as e:
                        retrieved_profile = ""
                        turn_record["pre_adapt_retrieved_summary"] = {"success": False, "reason": str(e)}
                    if retrieved_profile:
                        working_profile = retrieved_profile
                        turn_record["summary"] = working_profile
                        turn_record["pre_adapt_retrieved_summary"] = {"success": True}

                inference_profile = working_profile
                shared_belief_retrieval = None
                if self.tracer_config.inference_profile_source == "retrieved":
                    try:
                        inference_profile, inference_retrieval = retrieve_inference_profile(
                            conversation_history,
                            context,
                            fallback_profile=working_profile,
                            include_belief_retrieval=update_mode == "retrieve_replace",
                        )
                        turn_record["inference_profile"] = inference_profile
                        turn_record["inference_retrieval"] = inference_retrieval
                        shared_belief_retrieval = inference_retrieval.get("belief_retrieval")
                    except Exception as e:
                        inference_profile = working_profile
                        turn_record["inference_profile"] = inference_profile
                        turn_record["inference_retrieval"] = {
                            "source": "working_fallback",
                            "reason": str(e),
                            "retrieved": [],
                        }

                turn_record["adapted"] = generate_adapted_response(
                    conversation_history=conversation_history, 
                    profile=inference_profile,
                    context=context
                )

                # Online Update
                structured_candidates, preprocess_status = preprocess_candidates(conversation_history, context)
                turn_record["preprocess"] = preprocess_status
                if not preprocess_status["success"] or preprocess_status["skip"]:
                    if working_profile:
                        turn_record["summary"] = working_profile
                    records["turns"].append(turn_record)
                    continue
                candidates_with_choice = "[CandidateSet]\n" + json.dumps(structured_candidates, ensure_ascii=False, indent=2)
                candidates_for_filter = "[CandidateSet]\n" + json.dumps(
                    [{"i": item["i"], "summary": item["summary"], "content": item["content"]} for item in structured_candidates],
                    ensure_ascii=False,
                    indent=2,
                )
                if update_mode == "retrieve_replace" and context.hypothesis_set.hypotheses:
                    if shared_belief_retrieval is not None:
                        turn_record["belief_retrieval"] = shared_belief_retrieval
                        initialized = apply_belief_retrieval(shared_belief_retrieval, context)
                    else:
                        turn_record["belief_retrieval"] = retrieve_belief_for_update(turn.user_message, context)
                        initialized = apply_belief_retrieval(turn_record["belief_retrieval"], context)

                if not initialized:
                    initialize_record = initialize_hypothesis(conversation_history, candidates_with_choice, context)
                    turn_record["initialize"] = initialize_record
                    if not (initialized := initialize_record["success"]):
                        records["turns"].append(turn_record)
                        continue
                    if update_mode in NON_CONSOLIDATING_MODES:
                        initialized_once = True
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
                if update_mode == "retrieve_replace":
                    replace_global_priors_from_current_belief(context)
                    turn_record["prior_update"] = {"mode": "replace", "ids": list(context.belief.ids)}
                turn_record["hypotheses"] = context.belief.log_dict(
                    include_category=self.tracer_config.use_hypothesis_topics,
                )
                turn_record["summary"] = working_profile
                records["turns"].append(turn_record)
            
            # Consolidate at the end of conversation
            if conversation_history and initialized and update_mode not in NON_CONSOLIDATING_MODES:
                records["turns"][-1]["consolidate"] = consolidate_hypotheses(conversation_history, context)
                
        # Export final profile for offline evaluation
        if context.current_belief is not None:
            records["final_profile"] = summarize_profile(context)
        return records
        
        
