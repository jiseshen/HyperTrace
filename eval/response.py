from typing import Any, Dict, List, Optional
from model import BaseLM, GenerationConfig
from pydantic import Field, create_model, confloat, conlist
from data import Turn
from .utils import candidate_similarity_scores, EmbedConfig
import logging
from prompt import PromptSet, prism_prompts
from prompt.base import RESPONSE_EVALUATION_PROMPT

EVALUATE_PROMPT = RESPONSE_EVALUATION_PROMPT


EVAL_BUDGET = 384
logger = logging.getLogger(__name__)


def evaluate_generation(
    eval_model: BaseLM,
    conversation_history: List[Turn],
    adapted_response: str,
    embed_cfg: EmbedConfig,
    evaluation_cfg: Optional[GenerationConfig] = None,
    prompts: Optional[PromptSet] = None,
) -> Dict[str, Any]:
    current_turn = conversation_history[-1]
    c = len(current_turn.candidates)
    similarity_scores = candidate_similarity_scores(adapted_response, current_turn.candidates, embed_cfg)
    similarity_score = similarity_scores[current_turn.chosen_idx]
    rejected_similarity_scores = [
        score for idx, score in enumerate(similarity_scores)
        if idx != current_turn.chosen_idx
    ]
    relative_score = similarity_score - max(rejected_similarity_scores) if rejected_similarity_scores else 0.0
    relative_mean_score = similarity_score - (sum(rejected_similarity_scores) / len(rejected_similarity_scores)) if rejected_similarity_scores else 0.0
    if c < 2:
        return {
            "gpt_score": 2.5,
            "relative_gpt_score": 0,
            "relative_mean_gpt_score": 0,
            "gpt_scores": [],
            "similarity_score": similarity_score,
            "relative_score": relative_score,
            "relative_mean_score": relative_mean_score,
            "similarity_scores": similarity_scores,
            "chosen_idx": current_turn.chosen_idx,
            "rejected_similarity_scores": rejected_similarity_scores,
            "error": "Evaluation skipped due to insufficient candidates",
        }
    prompt_set = prompts or prism_prompts()
    evaluate_prompt = prompt_set.response_evaluation.format(
        current_turn=current_turn.format(include_candidates=True, include_choice=False),
        adapted=adapted_response,
        c=c,
    )
    EvalSchema = create_model(
        "EvalSchema",
        dimensions=(list[str], Field(..., description="List of key dimensions used for evaluation")),
        scores=(conlist(confloat(ge=0, le=5), min_length=c, max_length=c), Field(..., description=f"List of {c} similarity scores in 0-5 for each candidate response")),
        justification=(str, Field(..., description="a brief explanation")),
    )
    try:
        evaluate_output = eval_model.generate(
            prompt=evaluate_prompt,
            schema=EvalSchema,
            cfg=evaluation_cfg,
            max_tokens=EVAL_BUDGET,
        )["output"]
    except Exception as e:
        logger.exception("Response evaluation failed")
        return {
            "gpt_score": None,
            "relative_gpt_score": None,
            "relative_mean_gpt_score": None,
            "gpt_scores": [],
            "similarity_score": similarity_score,
            "relative_score": relative_score,
            "relative_mean_score": relative_mean_score,
            "similarity_scores": similarity_scores,
            "chosen_idx": current_turn.chosen_idx,
            "rejected_similarity_scores": rejected_similarity_scores,
            "error": "Evaluation: " + str(e),
        }
    scores = evaluate_output["scores"]
    gpt_score = scores[current_turn.chosen_idx]
    rejected_gpt_scores = [
        score for idx, score in enumerate(scores)
        if idx != current_turn.chosen_idx
    ]
    relative_gpt_score = gpt_score - max(rejected_gpt_scores)
    relative_mean_gpt_score = gpt_score - (sum(rejected_gpt_scores) / len(rejected_gpt_scores))
    return {
        "gpt_score": gpt_score,
        "relative_gpt_score": relative_gpt_score,
        "relative_mean_gpt_score": relative_mean_gpt_score,
        "gpt_scores": scores,
        "similarity_score": similarity_score,
        "relative_score": relative_score,
        "relative_mean_score": relative_mean_score,
        "similarity_scores": similarity_scores,
        "chosen_idx": current_turn.chosen_idx,
        "rejected_gpt_scores": rejected_gpt_scores,
        "rejected_similarity_scores": rejected_similarity_scores,
    }
