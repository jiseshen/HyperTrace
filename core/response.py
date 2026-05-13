from typing import Dict, List, Annotated
from pydantic import Field, create_model, StringConstraints
from data import Turn
from .utils import TracerContext
import numpy as np
import logging
from prompt.base import RESPONSE_PROMPT

GENERATE_PROMPT = RESPONSE_PROMPT

GEN_BUDGET = 1024
RESPONSE_MAX_WORD_COUNT = 400
logger = logging.getLogger(__name__)

def generate_adapted_response(
    conversation_history: List[Turn],
    profile: str,
    context: TracerContext
) -> Dict[str, str]:
    prev_turns = conversation_history[:-1]
    current_turn = conversation_history[-1]
    current_message = current_turn.user_message
    word_counts = [len(c.split()) for c in current_turn.candidates]
    upper_bound = min(RESPONSE_MAX_WORD_COUNT, int(np.quantile(word_counts, 0.9) * 1.1))
    lower_bound = max(min(RESPONSE_MAX_WORD_COUNT * 0.9, int(np.quantile(word_counts, 0.1) * 0.9)), 1)
    if lower_bound > int(upper_bound * 0.9):
        lower_bound = max(1, int(upper_bound * 0.9))
    generate_prompt = context.prompts.response.format(
        profile=profile,
        prev_turns="\n\n".join([turn.format(include_candidates=False) for turn in prev_turns]),
        current_message=current_message,
        l=lower_bound,
        r=upper_bound,
    )
    GenSchema = create_model(
        "GenSchema",
        adaptation_plan=(list[str], Field(..., description="List of actionable constraints to adapt the response to the user preferences")),
        response=(Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)], Field(..., description="the response to current user message adapted to the user preferences")),
    )
    overrides = context.get_generation_overrides("response_override")
    try:
        generate_output = context.model.generate(
            prompt=generate_prompt,
            schema=GenSchema,
            cfg=context.generation_config,
            max_tokens=GEN_BUDGET,
            **overrides,
        )["output"]
    except Exception as e:
        logger.exception("Failed to generate adapted response")
        return {"success": False, "response": "", "reason": str(e)}
    return {
        "success": True,
        "response": generate_output["response"]
    }
