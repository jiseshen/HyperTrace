from typing import Any, Dict, List
import asyncio
import logging
import numpy as np
from pydantic import Field, create_model, conlist, confloat
from core.utils import TracerContext
from core.hypothesis_set import Update
from data.base import Turn
from prompt.base import LIKELIHOOD_PROMPT


FILTER_BUDGET = 64
logger = logging.getLogger(__name__)

def bradley_terry_softmax(logits: List[float], t: float = 1) -> np.ndarray:
    scaled_logits = np.array(logits) / t
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))  # for numerical stability
    return exp_logits / np.sum(exp_logits)

def weight_hypothesis(conversation_history: List[Turn], candidates: str, context: TracerContext) -> Dict[str, Any]:
    prev_turns = conversation_history[:-1]
    current_turn = conversation_history[-1]
    hypotheses = context.belief.get_hypotheses()
    likelihood_prompts = [
        context.prompts.likelihood.format(
            prev_turns="\n\n".join([turn.format(include_candidates=False) for turn in prev_turns[-context.tracer_config.max_history_turns:]]),
            user_message=current_turn.user_message,
            candidates=candidates,
            hypothesis=h.format(include_category=context.tracer_config.use_hypothesis_topics)
        ) for h in hypotheses
    ]
    c = len(current_turn.candidates)
    if c < 2:
        logger.warning("Only one candidate available; skipping likelihood weighting")
        return {"invalid": len(hypotheses)}
    FilterSchema = create_model(
        "FilterSchema",
        scores=(conlist(confloat(ge=0, le=5), min_length=c, max_length=c), Field(..., description=f"List of {c} alignment scores in 0-5 for each candidate response"))
    )
    
    try:
        filter_overrides = context.get_generation_overrides("filter_override")
        async_outputs = asyncio.run(
            context.model.async_generate(
                likelihood_prompts, schema=FilterSchema, cfg=context.generation_config, max_tokens=FILTER_BUDGET, **filter_overrides
            )
        )
    except Exception:
        logger.exception("Likelihood weighting generation failed")
        async_outputs = [None for _ in likelihood_prompts]
    outputs = [o["output"] if isinstance(o, dict) else None for o in async_outputs]
    updates = []
    invalid = 0
    for h, o in zip(hypotheses, outputs):
        if o is None:
            invalid += 1
        likelihood = bradley_terry_softmax(o['scores'], t=context.tracer_config.bradley_terry_temp)[current_turn.chosen_idx] if o else bradley_terry_softmax([3] * len(current_turn.candidates))[current_turn.chosen_idx]
        update = Update(
            id=h.id,
            likelihood=float(likelihood)
        )
        updates.append(update)
    context.belief.update(updates=updates)
    return {"invalid": invalid}
