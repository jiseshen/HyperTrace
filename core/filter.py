from typing import Any, Dict, List
import asyncio
import logging
import numpy as np
from pydantic import Field, create_model, conlist, confloat
from core.utils import TracerContext
from core.hypothesis_set import Update
from data.base import Turn


LIKELIHOOD_PROMPT = """
Role:
You are scoring candidate-hypothesis alignment for choice likelihood estimation.

Given:
- Hypothesis z (assume z is true)
- Multiple candidates for the same user message

Goal:
Assign each candidate i an alignment score s_i in [0, 5] for how well it matches z.
Score relative alignment under z, not general response quality.

Score anchors:
- 5: best match to z with clear margin
- 4: strong match, top or tied-top
- 3: moderate match, plausible but not top
- 2: weak match, conflicts on an important dimension
- 1: poor match, largely mismatched
- 0: opposes or ignores z

Rules:
- Candidates are intentionally unlabeled; do not assume which one was chosen.
- Compare candidates relatively under z.
- If z does not explain candidate differences, keep scores near-equal.
- Use wider score spread only when evidence supports it.
- Do not invent preferences outside z and the user message.
- Keep the original candidate order: scores[i] must map to candidate i.

Output (JSON only):
{{
  "scores": [s0, s1, ...]
}}

[Conversation history]
{prev_turns}

[Current User Message]
{user_message}

[Candidate Responses]
{candidates}

Candidate Responses are provided as JSON list items with:
- i: candidate index
- summary: compact candidate summary
- content: compact candidate content
- There is no choice label in this input.

[Hypothesis z]
{hypothesis}
"""
# TODO: Binary (True or false) scoring + 2 Examples steering

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
        LIKELIHOOD_PROMPT.format(
            prev_turns="\n\n".join([turn.format(include_candidates=False) for turn in prev_turns[-context.tracer_config.max_history_turns:]]),
            user_message=current_turn.user_message,
            candidates=candidates,
            hypothesis=h.format()
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
    outputs = [o["output"] if not isinstance(o, Exception) else None for o in async_outputs]
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
