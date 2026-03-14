from typing import Any, Dict, List
import asyncio
import logging
import numpy as np
from pydantic import create_model, conlist, confloat
from core.utils import TracerContext
from core.hypothesis_set import Update
from data.base import Turn


LIKELIHOOD_PROMPT = """
You are scoring alignment between candidate responses and a hypothesis z in order to estimate choice likelihood.

Definitions:
- z: a single hypothesis about the user's latent preference/value. Assume z is TRUE.
- candidates: multiple responses to the same user message.

Goal:
For each candidate i, assign an ALIGNMENT SCORE s_i in 0-5 indicating how well the candidate matches z,
based on preference-relevant differences (values/style/structure/constraints implied by z), NOT general quality.

Scoring anchors (use these strictly):
5 = Best match to z; clearly satisfies z better than others (large margin)
4 = Strong match; among the top or tied-top under z
3 = Moderate match; plausible under z but not top
2 = Weak match; conflicts with z in at least one important way
1 = Poor match; largely mismatched to z
0 = Opposes z or ignores it entirely

Rules:
- Compare candidates RELATIVELY under z. Do not score absolute correctness unless z explicitly cares about it.
- If z is irrelevant to differences among candidates, give near-equal scores (e.g., all 3, or 3/3/2 with tiny variance).
- Use the full range when justified; avoid defaulting to 3 unless genuinely ambiguous.
- Do NOT invent preferences not stated in z or the user message.
- The output list must have exactly the same length as the number of candidates.
- scores[i] MUST correspond to candidate i in the given order. Do NOT reorder candidates.

Output valid, parsable JSON only:
{{
  "scores": [s0, s1, ...]
}}

[Conversation history]
{prev_turns}

[Current User Message]
{user_message}

[Candidate Responses]
{candidates}

[Hypothesis z]
{hypothesis}
"""

FILTER_BUDGET = 64
logger = logging.getLogger(__name__)

def softmax(logits: List[float], t: float = 1) -> np.ndarray:
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
    FilterSchema = create_model(
        "FilterSchema",
        scores=(conlist(confloat(ge=0, le=5), min_length=c, max_length=c), ...)
    )
    
    try:
        async_outputs = asyncio.run(
            context.model.async_generate(
                likelihood_prompts, schema=FilterSchema, cfg=context.generation_config, max_tokens=FILTER_BUDGET
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
        likelihood = softmax(o['scores'], t=0.5)[current_turn.chosen_idx] if o else softmax([3] * len(current_turn.candidates))[current_turn.chosen_idx]
        update = Update(
            id=h.id,
            likelihood=float(likelihood)
        )
        updates.append(update)
    context.belief.update(updates=updates)
    return {"invalid": invalid}
