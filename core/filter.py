from typing import Any, Dict, List
from pydantic import BaseModel, Field
from core.utils import TracerContext
from core.hypothesis_set import WorkingBelief, Update, Hypothesis
from data.base import Turn


LIKELIHOOD_PROMPT = """
You are estimating the behavioral likelihood P(a | z).

Interpretation:
- a = the candidate the user actually chose.
- z = a single hypothesis about the user's latent preference/value.
- You must assume z is TRUE for the purpose of this evaluation.
- Your task is to estimate how likely a rational user with preference z would choose the selected candidate a over the alternatives.

Given:
- Current user message
- All candidate responses with the chosen one (a) indicated
- Hypothesis z (a statement describing a latent user preference/value)

Step 2 — Likelihood reasoning.
Assuming hypothesis z is true:
- Does z strongly favor the chosen candidate?
- Does z weakly favor it?
- Is z irrelevant to the distinguishing dimensions?
- Does z favor a different candidate instead?

Think in terms of comparative preference, not absolute quality.

Step 3 — Assign a probability P(a | z).

Use the following calibrated anchors:

Very Likely       ≈ 0.90
Likely            ≈ 0.70
Somewhat Likely   ≈ 0.50
Somewhat Unlikely ≈ 0.30
Unlikely          ≈ 0.10
Very Unlikely     ≈ 0.05

You may choose intermediate numeric values (e.g., 0.65, 0.82) if justified,
but stay within [0.05, 0.95] unless the case is extremely clear.

Important:
- This is NOT judging whether z is correct.
- This is NOT judging which candidate is objectively best.
- Only estimate: if z were true, how likely would a be chosen?
- Do not assume additional hidden preferences.
- Base your judgment only on observable differences.

Output JSON only:
{{
  "reason": "2-4 sentences briefly analyzing how z is related to the choice.",
  "likelihood_bucket": "Very Likely | Likely | Somewhat Likely | Somewhat Unlikely | Unlikely | Very Unlikely",
  "likelihood": 0.xx
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

class FilterSchema(BaseModel):
    likelihood: float = Field(ge=0.0, le=1.0)


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
    
    outputs = [o["output"] if not isinstance(o, Exception) else None for o in context.model.async_generate(likelihood_prompts, schema=FilterSchema, cfg=context.generation_config) ]
    updates = []
    invalid = 0
    for h, o in zip(hypotheses, outputs):
        if o is None:
            invalid += 1
        likelihood = o['likelihood'] if o else 0.5
        update = Update(
            id=h.id,
            likelihood=likelihood
        )
        updates.append(update)
    context.belief.update(updates=updates)
    return {"invalid": invalid}
