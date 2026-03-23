from typing import Dict, List, Annotated
from pydantic import Field, create_model, StringConstraints
from data import Turn
from .utils import TracerContext
import numpy as np
import logging

GENERATE_PROMPT = """
You are an assistant that adapts responses to a user's preferences and values.

Given:
- User profile: a concise summary of the user's current preferences and values
- Conversation history (optional)
- Current user message

Task:
Step 1 — Produce an adaptation_plan (a list of actionable constraints).
- Include ONLY constraints clearly supported by the user profile AND directly relevant to the current message.
- Each item must be a specific, actionable instruction (e.g., "Use bullet points", "Avoid jargon").
- If the profile is empty or irrelevant to the current message, return an empty list.

Step 2 — Generate the final response.
- Apply every constraint in the adaptation_plan.
- Directly address the current user message.
- Do NOT mention the profile, adaptation plan, or personalization process.
- STRICTLY follow the response length constraint.

Conflict resolution:
- If the current message explicitly requests something that conflicts with the profile, follow the current message.
- If the conflict is partial (e.g., profile says "be concise" but user asks for a detailed breakdown),
  honor the explicit request but apply non-conflicting constraints from the plan.
- If the user requests/prefers a detailed breakdown but the length constraint is tight, be detailed by wording and structure (cover all major points with bullets/steps), not by length: elaborate on essential aspects and trim unnecessary points.

Output valid JSON only. No preamble, no markdown fences.
{{
  "adaptation_plan": [
    "<actionable constraint 1>",
    "<actionable constraint 2>"
  ],
  "response": "<your response to the current message>"
}}

[Response length]
Keep the response length at {l} to {r} words.

[User preference profile]
{profile}

[Conversation history]
{prev_turns}

[Current user message]
{current_message}
"""

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
    generate_prompt = GENERATE_PROMPT.format(
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
