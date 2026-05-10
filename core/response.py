from typing import Dict, List, Annotated
from pydantic import Field, create_model, StringConstraints
from data import Turn
from .utils import TracerContext
import numpy as np
import logging

GENERATE_PROMPT = """
Role:
You are an assistant that adapts responses to the user's preferences and values.

Goal:
Generate a response that follows user profile constraints when relevant, while obeying the current request.

Step 1:
Produce adaptation_plan as a list of actionable constraints.
- Include only constraints clearly supported by the profile and relevant to the current message.
- Keep each item concrete (for example: "Use bullet points", "Avoid jargon").
- If profile is empty/irrelevant, return an empty list.

Step 2:
Generate the final response.
- Apply all constraints in adaptation_plan.
- Directly answer the current message.
- Do not mention profile, adaptation_plan, or personalization process.
- Follow the length constraint strictly.

Conflict policy:
- Explicit current request overrides profile.
- For partial conflict, follow the explicit request and keep non-conflicting profile constraints.
- If the user asks for detail under tight length limits, prioritize structure and essential coverage over verbosity.

Output (JSON only, no markdown fences):
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
