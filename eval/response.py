import json
from typing import Dict, List
from model import BaseLM, GenerationConfig
from data import Turn

GENERATE_PROMPT = """
You are an assistant that adapts responses to a user's preferences and values.

Given:
- User-specific generation guidelines
- Conversation history (optional)
- Current user message

Task:

Step 1 — Produce a brief adaptation_plan.
Include ONLY aspects that are clearly supported by the guidelines and relevant to the current message.
Do NOT force-fill categories.
Express each item as an actionable constraint (not a vague description).

Possible aspects (not exhaustive, include only if applicable):
- Values constraints (what must be respected or avoided)
- Information density (concise / detailed / balanced)
- Structure (e.g., bullets first, step-by-step, narrative)
- Level of abstraction (high-level vs technical detail)
- Framing (neutral, analytical, persuasive, etc.)
- Actionability (the degree of concrete next steps)
- Tone (formal, casual, direct, supportive, etc.)

Step 2 — Generate the final response.
The response must:
- Follow the adaptation_plan
- Directly address the current user message
- Be helpful and relevant
- Not mention the profile or adaptation process

Conflict resolution rule:
If the current user message explicitly requests something that conflicts with the profile,
follow the explicit request in the current message.

If the user profile is empty or clearly irrelevant:
- Return an empty adaptation_plan
- Provide a helpful and relevant response

Output JSON only:
{{
  "adaptation_plan": {{
      "...": "...",
      "...": "..."
  }},
  "response": "..."
}}

[User preference profile]
{profile}

[Conversation history]
{prev_turns}

[Current user message]
{current_message}
"""

EVALUATE_PROMPT = """
You are evaluating whether an Adapted response aligns more closely with the user's chosen response.

Given:
- Current user message
- Candidate responses (with chosen or rejected specified)
- Adapted response (to evaluate)

Goal:
Determine whether the Adapted response is more similar to the Chosen candidate than to the Rejected candidate.

Step 1:
Identify key preference-relevant differences between Chosen and Rejected.
Focus on (not exhaustive, and only include if relevant):
- Underlying value
- Tone
- Structure
- Information density
- Level of abstraction
- Actionability
- Framing

Step 2:
Compare the Adapted response to both candidates along those dimensions.

Evaluation principles:
- Only consider dimensions that distinguish Chosen from Rejected.
- Do not assume hidden user traits.

Scoring rubric (1-10):

9-10: Adapted clearly reflects the distinguishing qualities of the Chosen candidate.
7-8: Adapted mostly reflects Chosen, with minor resemblance to Rejected.
5-6: Mixed; partially resembles both.
3-4: Adapted resembles Rejected more on key distinguishing aspects.
1-2: Adapted strongly resembles Rejected.

Output JSON only:
{
  "reason": "2-3 sentences describing the key distinguishing signals and how Adapted compares.",
  "score": 1-10
}

[Interaction]
{current_turn}

[Adapted response]
{adapted}
"""


def evaluate_generation(model: BaseLM, conversation_history: List[Turn], profile: str, generation_cfg: GenerationConfig = None) -> Dict[str, float]:
    prev_turns = conversation_history[:-1]
    current_turn = conversation_history[-1]
    current_message = current_turn.user_message
    generate_prompt = GENERATE_PROMPT.format(profile=profile, prev_turns=prev_turns, current_message=current_message)
    
    retries = 0
    while True:
        try:
            generate_output = model.generate(
                prompt=generate_prompt,
                generation_config=generation_cfg
            )["output"]
            adapted_response = json.loads(generate_output)["response"]
            evaluate_prompt = EVALUATE_PROMPT.format(
                current_turn=current_turn.format(include_candidates=True, include_choice=True),
                adapted=adapted_response
            )
            evaluate_output = model.generate(             
                prompt=evaluate_prompt,
                generation_config=generation_cfg
            )["output"]
            evaluation = json.loads(evaluate_output)
            return {
                "generation_score": evaluation["score"],
            }
        except Exception as e:
            retries += 1
            if retries > generation_cfg.max_retries:
                raise ValueError(f"Failed to evaluate generation after {generation_cfg.max_retries} attempts. Error: {e}")