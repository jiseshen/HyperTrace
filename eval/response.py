from typing import Dict, List, Optional
from model import BaseLM, GenerationConfig
from pydantic import BaseModel, Field
from data import Turn
from .utils import text_similarity, relative_similarity_score, EmbedConfig

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

class GenSchema(BaseModel):
    response: str

class EvalSchema(BaseModel):
    score: int = Field(ge=1, le=10)


def evaluate_generation(gen_model: BaseLM, conversation_history: List[Turn], profile: str, embed_cfg: EmbedConfig, generation_cfg: Optional[GenerationConfig] = None, eval_model: Optional[BaseLM] = None, evaluation_cfg: Optional[GenerationConfig] = None) -> Dict[str, float]:
    prev_turns = conversation_history[:-1]
    current_turn = conversation_history[-1]
    current_message = current_turn.user_message
    generate_prompt = GENERATE_PROMPT.format(profile=profile, prev_turns=prev_turns, current_message=current_message)
    if eval_model is None:
        eval_model = gen_model
    if evaluation_cfg is None:
        evaluation_cfg = generation_cfg
    try:
        generate_output = gen_model.generate(
            prompt=generate_prompt,
            schema=GenSchema,
            generation_config=generation_cfg,
        )["output"]
    except Exception as e:
        return {"gpt_score": 1.0, "similarity_score": 0.0, "relative_score": 0.0, "error": "Generation: " + str(e)}
    adapted_response = generate_output["response"]
    relative_score = relative_similarity_score(adapted_response, current_turn.candidates, current_turn.chosen_idx, embed_cfg)
    similarity_score = text_similarity(adapted_response, current_turn.chosen, embed_cfg)
    evaluate_prompt = EVALUATE_PROMPT.format(
        current_turn=current_turn.format(include_candidates=True, include_choice=True),
        adapted=adapted_response
    )
    try:
        evaluate_output = eval_model.generate(             
            prompt=evaluate_prompt,
            schema=EvalSchema,
            generation_config=evaluation_cfg
        )["output"]
    except Exception as e:
        return {"gpt_score": 5.0, "similarity_score": similarity_score, "relative_score": relative_score, "error": "Evaluation: " + str(e)}
    return {
        "gpt_score": evaluate_output["score"], 
        "similarity_score": similarity_score,
        "relative_score": relative_score,
    }