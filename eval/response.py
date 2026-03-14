from typing import Dict, List, Optional
from model import BaseLM, GenerationConfig
from pydantic import BaseModel, create_model, confloat, conlist
from data import Turn
from .utils import text_similarity, relative_similarity_score, EmbedConfig
import numpy as np

GENERATE_PROMPT = """
You are an assistant that adapts responses to a user's preferences and values.

Given:
- User profile: a concise summary of the user's current preferences and values
- Conversation history (optional)
- Current user message

Task:

Step 1 — Produce a brief adaptation_plan.
Include ONLY aspects that are clearly supported by the user profile and relevant to the current message.
Do NOT force-fill categories.
Express each item as an actionable constraint (not a vague description).

Possible aspects (not exhaustive, include only if applicable):
- Values constraints
- Information density
- Structure
- Level of abstraction
- Framing
- Actionability
- Tone

Step 2 — Generate the final response.
The response must:
- Follow the adaptation_plan and align with the user profile
- Directly address the current user message
- Be helpful and relevant
- NOT mention the profile or adaptation process directly

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
You are evaluating the similarity between an Adapted response and a set of candidate responses in the context of the user preferences.

Inputs:
- Current turn with candidates (chosen_index indicated)
- Adapted response

Step 1:
Identify 2-4 preference-relevant dimensions that distinguish the candidates from at least one alternative.
Use short canonical labels (e.g., "values", "information_density", "structure", "actionability", "tone", "framing", "abstraction") (Not exhaustive and do not force-fit).

Step 2:
Using ONLY those dimensions, score how similar the Adapted response is to EACH candidate.
Scores are 0-5:
5 = Near-identical on the key dimensions; Adapted matches candidate's stance/style/structure with no meaningful drift.
4 = Strong match on most key dimensions; minor drift on at most one dimension.
3 = Partial match; aligns on some key dimensions but differs on others OR ambiguity prevents a clear judgment.
2 = Weak match; differs on one or more key dimensions in a way that matters.
1 = Very weak match; mostly reflects the opposite of the candidate on key dimensions.
0 = Opposes/contradicts the candidate on the key dimensions (clear mismatch).

Output valid, parsable JSON only:
{{
  "dimensions": ["...", "..."],
  "scores": [s0, s1, ...]
}}

Rules:
- scores length MUST equal number of candidates; scores[i] corresponds to candidate i. Do NOT change the candidate order.
- Use the same dimensions for scoring all candidates.

[Current turn]
{current_turn}

[Adapted]
{adapted}
"""

class GenSchema(BaseModel):
    response: str

GEN_BUDGET = 1024
EVAL_BUDGET = 128

def evaluate_generation(gen_model: BaseLM, conversation_history: List[Turn], profile: str, embed_cfg: EmbedConfig, generation_cfg: Optional[GenerationConfig] = None, eval_model: Optional[BaseLM] = None, evaluation_cfg: Optional[GenerationConfig] = None) -> Dict[str, float]:
    prev_turns = conversation_history[:-1]
    current_turn = conversation_history[-1]
    current_message = current_turn.user_message
    generate_prompt = GENERATE_PROMPT.format(
        profile=profile,
        prev_turns="\n\n".join([turn.format(include_candidates=False) for turn in prev_turns]),
        current_message=current_message,
    )
    if eval_model is None:
        eval_model = gen_model
    if evaluation_cfg is None:
        evaluation_cfg = generation_cfg
    try:
        generate_output = gen_model.generate(
            prompt=generate_prompt,
            schema=GenSchema,
            cfg=generation_cfg,
            max_tokens=GEN_BUDGET
        )["output"]
    except Exception as e:
        return {"gpt_score": 0.0, "similarity_score": 0.0, "relative_score": 0.0, "error": "Generation: " + str(e)}
    adapted_response = generate_output["response"]
    relative_score = relative_similarity_score(adapted_response, current_turn.candidates, current_turn.chosen_idx, embed_cfg)
    similarity_score = text_similarity(adapted_response, current_turn.chosen, embed_cfg)
    evaluate_prompt = EVALUATE_PROMPT.format(
        current_turn=current_turn.format(include_candidates=True, include_choice=True),
        adapted=adapted_response
    )
    c = len(current_turn.candidates)
    EvalSchema = create_model(
        "EvalSchema", 
        scores=(conlist(confloat(ge=0, le=5), min_length=c, max_length=c), ...)
    )
    try:
        evaluate_output = eval_model.generate(             
            prompt=evaluate_prompt,
            schema=EvalSchema,
            cfg=evaluation_cfg,
            max_tokens=EVAL_BUDGET
        )["output"]
    except Exception as e:
        return {"gpt_score": 2.5, "relative_gpt_score": 0, "similarity_score": similarity_score, "relative_score": relative_score, "error": "Evaluation: " + str(e)}
    gpt_score = evaluate_output["scores"][current_turn.chosen_idx]
    relative_gpt_score = gpt_score - np.mean(evaluate_output["scores"][i] for i in range(c) if i != current_turn.chosen_idx)
    return {
        "gpt_score": gpt_score,
        "relative_gpt_score": float(relative_gpt_score),
        "similarity_score": similarity_score,
        "relative_score": relative_score,
    }
