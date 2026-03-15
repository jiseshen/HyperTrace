from typing import Dict, List, Optional
from model import BaseLM, GenerationConfig
from pydantic import create_model, confloat, conlist, constr
from data import Turn
from .utils import text_similarity, relative_similarity_score, EmbedConfig
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

EVALUATE_PROMPT = """
You are evaluating the similarity between an Adapted response and a set of candidate responses in the context of the user preferences.

Inputs:
- Current turn with c candidates
- Adapted response

Step 1:
Identify 2-4 preference-relevant dimensions that distinguish the candidates.
Use short canonical labels (e.g., "values", "information_density", "structure", "actionability", "tone", "framing", "abstraction") (Not exhaustive and do not force-fit).

Step 2:
Using those dimensions, score how similar the Adapted response is to EACH candidate.
Scores are 0-5:
5 = Near-identical on the key dimensions; Adapted matches candidate's stance/style/structure with no meaningful drift.
4 = Strong match on most key dimensions; minor drift on at most one dimension.
3 = Partial match; aligns on some key dimensions but differs on others OR ambiguity prevents a clear judgment.
2 = Weak match; differs on one or more key dimensions in a way that matters.
1 = Very weak match; mostly reflects the opposite of the candidate on key dimensions.
0 = Opposes/contradicts the candidate on the key dimensions (clear mismatch).

Output valid, parsable JSON only without any extra commentary:
{{
  "dimensions": ["...", "..."],
  "scores": [s0, s1, ..., s{{c-1}}]
}}

Rules:
- scores length MUST equal number of candidates; scores[i] corresponds to candidate i. Do NOT change the candidate order.
- Use the same dimensions for scoring all candidates.

[Current turn]
{current_turn}

[Number of candidates]
c={c}

[Adapted]
{adapted}
"""

GEN_BUDGET = 1024
RESPONSE_MAX_WORD_COUNT = 400
EVAL_BUDGET = 128
logger = logging.getLogger(__name__)

def evaluate_generation(gen_model: BaseLM, conversation_history: List[Turn], profile: str, embed_cfg: EmbedConfig, generation_cfg: Optional[GenerationConfig] = None, eval_model: Optional[BaseLM] = None, evaluation_cfg: Optional[GenerationConfig] = None) -> Dict[str, float]:
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
        r=upper_bound
    )
    if eval_model is None:
        eval_model = gen_model
    if evaluation_cfg is None:
        evaluation_cfg = generation_cfg
    GenSchema = create_model(
        "GenSchema",
        response=(constr(strip_whitespace=True, min_length=1), ...)
    )
    try:
        generate_output = gen_model.generate(
            prompt=generate_prompt,
            schema=GenSchema,
            cfg=generation_cfg,
            max_tokens=GEN_BUDGET
        )["output"]
    except Exception as e:
        logger.exception(f"Generation for evaluation failed at turn {current_turn.turn_id}")
        return {"gpt_score": 0.0, "relative_gpt_score": 0, "similarity_score": 0.0, "relative_score": 0.0, "error": "Generation: " + str(e)}
    adapted_response = generate_output["response"]
    relative_score = relative_similarity_score(adapted_response, current_turn.candidates, current_turn.chosen_idx, embed_cfg)
    similarity_score = text_similarity(adapted_response, current_turn.chosen, embed_cfg)
    c = len(current_turn.candidates)
    if c < 2:
        return {"gpt_score": 2.5, "relative_gpt_score": 0, "similarity_score": 0.0, "relative_score": 0.0, "error": "Evaluation skipped due to insufficient candidates"}
    evaluate_prompt = EVALUATE_PROMPT.format(
        current_turn=current_turn.format(include_candidates=True, include_choice=False),
        adapted=adapted_response,
        c=c
    )
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
        logger.exception("Response evaluation failed")
        return {"gpt_score": 2.5, "relative_gpt_score": 0, "similarity_score": similarity_score, "relative_score": relative_score, "error": "Evaluation: " + str(e)}
    scores = evaluate_output["scores"]
    gpt_score = scores[current_turn.chosen_idx]
    relative_gpt_score = gpt_score - max(scores[i] for i in range(c) if i != current_turn.chosen_idx)
    return {
        "gpt_score": gpt_score,
        "relative_gpt_score": relative_gpt_score,
        "similarity_score": similarity_score,
        "relative_score": relative_score,
    }
