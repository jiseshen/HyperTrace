from typing import Dict, List, Optional
from model import BaseLM, GenerationConfig
from pydantic import Field, create_model, confloat, conlist
from data import Turn
from .utils import text_similarity, relative_similarity_score, EmbedConfig
import logging

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
  "scores": [s0, s1, ..., s{{c-1}}],
  "justification": "a brief (1-2 sentences) explanation of the key similarities/differences between the Adapted response and each candidates."
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


EVAL_BUDGET = 384
logger = logging.getLogger(__name__)


def evaluate_generation(
    eval_model: BaseLM,
    conversation_history: List[Turn],
    adapted_response: str,
    embed_cfg: EmbedConfig,
    evaluation_cfg: Optional[GenerationConfig] = None,
) -> Dict[str, float]:
    current_turn = conversation_history[-1]
    relative_score = relative_similarity_score(adapted_response, current_turn.candidates, current_turn.chosen_idx, embed_cfg)
    similarity_score = text_similarity(adapted_response, current_turn.chosen, embed_cfg)
    c = len(current_turn.candidates)
    if c < 2:
        return {"gpt_score": 2.5, "relative_gpt_score": 0, "similarity_score": 0.0, "relative_score": 0.0, "error": "Evaluation skipped due to insufficient candidates"}
    evaluate_prompt = EVALUATE_PROMPT.format(
        current_turn=current_turn.format(include_candidates=True, include_choice=False),
        adapted=adapted_response,
        c=c,
    )
    EvalSchema = create_model(
        "EvalSchema",
        dimensions=(list[str], Field(..., description="List of key dimensions used for evaluation")),
        scores=(conlist(confloat(ge=0, le=5), min_length=c, max_length=c), Field(..., description=f"List of {c} similarity scores in 0-5 for each candidate response")),
        justification=(str, Field(..., description="a brief explanation")),
    )
    try:
        evaluate_output = eval_model.generate(
            prompt=evaluate_prompt,
            schema=EvalSchema,
            cfg=evaluation_cfg,
            max_tokens=EVAL_BUDGET,
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
