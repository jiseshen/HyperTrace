import logging
import re
from model import GenerationConfig, BaseLM
from data import Turn
from typing import Dict, List
from pydantic import Field, create_model, conlist

PREDICT_PROMPT = """
You are ranking candidate responses for a user based on a given user preference profile.

Given:
- User preference profile: a concise summary of the user's stable preferences, values, and communication style
- Conversation history (optional)
- Current user message
- Candidate responses. Each candidate has a unique ID in square brackets, such as [C1], [C2].

Internally:
- Compare each candidate response over how likely the user is to prefer each one.
- Evaluate alignment based on explicit signals in the profile.
- If the profile is empty or clearly irrelevant to this turn, rank candidates based on overall quality, clarity, and usefulness.

Then:
Rank candidates from best to worst according to alignment to the user preference.

Output Format:

Return a JSON object:
{{
  "ranking": ["id1", "id2"],
  "justification": "a brief (2-3 sentences) explanation of the ranking"
}}

Rules:
- Output valid, parsable JSON only, without any extra commentary.
- When referring to candidates in your output, use only their IDs (e.g., C1, C2).
- Keep the number in the ranking exactly the same as the number of given candidates.
- Do not invent preference signals not present in the profile.

User preference profile:
{profile}

Conversation history:
{prev_turns}

Current interaction:
{current_turn}

Number of candidates: {c}
"""

PREDICT_BUDGET = 128
logger = logging.getLogger(__name__)

def predict_choice(model: BaseLM, conversation_history: List[Turn], profile: str, generation_cfg: GenerationConfig = None) -> Dict[str, float]:
    prev_turns = conversation_history[:-1]
    current_turn = conversation_history[-1]
    gt_choice = current_turn.chosen_idx + 1
    
    c = len(current_turn.candidates)
    prompt = PREDICT_PROMPT.format(
        profile=profile,
        prev_turns='\n\n'.join([turn.format(include_candidates=False) for turn in prev_turns]),
        current_turn=current_turn.format(include_candidates=True, include_choice=False),  # ids are (idx + 1)
        c=c
    )

    Schema = create_model(
        "PredictSchema", 
        ranking=(conlist(str, min_length=c, max_length=c), Field(..., description=f"List of {c} candidate IDs ranked from best to worst")),
        justification=(str, Field(..., description="a brief explanation"))
    )
    cfg = generation_cfg or GenerationConfig()
    prediction = None
    try:
        prediction = model.generate(prompt, schema=Schema, cfg=cfg, max_tokens=PREDICT_BUDGET)["output"]
    except Exception as e:
        logger.exception("Prediction generation failed.")
        return {"success": False, "accuracy": 0.0, "ranking_score": 0.5, "reason": str(e)}
    ranking = []
    for r in prediction["ranking"]:
        m = re.search(r"C(\d+)", r)
        try:
            ranking.append(int(m.group(1)))
        except Exception:
            logger.exception("ID unmatched: " + r)
    if gt_choice in ranking:
        rank = ranking.index(gt_choice) + 1
        ranking_score = 1.0 if len(ranking) == 1 else (len(ranking) - rank) / (len(ranking) - 1)
        accuracy = 1.0 if rank == 1 else 0.0
        return {
            "success": True,
            "accuracy": accuracy,
            "ranking_score": ranking_score,
        }
    return {"success": False, "accuracy": 0.0, "ranking_score": 0.5, "reason": "Ground truth choice ID not found in the predicted ranking."}