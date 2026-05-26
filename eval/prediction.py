import logging
import re
from model import GenerationConfig, BaseLM
from data import Turn
from typing import Any, Dict, List, Optional
from pydantic import Field, create_model, conlist
from prompt import PromptSet, prism_prompts
from prompt.base import PREDICTION_PROMPT

PREDICT_PROMPT = PREDICTION_PROMPT

PREDICT_BUDGET = 256
logger = logging.getLogger(__name__)

def predict_choice(
    model: BaseLM,
    conversation_history: List[Turn],
    profile: str,
    generation_cfg: GenerationConfig = None,
    generation_overrides: Optional[Dict[str, Any]] = None,
    prompts: Optional[PromptSet] = None,
    loose_schema: bool = False,
) -> Dict[str, float]:
    prev_turns = conversation_history[:-1]
    current_turn = conversation_history[-1]
    gt_choice = current_turn.chosen_idx + 1
    
    c = len(current_turn.candidates)
    prompt_set = prompts or prism_prompts()
    prompt = prompt_set.prediction.format(
        profile=profile,
        prev_turns='\n\n'.join([turn.format(include_candidates=False) for turn in prev_turns]),
        current_turn=current_turn.format(include_candidates=True, include_choice=False),  # ids are (idx + 1)
        c=c
    )

    ranking_type = list[str] if loose_schema else conlist(str, min_length=c, max_length=c)
    Schema = create_model(
        "PredictSchema",
        ranking=(ranking_type, Field(..., description=f"List of {c} candidate IDs ranked from best to worst")),
        justification=(str, Field(..., description="a brief explanation"))
    )
    cfg = generation_cfg or GenerationConfig()
    overrides = generation_overrides or {}
    prediction = None
    try:
        prediction = model.generate(prompt, schema=Schema, cfg=cfg, max_tokens=PREDICT_BUDGET, **overrides)["output"]
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
    if loose_schema:
        seen = set()
        ranking = [
            item for item in ranking
            if 1 <= item <= c and not (item in seen or seen.add(item))
        ]
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
