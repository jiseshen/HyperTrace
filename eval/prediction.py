"""
Given a user's preference profile and a conversation history,
prompt the model to rank candidate responses for the new message,
and evaluate the ranking against the ground truth choice,
as an online evaluation of the personalization method.

Output format:
{
    "accuracy": 0 or 1,
    "ranking_score": the linear ranking score of the prediction
}
"""

import json
from model import GenerationConfig, BaseLM
from data import Turn
from typing import Dict, List
import numpy as np

PREDICT_PROMPT = """
You are ranking candidate responses for a user based on a given user preference profile.

Given:
- User preference profile: a concise summary of the user's stable preferences, values, and communication style
- Conversation history (optional)
- Current user message
- Candidate responses with ids

Internally:
- Compare each candidate response.
- Consider observable differences only.
- If the user profile is non-empty, evaluate alignment strictly based on explicit signals in the profile.
- Do NOT speculate about hidden motivations.
- If the profile is empty or clearly irrelevant to this turn, rank candidates based on overall quality, clarity, and usefulness.

Possible dimensions (do not force-fit):
- Underlying values
- Information density
- Structure
- Level of abstraction
- Framing
- Actionability
- Tone

Then:
Rank candidates from best to worst according to alignment (or overall quality if alignment is not applicable).

Output Format:

Return a JSON object:
{{
  "reason": "Brief explanation (2-3 sentences). Mention only concrete observable differences.",
  "ranking": ["id1", "id2", ...]
}}

Rules:
- Output valid JSON only.
- Use the provided candidate ids.
- Do not include extra commentary.
- Do not invent preference signals not present in the profile.

User preference profile:
{profile}

Conversation history:
{prev_turns}

Current interaction:
{current_turn}
"""

def predict_choice(model: BaseLM, conversation_history: List[Turn], profile: str, generation_cfg: GenerationConfig = None) -> Dict[str, float]:
    prev_turns = conversation_history[:-1]
    current_turn = conversation_history[-1]
    gt_choice = current_turn.chosen_idx + 1
    
    prompt = PREDICT_PROMPT.format(
        profile=profile,
        prev_turns='\n\n'.join([turn.format(include_candidates=False) for turn in prev_turns]),
        current_turn=current_turn.format(include_candidates=True, include_choice=False)  # ids are (idx + 1)
    )
    
    retries = 0
    while True:
        try:
            prediction = model.generate(prompt, generation_config=generation_cfg)["output"]
            prediction_data = json.loads(prediction)
            ranking = prediction_data.get('ranking')
            rank = ranking.index(gt_choice) + 1
            ranking_score = (len(ranking) - rank) / (len(ranking) - 1)
            accuracy = 1.0 if rank == 1 else 0.0
            return {
                "accuracy": accuracy,
                "ranking_score": ranking_score,
            }
        except Exception as e:
            retries += 1
            if retries > generation_cfg.max_retries:
                raise ValueError(f"Failed to parse model output after {generation_cfg.max_retries} attempts. Prompt: {prompt}. Last output: {prediction} Error: {e}.")