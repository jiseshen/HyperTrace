from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, create_model, conlist
from core.utils import TracerContext
from data.base import Turn


PREPROCESSING_PROMPT = """
You are a preprocessing engine for an LLM personalization system.
Convert raw candidate responses into a stable, compact representation that highlights preference-relevant differences.

Inputs:
- user_message
- candidates (list of responses)

Step 1 — Gate (skip decision)
Return skip=true if BOTH hold:
(A) The user message contains no actionable preference signal (e.g., greeting/ack only, or no constraints/refinements), AND
(B) Candidate differences are NOT preference-relevant, meaning they are mainly correctness/completeness with no clear stylistic/structural/value contrast.

If skip=true:
- rationale: one short sentence
- dimensions: []
- candidates: []

Step 2 — Dimensions (if not skipped)
Identify 1-4 key dimensions that clearly differ across candidates.
Use short canonical labels (e.g., "values", "information_density", "structure", "actionability", "tone", "framing", "abstraction") (Not exhaustive and do not force-fit).
Only include dimensions with obvious contrast.

Step 3 — Candidate previews (if not skipped)
For each candidate, output a compact preview (<50 words) aligned to the contrasting dimensions.
- The preview should focus on the most salient differences.
- For each dimension listed in "dimensions", the preview must address how this candidate differs on that dimension.
- You may add at most ONE extra detail not covered by the dimensions.
You may cite representative parts of the candidate to highlight the differences, but do NOT restate the whole candidate.

Output JSON only:
{{
  "rationale": "a short justification about the skip decision and dimension identification",
  "skip": boolean,
  "dimensions": ["dimension 1", ...],
  "candidates": [
    {{"i": 0, "preview": "a brief preview for candidate 0"}},
    {{"i": 1, "preview": "a brief preview for candidate 1"}},
    ...
  ]
}}

Rules:
- Valid JSON only.
- If skip=true, candidates must be [] and dimensions must be [].
- You must keep the order of candidates as given.

[user_message]
{user_message}

[candidates]
{candidates}
"""

class CandidateSchema(BaseModel):
    i: int
    preview: str
    
class SkipSchema(BaseModel):
    skip: Literal[True]

def preprocess_candidates(conversation_history: List[Turn], context: TracerContext) -> Tuple[str, Optional[Dict[str, Any]]]:
    current_turn = conversation_history[-1]
    preprocess_prompt = PREPROCESSING_PROMPT.format(
        user_message=current_turn.user_message,
        candidates="\n".join([f"{i}. {c}" for i, c in enumerate(current_turn.candidates)])
    )
    n = len(current_turn.candidates)
    Schema = Union[create_model(
        "PreprocessSchema",
        skip=(Literal[True], ...),
        candidates=(conlist(CandidateSchema, min_length=n, max_length=n), ...),
    )]
    try:
        output = context.model.generate(preprocess_prompt, schema=Schema, cfg=context.generation_config)["output"]
    except Exception as e:
        print(f"Preprocessing failed with error: {e}")
        return "", {"success": False, "reason": str(e)}
    if output["skip"]:
        return "", {"success": True, "skip": True, "reason": output["rationale"]}
    else:
        previews = [c["preview"] for c in output["candidates"]]
        return "\n".join(
            [f"{i}. {"[CHOSEN]" if i == current_turn.chosen_idx else "[REJECTED]"} Preview: {preview} Content: {candidate[:50]}...{candidate[-50:]}" for i, (preview, candidate) in enumerate(zip(previews, current_turn.candidates))]
        ), {"success": True, "skip": False}