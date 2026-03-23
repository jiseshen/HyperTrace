import asyncio
from typing import Any, Dict, List, Optional, Tuple, Annotated
from pydantic import BaseModel, Field, conlist, create_model, StringConstraints
import logging
from .utils import TracerContext
from data.base import Turn

SKIP_PROMPT = """
You are a gating engine for an LLM personalization system.
Your task is to determine whether the current user interaction contains preference-relevant signals worth extracting later.

Important:
- If the current user message is merely a greeting, acknowledgment, filler, or other phatic/social utterance with no substantive preference, value, stance, boundary, or constraint signal, then return skip=true, even if the candidate responses differ stylistically.
- Do NOT skip merely because the content is sensitive, offensive, taboo, controversial, or politically charged. Such interactions may still contain important signals about values, stance, tone tolerance, safety boundaries, or preferred response framing.

Return skip=true if EITHER holds:
(A) The current user message is only a greeting / acknowledgment / filler and does not provide a meaningful preference, value, stance, boundary, or constraint signal.
(B) Although the message is substantive, the differences among candidates are not meaningfully preference-relevant, and are mainly about correctness, completeness, or minor wording differences rather than tone, framing, values, structure, boundaries, or response strategy.

Output JSON only:
{{
  "reason": "a brief (1-2 sentences) justification about why to skip or not",
  "skip": boolean
}}

[user_message]
{user_message}

[candidates]
{candidates}
"""

PREPROCESSING_PROMPT = """
You are a preprocessing engine for an LLM personalization system.
Convert raw candidate responses into a stable, compact representation that highlights preference-relevant differences.

Inputs:
- Current user message
- A list of candidate responses

Step 1 — Dimensions
Identify 1-4 key dimensions that clearly differ across candidates.
Use short canonical labels (e.g., "values", "information_density", "structure", "actionability", "tone", "framing", "abstraction") (Not exhaustive and do not force-fit).
Only include dimensions with obvious contrast.

Step 2 — Candidate previews
For each candidate, output a compact preview (<50 words) aligned to the contrasting dimensions.
- The preview should focus on the most salient differences.
- For each dimension listed in "dimensions", the preview must address how this candidate differs on that dimension.
- You may add at most ONE extra detail not covered by the dimensions.
You may cite representative parts of the candidate to highlight the differences, but do NOT restate the whole candidate.

Output JSON only:
{{
  "reason": "a short justification about the dimension identification",
  "dimensions": ["dimension 1", ...],
  "processed_candidates": [
    {{"i": 0, "preview": "a brief preview for candidate 0"}},
    {{"i": 1, "preview": "a brief preview for candidate 1"}}
  ]
}}

Rules:
- Return valid, parsable JSON only. Do not include any commentary, markdown, or comments.
- Use only the candidates explicitly provided in the input.
- Preserve the original candidate order and indices exactly as given.
- Return exactly one processed candidate item for each input candidate, and do not add, remove, or invent candidates.

[user_message]
{user_message}

[candidates]
{candidates}

Generate exactly {n} items in processed_candidates.
"""

SKIP_BUDGET = 128
UNIT_PREPROCESS_BUDGET = 256
logger = logging.getLogger(__name__)

class CandidateSchema(BaseModel):
    i: int = Field(..., description="the integer index of the candidate response")
    preview: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] = Field(..., description="a compact preview of a candidate response")

class SkipSchema(BaseModel):
    reason: str = Field(..., description="a brief justification")
    skip: bool = Field(..., description="whether to skip preference extraction for this turn")

def compact_text(s: str, head: int = 200, tail: int = 100) -> str:
    if len(s) <= head + tail + 3:
        return s
    return f"{s[:head]}...{s[-tail:]}"

def preprocess_candidates(conversation_history: List[Turn], context: TracerContext) -> Tuple[str, Optional[Dict[str, Any]]]:
    current_turn = conversation_history[-1]
    skip_prompt = SKIP_PROMPT.format(
        user_message=current_turn.user_message,
        candidates="\n".join([f"[{i}] {c}" for i, c in enumerate(current_turn.candidates)])
    )
    try:
        skip_overrides = context.get_generation_overrides("skip_override")
        async_output = asyncio.run(
            context.model.async_generate([skip_prompt for _ in range(context.tracer_config.n_hypotheses)], schema=SkipSchema, cfg=context.generation_config, max_tokens=SKIP_BUDGET, **skip_overrides)
        )
    except Exception as e:
        logger.exception("Skip generation failed")
        return "", {"success": False, "skip": True, "reason": str(e), "invalid": context.tracer_config.n_hypotheses}
    skip_outputs = [o["output"] if not isinstance(o, Exception) else None for o in async_output]
    skip = 0
    invalid = 0
    for o in skip_outputs:
        if o is None:
            invalid += 1
        elif o['skip']:
            skip += 1
    if skip > len(skip_outputs) / 2:  # Majority vote to skip
        return "", {"success": True, "skip": True, "invalid": invalid}
    n = len(current_turn.candidates)
    preprocess_prompt = PREPROCESSING_PROMPT.format(
        user_message=current_turn.user_message,
        candidates="\n".join([f"[{i}] {c}" for i, c in enumerate(current_turn.candidates)]),
        n=n
    )
    budget = UNIT_PREPROCESS_BUDGET * n

    PreprocessSchema = create_model(
        "PreprocessSchema",
        processed_candidates=(conlist(CandidateSchema, min_length=n, max_length=n), Field(..., description=f"List of {n} processed candidates"))
    )
    try:
        preprocess_overrides = context.get_generation_overrides("preprocess_override")
        output = context.model.generate(preprocess_prompt, schema=PreprocessSchema, cfg=context.generation_config, max_tokens=budget, **preprocess_overrides)["output"]
    except Exception as e:
        logger.exception("Preprocessing failed")
        return "", {"success": False, "skip": False, "reason": str(e), "invalid": invalid}
    else:
        previews = [c["preview"] for c in output["processed_candidates"]]
        lines = []
        for i, (preview, candidate) in enumerate(zip(previews, current_turn.candidates)):
            marker = "[CHOSEN]" if i == current_turn.chosen_idx else "[REJECTED]"
            lines.append(f"[{i}] {marker} Preview: {preview} Content: {compact_text(candidate)}")
        return "\n".join(lines), {"success": True, "skip": False, "invalid": invalid}
