import asyncio
from typing import Any, Dict, List, Optional, Tuple, Annotated
from pydantic import BaseModel, Field, conlist, create_model, StringConstraints
import logging
from .utils import TracerContext
from data.base import Turn

SKIP_PROMPT = """
Role:
You are the gating module for an LLM personalization system.

Goal:
Decide whether this turn contains usable preference evidence.

Set "skip": true if either condition is true:
- The user message is only greeting/ack/filler and has no meaningful preference, value, stance, boundary, or constraint signal.
- The candidate differences are not preference-relevant and are mainly correctness/completeness/minor wording differences.

Important:
- Do not skip only because the topic is sensitive or controversial.
- Focus on preference signal, not toxicity level.

Output (JSON only):
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
Role:
You are the preprocessing module for an LLM personalization system.

Goal:
Convert raw candidates into stable, compact summaries that preserve preference-relevant differences.

Step 1:
Identify 1-4 high-contrast dimensions across candidates.
Use short canonical labels (examples: values, information_density, structure, actionability, tone, framing, abstraction).
Only keep dimensions with clear contrast.

Step 2:
For each candidate, produce one concise summary (<50 words).
- Summary must cover the listed dimensions for that candidate.
- You may add at most one extra salient detail.
- Do not restate the whole candidate.

Output (JSON only):
{{
  "reason": "a short justification about the dimension identification",
  "dimensions": ["dimension 1", ...],
  "summarized_candidates": [
    {{"i": 0, "summary": "a brief summary for candidate 0"}},
    {{"i": 1, "summary": "a brief summary for candidate 1"}}
  ]
}}

Rules:
- Return valid JSON only (no markdown, no comments).
- Use only provided candidates.
- Preserve candidate indices exactly.
- Return exactly one summarized candidate per input candidate.

[user_message]
{user_message}

[candidates]
{candidates}

Generate exactly {n} items in summarized_candidates.
"""

SKIP_BUDGET = 128
UNIT_PREPROCESS_BUDGET = 256
logger = logging.getLogger(__name__)

class CandidateSummarySchema(BaseModel):
    i: int = Field(..., description="the integer index of the candidate response")
    summary: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] = Field(..., description="a compact summary of a candidate response")

class SkipSchema(BaseModel):
    reason: str = Field(..., description="a brief justification")
    skip: bool = Field(..., description="whether to skip preference extraction for this turn")

def compact_text(s: str, head: int = 200, tail: int = 100) -> str:
    if len(s) <= head + tail + 3:
        return s
    return f"{s[:head]}...{s[-tail:]}"

def preprocess_candidates(conversation_history: List[Turn], context: TracerContext) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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
        return [], {"success": False, "skip": True, "reason": str(e), "invalid": context.tracer_config.n_hypotheses}
    skip_outputs = [o["output"] if not isinstance(o, Exception) else None for o in async_output]
    skip = 0
    invalid = 0
    for o in skip_outputs:
        if o is None:
            invalid += 1
        elif o['skip']:
            skip += 1
    if skip > len(skip_outputs) / 2:  # Majority vote to skip
        return [], {"success": True, "skip": True, "invalid": invalid}
    n = len(current_turn.candidates)
    preprocess_prompt = PREPROCESSING_PROMPT.format(
        user_message=current_turn.user_message,
        candidates="\n".join([f"[{i}] {c}" for i, c in enumerate(current_turn.candidates)]),
        n=n
    )
    budget = UNIT_PREPROCESS_BUDGET * n

    PreprocessSchema = create_model(
        "PreprocessSchema",
        reason=(Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)], Field(..., description="short reason for selected dimensions")),
        dimensions=(conlist(str, min_length=1, max_length=4), Field(..., description="high-contrast dimensions across candidates")),
        summarized_candidates=(conlist(CandidateSummarySchema, min_length=n, max_length=n), Field(..., description=f"List of {n} summarized candidates")),
    )
    try:
        preprocess_overrides = context.get_generation_overrides("preprocess_override")
        output = context.model.generate(preprocess_prompt, schema=PreprocessSchema, cfg=context.generation_config, max_tokens=budget, **preprocess_overrides)["output"]
    except Exception as e:
        logger.exception("Preprocessing failed")
        return [], {"success": False, "skip": False, "reason": str(e), "invalid": invalid}

    summarized_candidates = output["summarized_candidates"]
    observed_indices = [item["i"] for item in summarized_candidates]
    expected_indices = list(range(n))
    if sorted(observed_indices) != expected_indices:
        reason = f"Summarized candidate indices mismatch: expected {expected_indices}, got {observed_indices}"
        logger.error(reason)
        return [], {"success": False, "skip": False, "reason": reason, "invalid": invalid}

    summary_by_index = {item["i"]: item["summary"] for item in summarized_candidates}
    structured_candidates = []
    for i, candidate in enumerate(current_turn.candidates):
        structured_candidates.append(
            {
                "i": i,
                "summary": summary_by_index[i],
                "choice": "chosen" if i == current_turn.chosen_idx else "rejected",
                "content": compact_text(candidate),
            }
        )
    return structured_candidates, {
        "success": True,
        "skip": False,
        "invalid": invalid,
        "reason": output["reason"],
        "dimensions": output["dimensions"],
    }
