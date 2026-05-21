import asyncio
import time
from typing import Any, Dict, List, Tuple, Annotated
from pydantic import BaseModel, Field, conlist, create_model, StringConstraints
import logging
from .utils import TracerContext
from data.base import Turn
from prompt.base import PREPROCESSING_PROMPT, SKIP_PROMPT

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
    invalid = 0
    skip = 0
    if context.tracer_config.allow_skip:
        skip_prompt = context.prompts.skip.format(
            user_message=current_turn.user_message,
            candidates="\n".join([f"[{i}] {c}" for i, c in enumerate(current_turn.candidates)])
        )
        try:
            skip_overrides = context.get_generation_overrides("skip_override")
            async_output = asyncio.run(
                context.model.async_generate([skip_prompt for _ in range(context.tracer_config.n_hypotheses)], schema=SkipSchema, cfg=context.generation_config, max_tokens=SKIP_BUDGET, **skip_overrides)
            )
        except Exception as e:
            logger.exception("Skip generation failed; continuing to preprocessing instead of dropping the turn")
            invalid = context.tracer_config.n_hypotheses
        else:
            skip_outputs = [o["output"] if not isinstance(o, Exception) else None for o in async_output]
            for o in skip_outputs:
                if o is None:
                    invalid += 1
                elif o['skip']:
                    skip += 1
            if skip > len(skip_outputs) / 2:  # Majority vote to skip
                return [], {"success": True, "skip": True, "invalid": invalid, "skip_votes": skip, "total_votes": len(skip_outputs)}
    n = len(current_turn.candidates)
    preprocess_prompt = context.prompts.preprocessing.format(
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
    expected_indices = list(range(n))
    preprocess_overrides = context.get_generation_overrides("preprocess_override")
    preprocess_attempts = max(1, context.generation_config.max_retries)
    output = None
    last_reason = ""
    for attempt in range(preprocess_attempts):
        try:
            output = context.model.generate(
                preprocess_prompt,
                schema=PreprocessSchema,
                cfg=context.generation_config,
                max_tokens=budget,
                **preprocess_overrides,
            )["output"]
        except Exception as e:
            logger.exception("Preprocessing failed")
            return [], {"success": False, "skip": False, "reason": str(e), "invalid": invalid}

        summarized_candidates = output["summarized_candidates"]
        observed_indices = [item["i"] for item in summarized_candidates]
        if sorted(observed_indices) == expected_indices:
            break

        last_reason = f"Summarized candidate indices mismatch: expected {expected_indices}, got {observed_indices}"
        logger.warning("%s (preprocess attempt %s/%s)", last_reason, attempt + 1, preprocess_attempts)
        output = None
        if attempt < preprocess_attempts - 1:
            time.sleep(context.generation_config.retry_delay)

    if output is None:
        return [], {"success": False, "skip": False, "reason": last_reason, "invalid": invalid + 1}

    summarized_candidates = output["summarized_candidates"]
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
        "skip_votes": skip,
        "total_votes": context.tracer_config.n_hypotheses if context.tracer_config.allow_skip else 0,
        "reason": output["reason"],
        "dimensions": output["dimensions"],
    }
