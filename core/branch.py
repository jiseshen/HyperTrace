import asyncio
import logging
from .hypothesis_set import WorkingBelief, Update
from pydantic import BaseModel, Field
from .utils import TracerContext
from data import Turn
from typing import Any, Dict, List, Literal, Optional
from .initialize import initialize_hypothesis
from .consolidate import compute_importance
from prompt.base import BRANCHING_PROMPT


BRANCH_BUDGET = 256
logger = logging.getLogger(__name__)

class UpdatedHypothesisSchema(BaseModel):
    category: str = Field(..., description="the category of current topic")
    content: str = Field(..., description="the updated hypothesis content")

class BranchSchema(BaseModel):
    action: Literal["revise", "replace"] = Field(..., description="revise | replace")
    relevance: Literal["direct", "partial", "none"] = Field(..., description="direct | partial | none")
    justification: str = Field(..., description="One brief (1-2 sentences) justification of the update decision")
    updated_hypothesis: UpdatedHypothesisSchema

def branch_hypotheses(conversation_history: List[Turn], candidates: str, context: TracerContext) -> Optional[Dict[str, Any]]:
    """Propagate hypotheses based on new user message. Skip if no usable evidence, reinitialize if irrelevant, or simply revise."""
    prev_turns = conversation_history[:-1]
    current_turn = conversation_history[-1]
    current_hypotheses, current_weights = context.belief[:]
    prompts = [
        context.prompts.branching.format(
            prev_turns="\n\n".join([turn.format(include_candidates=False) for turn in prev_turns[-context.tracer_config.max_history_turns:]]),
            user_message=current_turn.user_message,
            candidates=candidates,
            current_hypothesis=h.format()
        ) for h in current_hypotheses
    ]
    try:
        branch_overrides = context.get_generation_overrides("branch_override")
        async_outputs = asyncio.run(
            context.model.async_generate(
                prompts, schema=BranchSchema, cfg=context.generation_config, max_tokens=BRANCH_BUDGET, **branch_overrides
            )
        )
    except Exception:
        logger.exception("Branching generation failed")
        async_outputs = [None for _ in prompts]
    outputs = [o["output"] if not isinstance(o, Exception) else None for o in async_outputs]
    replace, invalid = 0, 0
    for output in outputs:
        if output and output['action'] == 'replace':
            replace += 1
        if output is None:
            invalid += 1
    if replace > len(outputs) / 2:  # Vote to reinitialize
        context.belief.consolidate(compute_importance(len(conversation_history), context.current_belief.normalized_entropy()), context.tracer_config.consolidate_alpha)
        conversation_history[:] = conversation_history[-1:]  # Treat as a new conversation
        init_status = initialize_hypothesis(conversation_history, candidates, context)
        return {"reinit": True, **init_status, "replace": replace, "invalid": invalid}
    updates = []
    revised_ids = []
    revised_weights = []
    replaced_ids = []
    replaced_weights = []
    new = []
    for i, (h, o) in enumerate(zip(current_hypotheses, outputs)):
        if o is None:
            updates.append(Update(id=h.id, likelihood=0.5))
            revised_ids.append(h.id)
            revised_weights.append(current_weights[i])
            continue
        if o['action'] == 'revise':
            new_cat = o['updated_hypothesis']['category']
            new_cat = h.category if new_cat in h.category else h.category + ", " + new_cat
            updates.append(Update(h.id, category=new_cat, content=o['updated_hypothesis']['content']))
            revised_ids.append(h.id)
            revised_weights.append(current_weights[i])
        elif o['action'] == 'replace':
            new.append(o['updated_hypothesis'])
            replaced_ids.append(h.id)
            replaced_weights.append(current_weights[i])
    if replace > 0:
        context.hypothesis_set.consolidate_belief(replaced_ids, replaced_weights, compute_importance(len(conversation_history), context.current_belief.normalized_entropy()), context.tracer_config.consolidate_alpha)
    if updates:
        context.hypothesis_set.update_hypotheses(updates)
    new_ids = context.hypothesis_set.add_hypotheses(new)
    all_ids = revised_ids + new_ids
    all_weights = revised_weights + [0.5 for _ in new_ids]
    context.update_belief(WorkingBelief(ids=all_ids, priors=all_weights, repo=context.hypothesis_set))
    return {"replace": replace, "invalid": invalid}
