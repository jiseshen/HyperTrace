from pydantic import BaseModel, create_model, conlist, StringConstraints, Field
import logging
from .hypothesis_set import Hypothesis, WorkingBelief
from data import Turn
from .utils import TracerContext
from typing import Any, Dict, List, Literal, Annotated
from prompt.base import INITIALIZATION_PROMPT


UNIT_INITIALIZE_BUDGET = 256
logger = logging.getLogger(__name__)

class HypothesisSchema(BaseModel):
    id: str = Field(..., description="the ID of the hypothesis, either reused from retrieved hypotheses or newly created (e.g., 'new-1')")
    action: Literal["reuse", "new"] = Field(..., description="reuse | new")
    content: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] = Field(..., description="a clear hypothesis describing a latent user preference")
    justification: str = Field(..., description="a brief justification of why this hypothesis explains the chosen response")

def initialize_hypothesis(
    conversation_history: List[Turn],
    candidates: str,
    context: TracerContext
    ) -> Dict[str, Any]:
    """Initialize a working belief with retrieved hypotheses. Return None if skipping this turn."""
    prev_turns = conversation_history[:-1]
    current_turn = conversation_history[-1]
    candidate_hypotheses, _ = context.hypothesis_set.retrieve_hypotheses(
        current_turn.user_message, top_k=context.tracer_config.n_hypotheses
    )
    
    prompt = context.prompts.initialization.format(
        n_hypotheses=context.tracer_config.n_hypotheses,
        prev_turns="\n".join([turn.format(include_candidates=False) for turn in prev_turns[-context.tracer_config.max_history_turns:]]),
        user_message=current_turn.user_message,
        candidates=candidates,
        retrieved_hypotheses="\n".join([
            h.format(include_category=context.tracer_config.use_hypothesis_topics)
            for h in candidate_hypotheses
        ])
    )
    
    InitializeSchema = create_model(
        "InitializeSchema",
        category=(str, Field(..., description="the category of current topic")),
        hypotheses=(conlist(HypothesisSchema, min_length=context.tracer_config.n_hypotheses, max_length=context.tracer_config.n_hypotheses), Field(..., description=f"List of {context.tracer_config.n_hypotheses} hypotheses explaining user preferences in different aspects"))
    )
    budget = UNIT_INITIALIZE_BUDGET * context.tracer_config.n_hypotheses
    try:
        initialize_overrides = context.get_generation_overrides("initialize_override")
        output = context.model.generate(prompt, schema=InitializeSchema, cfg=context.generation_config, max_tokens=budget, **initialize_overrides)["output"]
    except Exception as e:
        logger.exception("Initialization failed")
        return {"success": False, "reason": str(e)}
    new_hypotheses: List[Hypothesis] = []
    reused_hypotheses: List[Hypothesis] = []
    for h in output['hypotheses']:
        if h['action'] == 'reuse' and h["id"] in context.hypothesis_set.hypotheses:
            prev_category = context.hypothesis_set[h['id']][0].category
            if output['category'] not in prev_category:
                new_category = prev_category + ", " + output['category']
            else:
                new_category = prev_category
            reused_hypotheses.append(Hypothesis(id=h['id'], category=new_category, content=h['content']))
        else:
            new_hypotheses.append(Hypothesis(id=h['id'], category=output['category'], content=h['content']))

    if reused_hypotheses:
        context.hypothesis_set.update_hypotheses(reused_hypotheses)
    new_ids = context.hypothesis_set.add_hypotheses(new_hypotheses)
    reused_ids = [h.id for h in reused_hypotheses]
    all_ids = new_ids + reused_ids
    _, priors = context.hypothesis_set.retrieve_hypotheses(all_ids)
    belief = WorkingBelief(
        ids=all_ids,
        priors=priors,
        repo=context.hypothesis_set,
    )
    context.update_belief(belief)
    return {"success": True}
