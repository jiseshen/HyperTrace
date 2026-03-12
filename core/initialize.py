from pydantic import BaseModel, create_model, conlist
import logging

from .hypothesis_set import Hypothesis, WorkingBelief
from data import Turn
from .utils import TracerContext
from typing import Any, Dict, List, Literal, Optional


INITIALIZATION_PROMPT = """
You are initializing user preference hypotheses for a personalization pipeline. These hypotheses will guide future generation. Focus only on stable, evidence-supported signals from the most recent comparison between chosen and rejected responses.

Task

Given:
- Conversation history (may be empty for first turn)
- The latest user message
- Candidate responses (whether chosen or rejected specified)
- Previously retrieved hypotheses (may be empty)

Internally:
- Analyze primarily the differences among the candidate responses.
- For hypotheses, focus on conversational style and underlying value preferences rather than specific topics.
- Use the user message only as auxiliary evidence, when it clearly provides explicit feedback to previous interaction or preference signals.

Then:

1. Identify the category/topic of the current conversation.
   This category is used only for organizing and retrieving hypotheses in the library.

2. Propose exactly {n_hypotheses} stable user preference hypotheses:
   - Each hypothesis should focus on a different aspect of preference or value that could explain the user's choice.
   - Reuse and revise relevant retrieved hypotheses when appropriate.
   - Otherwise generate new hypotheses.

Output Format

Return a JSON object:

{{
  "category": "string",
  "hypotheses": [
    {{
      "id": "string (reuse existing ID or 'new-1')",
      "action": "reuse" | "new",
      "content": "revised or new hypothesis content",
      "evidence": "short justification (1-2 sentences) citing specific candidate differences or user message cues that support this hypothesis"
    }}
  ]
}}

Rules:
- Output valid, parsable JSON only.
- Always include all required fields, and produce exactly {n_hypotheses} hypotheses.
- Ground each hypothesis in explicit evidence from the comparison.

Conversation History:
{prev_turns}

Current User Message:
{user_message}

Candidate Responses:
{candidates}

Previously Retrieved Hypotheses:
{retrieved_hypotheses}
"""

UNIT_INITIALIZE_BUDGET = 256
logger = logging.getLogger(__name__)

class HypothesisSchema(BaseModel):
    id: str
    action: Literal["reuse", "new"]
    content: str

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
    
    prompt = INITIALIZATION_PROMPT.format(
        n_hypotheses=context.tracer_config.n_hypotheses,
        prev_turns="\n".join([turn.format(include_candidates=False) for turn in prev_turns[-context.tracer_config.max_history_turns:]]),
        user_message=current_turn.user_message,
        candidates=candidates,
        retrieved_hypotheses="\n".join([h.format() for h in candidate_hypotheses])
    )
    
    InitializeSchema = create_model(
        "InitializeSchema",
        category=(str, ...),
        hypotheses=(conlist(HypothesisSchema, min_length=context.tracer_config.n_hypotheses, max_length=context.tracer_config.n_hypotheses), ...)
    )
    budget = UNIT_INITIALIZE_BUDGET * context.tracer_config.n_hypotheses
    try:
        output = context.model.generate(prompt, schema=InitializeSchema, cfg=context.generation_config, max_tokens=budget)["output"]
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
