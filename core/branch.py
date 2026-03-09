from hypothesis_set import WorkingBelief, Update
from pydantic import BaseModel
from .utils import TracerContext, compute_importance
from data import Turn
from typing import Any, Dict, List, Literal, Optional
from .initialize import initialize_hypothesis


BRANCHING_PROMPT = """
You are updating ONE working hypothesis in a personalization pipeline.

You are given:
- The current hypothesis (category + content)
- The latest user interaction: user message + candidate responses (chosen vs rejected)
- (Optional) brief conversation history from previous turns

Goal:
Decide whether this interaction provides usable preference evidence for updating this hypothesis, and if so, how.

Step 1: Relevance
Assess whether the interaction is relevant to this hypothesis:
- relevance="direct" if the interaction directly supports or contradicts this hypothesis.
- relevance="partial" if topic differs but the same underlying preference/value/style is evidenced.
- relevance="none" if no meaningful support exists.

Step 2: Update
Internally:
- Compare chosen vs rejected to extract preference-relevant differences.
- If user message acts as a follow-up or feedback to previous interaction, check for:
  * Explicit corrections
  * Stated preferences
  * Constraints
  * Dissatisfaction signals
  * Consistent behavioral patterns

Update decision:
- If relevance is "direct" or "partial":
    action="revise"
    Produce an updated hypothesis that:
      * Incorporates new evidence,
      * Reduces inconsistency,
      * Remains specific (do not over-generalize),
      * Keeps the original intent unless clearly contradicted.
- If relevance is "none":
    action="replace"
    Produce a new hypothesis at similar specificity.

Category update rules:
- If the current turn provides clear evidence that the topic category has shifted, update the category accordingly.
- Otherwise, keep the original category.
- If the original category contains multiple labels, select the single most appropriate one based on the current evidence.
- Do not introduce a new category unless clearly justified by the interaction.

Output JSON only:

{{
  "action": "revise" | "replace",
  "relevance": "direct" | "partial" | "none",
  "updated_hypothesis": {{
    "category": "string",
    "content": "string"
  }},
  "evidence": "one short justification"
}}

Rules:
- If action="revise": category should usually remain the same.
- If action="replace": do not reference the old hypothesis in the new content.
- Ground all updates strictly in observed evidence.
- Avoid speculation and over-generalization.

Conversation History:
{prev_turns}

Current User Message:
{user_message}

Candidate Responses:
{candidates}

Current Hypothesis:
{current_hypothesis}
"""

class UpdatedHypothesisSchema(BaseModel):
    category: str
    content: str

class BranchSchema(BaseModel):
    action: Literal["revise", "replace"]
    relevance: Literal["direct", "partial", "none"]
    updated_hypothesis: UpdatedHypothesisSchema

def branch_hypotheses(conversation_history: List[Turn], candidates: str, context: TracerContext) -> Optional[Dict[str, Any]]:
    """Propagate hypotheses based on new user message. Skip if no usable evidence, reinitialize if irrelevant, or simply revise."""
    prev_turns = conversation_history[:-1]
    current_turn = conversation_history[-1]
    current_hypotheses, current_weights = context.belief[:]
    prompts = [
        BRANCHING_PROMPT.format(
            prev_turns="\n\n".join([turn.format(include_candidates=False) for turn in prev_turns[-context.tracer_config.max_history_turns:]]),
            user_message=current_turn.user_message,
            candidates=candidates,
            current_hypothesis=h.format()
        ) for h in current_hypotheses
    ]
    outputs = [o["output"]  if not isinstance(o, Exception) else None for o in context.model.async_generate(prompts, schema=BranchSchema, cfg=context.generation_config)]
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
    context.hypothesis_set.update_hypotheses(updates)
    new_ids = context.hypothesis_set.add_hypotheses(new)
    all_ids = revised_ids + new_ids
    all_weights = revised_weights + [0.5 for _ in new_ids]
    context.update_belief(WorkingBelief(ids=all_ids, priors=all_weights, repo=context.hypothesis_set))
    return {"replace": replace, "invalid": invalid}