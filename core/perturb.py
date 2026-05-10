from typing import Any, Dict, List, Optional, Tuple, Annotated
import logging
from pydantic import BaseModel, Field, create_model, conlist, StringConstraints
from core.utils import TracerContext
from core.hypothesis_set import WorkingBelief
from data.base import Turn

AXIS_PROMPT = """
Role:
You extract explanatory axes for a list of user preference hypotheses.

Goal:
For each hypothesis, output the key latent axis it represents.

Output:
Raw text only, as a comma-separated list:
"axis for hypothesis 1", "axis for hypothesis 2", ...

Rules:
- Use short noun phrases only.
- No explanations, no extra text.
- Hypotheses may share the same axis.

[Hypotheses]
{hypotheses}
"""

MERGE_PROMPT = """
Role:
You merge a cluster of highly similar user preference hypotheses.

Goal:
Produce one canonical hypothesis.

Rules:
- Preserve stable components shared by the cluster.
- Remove stylistic rephrasing and redundancy.
- Keep it specific and evidence-grounded.
- Do not invent new preferences.

Output:
Raw merged hypothesis text only. No explanation.

[CollapsedCluster]
{collapsed_cluster}
"""

PERTURB_PROMPT = """
Role:
You generate rejuvenation hypotheses for a Sequential Monte Carlo personalization system.

Goal:
Generate K plausible new hypotheses that introduce new explanatory axes.

Requirements:
1. Identify one topic category for the current conversation.
2. Generate exactly K new hypotheses.
- Each hypothesis must reflect a distinct latent explanation (not surface rephrasing).
- Each hypothesis must introduce an axis not already in GlobalDiversitySummary.
- Each hypothesis must remain plausible under the conversation evidence.

Output (JSON only):
{{
  "category": "a concise topic label for the current conversation",
  "new_hypotheses": [
    {{"content": "...", "novel_axis": "...", "justification": "..."}}
  ]
}}

[ConversationHistory]
{conversation_history}

[CurrentUserMessage]
{user_message}

[CandidateResponses]
{candidates}

[GlobalDiversitySummary]
{global_axes_summary}

K={K}
"""

class PerturbedHypothesisSchema(BaseModel):
    content: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] = Field(..., description="a perturbed hypothesis describing a latent user preference")
    novel_axis: str = Field(..., description="the new axis name")
    justification: str = Field(..., description="a brief justification")

AXIS_BUDGET = 64
MERGE_BUDGET = 256
UNIT_PERTURB_BUDGET = 256
logger = logging.getLogger(__name__)


def perturb_hypotheses(conversation_history: List[Turn], candidates: str, similar_groups: List[List[int]], context: TracerContext) -> Dict[str, Any]:
    hypotheses = context.belief.get_hypotheses()
    axes_prompt = AXIS_PROMPT.format(hypotheses="\n\n".join([h.content for h in hypotheses]))
    axis_overrides = context.get_generation_overrides("axis_override")
    axes = context.model.generate(axes_prompt, cfg=context.generation_config, max_tokens=AXIS_BUDGET, **axis_overrides)["output"]
    perturbed_hids, perturbed_weights = [], []
    invalid = 0
    for group in similar_groups:
        new_hids, new_weights, new_axes = perturb_group(group=group, axes=axes, conversation_history=conversation_history, candidates=candidates, context=context)
        perturbed_hids.extend(new_hids)
        perturbed_weights.extend(new_weights)
        axes += f", {new_axes}" if new_axes else ""
        if new_axes is None:
            invalid += 1
    perturbed_belief = WorkingBelief(ids=perturbed_hids, priors=perturbed_weights, repo=context.hypothesis_set)
    context.update_belief(perturbed_belief)
    return {"groups": [group for group in similar_groups if len(group) > 1], "invalid": invalid}
    
def perturb_group(group: List[int], axes: str, conversation_history: List[Turn], candidates: str, context: TracerContext) -> Tuple[List[str], List[float], Optional[str]]:
    if len(group) == 1:
        hyps, weights = context.belief[group]
        return [hyps[0].id], weights.tolist(), ""
    prev_turns = conversation_history[-context.tracer_config.max_history_turns:]
    current_turn = conversation_history[-1]
    hypotheses, weights = context.belief[group]
    total_weight = weights.sum()
    merged_prior = max([context.hypothesis_set.global_prior[h.id] for h in hypotheses])
    category = max([h.category for h in hypotheses], key=lambda c: c.count(",") if c else 0)
    K = len(group) - 1
    merge_prompt = MERGE_PROMPT.format(collapsed_cluster="\n\n".join([h.content for h in hypotheses]))
    merge_overrides = context.get_generation_overrides("merge_override")
    merged_hypothesis = hypotheses[0].content if len(set(h.id for h in hypotheses)) == 1 else context.model.generate(merge_prompt, cfg=context.generation_config, max_tokens=MERGE_BUDGET, **merge_overrides)["output"]

    perturb_prompt = PERTURB_PROMPT.format(
        conversation_history="\n".join([turn.format(include_candidates=False) for turn in prev_turns]),
        user_message=current_turn.user_message,
        candidates=candidates,
        global_axes_summary=axes,
        K=K
    )
    PerturbSchema = create_model(
        "PerturbSchema",
        category=(str, Field(..., description="the category of current topic")),
        new_hypotheses=(conlist(PerturbedHypothesisSchema, min_length=K, max_length=K), Field(..., description=f"List of {K} perturbed hypotheses"))
    )
    budget = UNIT_PERTURB_BUDGET * K
    try:
        perturb_overrides = context.get_generation_overrides("perturb_override")
        output = context.model.generate(perturb_prompt, schema=PerturbSchema, cfg=context.generation_config, max_tokens=budget, **perturb_overrides)["output"]
    except Exception:
        logger.exception("Perturbation failed")
        return [h.id for h in hypotheses], weights.tolist(), None
    current_category = output['category']
    proposed_hypotheses = output['new_hypotheses']
    new_axes = ", ".join(ph['novel_axis'] for ph in proposed_hypotheses)

    for h in hypotheses:
        context.hypothesis_set.remove_hypothesis(h.id)
    new_items = [{"category": category, "content": merged_hypothesis, "prior": merged_prior}]
    new_items.extend([{"category": current_category, "content": ph["content"]} for ph in proposed_hypotheses])
    new_hids = context.hypothesis_set.add_hypotheses(new_items)
    new_weights = [total_weight / len(group)] * len(group)
    return new_hids, new_weights, new_axes
    
