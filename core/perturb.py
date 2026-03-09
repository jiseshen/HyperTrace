from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, create_model, conlist
from core.utils import TracerContext
from core.hypothesis_set import Hypothesis, WorkingBelief
from data.base import Turn

AXIS_PROMPT = """
You are extracting the explanatory axis along which a hypothesis z explains the user's preferences for each hypothesis.

Given:
- A list of hypotheses about the user's latent preferences/values

Task:
For each hypothesis, identify the key explanatory dimensions, 
e.g., "The user likely values neutrality and objectivity in geopolitical discussions." explains the interaction through the dimension of "objectiveness".

Output raw text only:
"dimensions for hypothesis 1", "dimensions for hypothesis 2", ...

Rules:
- Use brief phrases for each dimension, without extra explanation.
- It's possible that some hypotheses share the same dimension, don't force-distinguish them.

[Hypotheses]
{hypotheses}
"""

MERGE_PROMPT = """
You are merging a collapsed cluster of hypotheses about a user's latent preferences/values.

Given:
- A cluster of highly similar hypotheses

Task:
Merge the cluster into ONE canonical hypothesis.
- Preserve stable components strongly supported by the cluster.
- Remove stylistic rephrasing and redundant details.
- Keep it specific and evidence-grounded; do not invent new preferences.

Output ONLY the raw text of merged hypothesis without any explanation.

[CollapsedCluster]
{collapsed_cluster}
"""

PERTURB_PROMPT = """
You are generating new hypotheses for particle rejuvenation in a Sequential Monte Carlo personalization system.

Given:
- Conversation history
- Current user message and candidate responses (with choice indicated)
- A summary of existing global explanation axes
- The number K of new hypotheses to generate

Task:
1. Identify the category/topic of the current conversation.
2. Generate exactly K new hypotheses that:
- Differ in underlying latent explanation (not tone-only rephrasing).
- Each introduces a NEW explanation axis not already present in GlobalDiversitySummary.
- Each remains plausible given the conversation.
- For each hypothesis, provide:
  * "novel_axis": the new axis name (short phrase)
  * "justification": why it explains the chosen response

Output JSON only:
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
    content: str
    novel_axis: str


def perturb_hypotheses(conversation_history: List[Turn], candidates: str, similar_groups: List[List[int]], context: TracerContext) -> Dict[str, Any]:
    hypotheses = context.belief.get_hypotheses()
    axes_prompt = AXIS_PROMPT.format(hypotheses="\n\n".join([h.content for h in hypotheses]))
    axes = context.model.generate(axes_prompt, cfg=context.generation_config)["output"]
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
    merged_weight = total_weight * (1 - context.tracer_config.perturb_alpha)
    merged_prior = max([context.hypothesis_set.global_prior[h.id] for h in hypotheses])
    category = max([h.category for h in hypotheses], key=lambda c: c.count(",") if c else 0)
    K = len(group) - 1
    merge_prompt = MERGE_PROMPT.format(collapsed_cluster="\n\n".join([h.content for h in hypotheses]))
    merged_hypothesis = hypotheses[0].content if len(set(h.id for h in hypotheses)) == 1 else context.model.generate(merge_prompt, cfg=context.generation_config)["output"]
    # TODO: Micro-rejuvenation
    perturb_prompt = PERTURB_PROMPT.format(
        conversation_history="\n".join([turn.format(include_candidates=False) for turn in prev_turns]),
        user_message=current_turn.user_message,
        candidates=candidates,
        global_axes_summary=", ".join(axes),
        K=K
    )
    PerturbSchema = create_model(
        "PerturbSchema",
        category=(str, ...),
        new_hypotheses=(conlist(PerturbedHypothesisSchema, min_length=K, max_length=K), ...)
    )
    try:
        output = context.model.generate(perturb_prompt, schema=PerturbSchema, cfg=context.generation_config)["output"]
    except Exception as e:
        print(f"Perturbation failed with error: {e}")
        return [h.id for h in hypotheses], weights.tolist(), None
    current_category = output['category']
    proposed_hypotheses = output['new_hypotheses']
    new_axes = ", ".join(ph['novel_axis'] for ph in proposed_hypotheses)

    for h in hypotheses:
        context.hypothesis_set.remove_hypothesis(h.id)
    new_hids = context.hypothesis_set.add_hypotheses([
        {"category": category, "content": merged_hypothesis, "prior": merged_prior}
    ])
    new_weights = [merged_weight]
    proposed_ids = context.hypothesis_set.add_hypotheses([
        {"category": current_category, "content": ph['content']}
        for ph in proposed_hypotheses
    ])
    new_hids.extend(proposed_ids)
    new_weights.extend([(total_weight - merged_weight) / len(proposed_ids)] * len(proposed_ids))
    return new_hids, new_weights, new_axes
    