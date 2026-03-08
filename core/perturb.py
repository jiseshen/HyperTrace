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



PERTURB_PROMPT = """
You are performing particle rejuvenation in a Sequential Monte Carlo personalization system.

Context:
A cluster of hypotheses has collapsed due to high similarity. 
Your job is to:
1) Merge them into a single hypothesis.
2) Propose a small number of new, meaningfully distinct hypotheses.
3) Ensure new hypotheses do NOT overlap with existing global hypotheses.

Given:
- Conversation history
- Current user message and candidate responses (with choice indicated)
- The content of the collapsed cluster of hypotheses
- A summary of existing global hypotheses (including the merged one)

Task:

Step A — Canonical Merge:
- Merge the collapsed cluster into ONE canonical hypothesis.
- Preserve stable components that are strongly supported.
- Remove purely stylistic rephrasing.

Step B — Controlled Diversification:
- Generate exactly N-1 new hypotheses, where N is the number of hypotheses in the collapsed cluster.
- Each must differ in underlying latent explanation.
- Each must introduce a NEW explanation axis not already in GlobalDiversitySummary.
- Do NOT restate the canonical hypothesis.
- Each must remain plausible given the conversation.
- Also specify in "novel_axis" and "justification":
    * What dimension it differs on
    * Why it explains the chosen response
- Avoid shallow variation (tone-only or wording-only).

Also identify the category/topic of the current conversation in "category"

Output JSON only without extra commentary:

{
  "category": "a concise topic label for current conversation",
  "merged": "the merged hypothesis",
  "new_hypotheses": [
    {
      "content": "...",
      "novel_axis": "...",
      "justification": "..."
    }
  ]
}

[ConversationHistory]
{conversation_history}

[CurrentUserMessage]
{user_message}

[CandidateResponses]
{candidates}

[CollapsedCluster]
{collapsed_cluster}

[GlobalDiversitySummary]
The current global hypothesis space already includes the following explanation axes:
{global_axes_summary}

[Number of new hypotheses to generate]
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
    for group in similar_groups:
        new_hids, new_weights, new_axes = perturb_group(group=group, axes=axes, conversation_history=conversation_history, candidates=candidates, context=context)
        perturbed_hids.extend(new_hids)
        perturbed_weights.extend(new_weights)
        axes += f", {new_axes}" if new_axes else ""
    perturbed_belief = WorkingBelief(ids=perturbed_hids, priors=perturbed_weights, repo=context.hypothesis_set)
    return perturbed_belief
    
def perturb_group(group: List[int], axes: str, conversation_history: List[Turn], candidates: str, context: TracerContext) -> Tuple[List[str], List[float], Optional[str]]:
    if len(group) == 1:
        hyps, weights = context.belief[group]
        return [hyps[0].id], weights.tolist(), None
    prev_turns = conversation_history[-context.tracer_config.max_history_turns:]
    current_turn = conversation_history[-1]
    hypotheses, weights = context.belief[group]
    total_weight = weights.sum()
    merged_weight = total_weight * (1 - context.tracer_config.perturb_alpha)
    merged_prior = max([context.hypothesis_set.global_prior[h.id] for h in hypotheses])
    category = max([h.category for h in hypotheses], key=lambda c: c.count(",") if c else 0)
    K = len(group) - 1
    prompt = PERTURB_PROMPT.format(
        conversation_history="\n".join([turn.format(include_candidates=False) for turn in prev_turns]),
        user_message=current_turn.user_message,
        candidates=candidates,
        collapsed_cluster="\n\n".join([h.content for h in hypotheses]),
        global_axes_summary=", ".join(axes),
        K=K
    )
    PerturbSchema = create_model(
        "PerturbSchema",
        category=(str, ...),
        merged=(str, ...),
        new_hypotheses=(conlist(PerturbedHypothesisSchema, min_length=K, max_length=K), ...)
    )
    try:
        output = context.model.generate(prompt, schema=PerturbSchema, cfg=context.generation_config)["output"]
    except Exception as e:
        print(f"Perturbation failed with error: {e}")
        return [h.id for h in hypotheses], weights.tolist(), None
    current_category = output['category']
    merged_hypothesis = output['merged']
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
    