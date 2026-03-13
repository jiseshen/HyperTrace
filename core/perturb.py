from typing import Any, Dict, List, Literal, Tuple
import logging
from pydantic import BaseModel, create_model, conlist
from core.utils import TracerContext
from core.hypothesis_set import Hypothesis, WorkingBelief
from data.base import Turn

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
You are performing particle rejuvenation in a Sequential Monte Carlo (SMC) preference tracing system.

Your goal is to produce exactly K plausible hypotheses about the user's latent preference that are
consistent with the current interaction while maintaining diversity in the hypothesis set.

Given:
- Conversation history (optional)
- Current user message and candidate responses (with the chosen response indicated)
- A merged hypothesis representing the current local explanation
- A set of retrieved hypotheses from the global hypothesis repository (each with an id)
- The number K of hypotheses to produce

Important:
Retrieved hypotheses are only references. Some may be irrelevant to the current interaction.

Task:

Step 1 — Identify the topic/category of the current conversation.
This category is used only for organizing and retrieving hypotheses.

Step 2 — (Internally) Evaluate retrieved hypotheses.
For each retrieved hypothesis:
- Determine whether it is applicable to the current interaction.

If applicable:
- It may be revised to better explain the chosen response.

If not applicable:
- Ignore it and do not use it as the basis of a new hypothesis.
- It is possible that none of the retrieved hypotheses are relevant to the current interaction.
- In that case, generate hypotheses by reinterpreting the merged hypothesis instead.
- Do NOT force a retrieved hypothesis to fit if it is clearly unrelated.

Step 3 — Generate rejuvenated hypotheses.

Use the following strategies:

Strategy A — Retrieval-based rejuvenation
- Start from an applicable retrieved hypothesis.
- Revise it so it better explains the current chosen response in the content.
- Preserve its core idea or explanatory dimension.
- The output hypothesis MUST keep the same id as the retrieved hypothesis. The corresponding action is "reuse".

Strategy B — Local reinterpretation
- Start from the merged hypothesis.
- Rephrase or reinterpret it from a slightly different perspective in the content.
- Assign a new id such as "new-1", "new-2", etc. The action is "new".

Guidelines:
- Prefer Strategy A whenever possible.
- If fewer than K applicable retrieved hypotheses exist, use Strategy B to generate the remaining hypotheses.
- Avoid purely stylistic rewording.
- Do not introduce unsupported preferences.
- Keep the conciseness and concreteness of the hypotheses.

Output JSON only:

{{
  "category": "...",
  "new_hypotheses": [
    {{
      "id": "...",
      "action": "reuse" | "new",
      "content": "...",
      "justification": "..."
    }}
  ]
}}

Requirements:
- Produce exactly K={K} hypotheses.
- Each hypothesis must have id, content, and justification.
- The id must be either a retrieved hypothesis id or a new id ("new-x").

[ConversationHistory]
{conversation_history}

[CurrentUserMessage]
{user_message}

[CandidateResponses]
{candidates}

[MergedHypothesis]
{merged_hypothesis}

[RetrievedHypotheses]
{retrieved_hypotheses}

K={K}
"""

class PerturbedHypothesisSchema(BaseModel):
    id: str
    action: Literal["reuse", "new"]
    content: str

AXIS_BUDGET = 64
MERGE_BUDGET = 256
UNIT_PERTURB_BUDGET = 256
logger = logging.getLogger(__name__)


def perturb_hypotheses(conversation_history: List[Turn], candidates: str, similar_groups: List[List[int]], context: TracerContext) -> Dict[str, Any]:
    perturbed_hids, perturbed_weights = [], []
    for group in similar_groups:
        new_hids, new_weights = perturb_group(group=group, conversation_history=conversation_history, candidates=candidates, context=context)
        perturbed_hids.extend(new_hids)
        perturbed_weights.extend(new_weights)
    perturbed_belief = WorkingBelief(ids=perturbed_hids, priors=perturbed_weights, repo=context.hypothesis_set)
    context.update_belief(perturbed_belief)
    return {"groups": [group for group in similar_groups if len(group) > 1]}
    
def perturb_group(group: List[int], conversation_history: List[Turn], candidates: str, context: TracerContext) -> Tuple[List[str], List[float]]:
    if len(group) == 1:
        hyps, weights = context.belief[group]
        return [hyps[0].id], weights.tolist()
    prev_turns = conversation_history[-context.tracer_config.max_history_turns:]
    current_turn = conversation_history[-1]
    hypotheses, weights = context.belief[group]
    exclude_ids = context.belief.ids
    total_weight = weights.sum()
    merged_weight = total_weight * (1 - context.tracer_config.perturb_alpha)
    merged_prior = max([context.hypothesis_set.global_prior[h.id] for h in hypotheses])
    category = max([h.category for h in hypotheses], key=lambda c: c.count(",") if c else 0)
    K = len(group) - 1
    merge_prompt = MERGE_PROMPT.format(collapsed_cluster="\n\n".join([h.content for h in hypotheses]))
    merged_hypothesis = hypotheses[0].content if len(set(h.id for h in hypotheses)) == 1 else context.model.generate(merge_prompt, cfg=context.generation_config, max_tokens=MERGE_BUDGET)["output"]
    retrieved_hypotheses, _ = context.hypothesis_set.retrieve_hypotheses(merged_hypothesis, top_k=K, exclude_ids=exclude_ids)
    perturb_prompt = PERTURB_PROMPT.format(
        conversation_history="\n".join([turn.format(include_candidates=False) for turn in prev_turns]),
        user_message=current_turn.user_message,
        candidates=candidates,
        merged_hypothesis=merged_hypothesis,
        retrieved_hypotheses="\n".join([h.format() for h in retrieved_hypotheses]),
        K=K
    )
    PerturbSchema = create_model(
        "PerturbSchema",
        category=(str, ...),
        new_hypotheses=(conlist(PerturbedHypothesisSchema, min_length=K, max_length=K), ...)
    )
    budget = UNIT_PERTURB_BUDGET * K
    try:
        output = context.model.generate(perturb_prompt, schema=PerturbSchema, cfg=context.generation_config, max_tokens=budget)["output"]
    except Exception:
        logger.exception("Perturbation failed")
        return [h.id for h in hypotheses], weights.tolist()
    current_category = output['category']
    proposed_hypotheses = output['new_hypotheses']

    for h in hypotheses:
        context.hypothesis_set.remove_hypothesis(h.id)
    update_list = []
    add_list = [{"category": category, "content": merged_hypothesis, "prior": merged_prior}]
    reused_priors = []
    new_hids = []
    for ph in proposed_hypotheses:
        if ph['action'] == 'reuse' and ph['id'] in context.hypothesis_set.hypotheses:
            prev_category = context.hypothesis_set[ph['id']][0].category
            new_hids.append(ph['id'])
            new_category = current_category
            if new_category not in prev_category:
                new_category = prev_category + ", " + new_category
            update_list.append(Hypothesis(id=ph['id'], category=new_category, content=ph['content']))
            reused_priors.append(context.hypothesis_set[ph['id']][1])
        else:
            add_list.append({"category": current_category, "content": ph['content']})
    new_hids.extend(context.hypothesis_set.add_hypotheses(add_list))
    context.hypothesis_set.update_hypotheses(update_list)
    if len(reused_priors) == 0:
        merged_weight = total_weight
    new_weights = [(total_weight - merged_weight) * p / sum(reused_priors) for p in reused_priors]
    new_weights.extend([merged_weight / len(add_list)] * len(add_list))
    return new_hids, new_weights
    
