from typing import Any, Dict, List

import numpy as np

from data.base import Turn
from .utils import TracerContext


CONSOLIDATE_PROMPT = """
Role:
You consolidate a cluster of similar user preference hypotheses.

Goal:
Merge the cluster into one generalized but specific hypothesis.

Rules:
- Preserve stable components strongly supported by the cluster.
- Remove redundant wording and stylistic rephrasing.
- Keep the result evidence-grounded and close in length to source hypotheses.
- Do not invent new preferences.

Output:
Raw merged hypothesis text only. No explanation.

[Cluster]
{collapsed_cluster}
"""

CONSOLIDATE_BUDGET = 256


def compute_importance(conversation_length: int, entropy: float) -> float:
    g = 1 - np.exp(-conversation_length)
    h = np.sqrt(max(1 - entropy, 0.5))
    return float(g * h)

def deduplicate_group(group: List[str], context: TracerContext):
    hyps, _ = context.hypothesis_set[group]
    category = ", ".join(set(c for h in hyps for c in h.category.split(", ")))
    merge_prompt = CONSOLIDATE_PROMPT.format(collapsed_cluster="\n\n".join([h.content for h in hyps]))
    merge_overrides = context.get_generation_overrides("merge_override")
    merged_hypothesis = context.model.generate(merge_prompt, cfg=context.generation_config, max_tokens=CONSOLIDATE_BUDGET, **merge_overrides)["output"]
    context.hypothesis_set.merge_hypotheses(group, {"category": category, "content": merged_hypothesis})

def consolidate_hypotheses(conversation_history: List[Turn], context: TracerContext) -> Dict[str, Any]:
    importance = compute_importance(len(conversation_history), context.current_belief.normalized_entropy())
    context.belief.consolidate(importance, context.tracer_config.consolidate_alpha)
    similar_groups = context.hypothesis_set.get_similarity_groups(context.tracer_config.similarity_threshold)
    for group in similar_groups:
        if len(group) > 1:
            deduplicate_group(group, context)
    return {"groups": [group for group in similar_groups if len(group) > 1], "importance": importance}
