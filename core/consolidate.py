from typing import Any, Dict, List

import numpy as np

from data.base import Turn
from .utils import TracerContext, compute_importance
from .hypothesis_set import Hypothesis, WorkingBelief


CONSOLIDATE_PROMPT = """
You are consolidating a set of similar hypotheses about user preferences/values into a more general one.

Given:
- A cluster of similar hypotheses across multiple topics

Task:
Merge the cluster into ONE generalized hypothesis.
- Preserve stable components strongly supported by the cluster.
- Remove stylistic rephrasing and redundant details.
- Keep it specific and evidence-grounded; do not invent new preferences.

Output ONLY the raw text of merged hypothesis without any explanation.

[Cluster]
{collapsed_cluster}
"""


def compute_importance(conversation_length: int, entropy: float) -> float:
    g = 1 - np.exp(-conversation_length)
    h = 1 - entropy
    return g * h

def deduplicate_group(group: List[str], context: TracerContext):
    hyps, _ = context.hypothesis_set[group]
    category = ", ".join(set(c for h in hyps for c in h.category.split(", ")))
    merge_prompt = CONSOLIDATE_PROMPT.format(collapsed_cluster="\n\n".join([h.content for h in hyps]))
    merged_hypothesis = context.model.generate(merge_prompt, cfg=context.generation_config)["output"]
    context.hypothesis_set.merge_hypotheses(group, {"category": category, "content": merged_hypothesis})

def consolidate_hypotheses(conversation_history: List[Turn], context: TracerContext) -> Dict[str, Any]:
    importance = compute_importance(len(conversation_history), context.current_belief.normalized_entropy())
    context.belief.consolidate(importance, context.tracer_config.consolidate_alpha)
    similar_groups = context.hypothesis_set.get_similarity_groups(context.tracer_config.similarity_threshold)
    for group in similar_groups:
        if len(group) > 1:
            deduplicate_group(group, context)
    return {"groups": [group for group in similar_groups if len(group) > 1], "importance": importance}