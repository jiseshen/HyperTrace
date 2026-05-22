from typing import Any, Dict, List, Tuple

from data import Turn

from .utils import TracerContext
from prompt.base import PROFILE_PROMPT, SUMMARY_PROMPT


SUMMARY_BUDGET = 256
MIN_FILTERED_RETRIEVALS = 3

def summarize_hypotheses(context: TracerContext) -> str:
    hypotheses, weights = context.belief[:]
    if context.tracer_config.summary_profile_source == "belief_with_retrieved_long_term":
        long_term = retrieve_long_term_summary_items(context)
        consolidated_text = _format_summary_long_term_items(long_term)
    else:
        top_consolidated_hypotheses = context.hypothesis_set.top_p_retrieve(
            p=context.tracer_config.profile_top_p,
            max_k=context.tracer_config.n_hypotheses,
        )
        consolidated_text = "\n\n".join([
            f"[G{i+1}] {h.content} (prior rank: {i+1})"
            for i, h in enumerate(top_consolidated_hypotheses)
        ])
    prompt = context.prompts.summary.format(
        consolidated_hypotheses=consolidated_text,
        current_hypotheses="\n\n".join([f"[H{i+1}] {h.content} (weight: {w:.2f})" for i, (h, w) in enumerate(zip(hypotheses, weights))]),
    )
    summary_overrides = context.get_generation_overrides("summary_override")
    output = context.model.generate(prompt, cfg=context.generation_config, max_tokens=SUMMARY_BUDGET, **summary_overrides)["output"]
    return output


def _format_summary_long_term_items(items: List[Dict[str, Any]]) -> str:
    formatted = []
    for i, item in enumerate(items):
        parts = [f"[L{i+1}] {item['content']}"]
        if "category" in item:
            parts.append(f"category: {item['category']}")
        parts.extend([
            f"prior: {item['prior']:.3f}",
            f"retrieval similarity: {item['similarity']:.3f}",
        ])
        formatted.append(parts[0] + " (" + "; ".join(parts[1:]) + ")")
    return "\n\n".join(formatted)


def retrieve_long_term_summary_items(context: TracerContext) -> List[Dict[str, Any]]:
    if context.current_belief is None or not context.hypothesis_set.hypotheses:
        return []
    current_hypotheses = context.belief.get_hypotheses()
    query = "\n\n".join(h.content for h in current_hypotheses)
    retrieved_items = retrieve_hypothesis_items(
        query,
        context,
        top_k=max(context.tracer_config.summary_retrieve_pool_k, context.tracer_config.summary_long_term_top_k),
    )
    current_ids = set(context.belief.ids)
    selected: List[Dict[str, Any]] = []
    for item in retrieved_items:
        if item["id"] in current_ids:
            continue
        if item["prior"] < context.tracer_config.summary_min_prior:
            continue
        overlaps_current = False
        for current_id in current_ids:
            try:
                similarity = context.hypothesis_set.get_similarity(item["id"], current_id)
            except Exception:
                similarity = None
            if similarity is not None and similarity >= context.tracer_config.similarity_threshold:
                overlaps_current = True
                break
        if overlaps_current:
            continue
        selected.append(item)
        if len(selected) >= context.tracer_config.summary_long_term_top_k:
            break
    return selected


def summarize_retrieved_hypotheses(query: str, context: TracerContext) -> str:
    retrieved_hypotheses, _ = context.hypothesis_set.retrieve_hypotheses(
        query,
        top_k=context.tracer_config.n_hypotheses,
    )
    if not retrieved_hypotheses:
        return ""
    prompt = context.prompts.profile.format(
        hypotheses="\n\n".join([
            h.format(include_category=context.tracer_config.use_hypothesis_topics)
            for h in retrieved_hypotheses
        ])
    )
    summary_overrides = context.get_generation_overrides("summary_override")
    output = context.model.generate(prompt, cfg=context.generation_config, max_tokens=SUMMARY_BUDGET, **summary_overrides)["output"]
    return output


def build_inference_retrieval_query(
    conversation_history: List[Turn],
    max_history_turns: int,
) -> str:
    recent_turns = conversation_history[-max_history_turns:]
    previous_user_messages = [turn.user_message for turn in recent_turns[:-1]]
    current_message = conversation_history[-1].user_message
    parts = []
    if previous_user_messages:
        parts.append(
            "[Recent user messages]\n"
            + "\n".join(f"- {message}" for message in previous_user_messages)
        )
    parts.append(f"[Current user message]\n{current_message}")
    return "\n\n".join(parts)


def _normalize_content(content: str) -> str:
    return " ".join(content.lower().split())


def _retrieved_item(
    hyp,
    prior: float,
    similarity: float,
    rank: int,
    include_category: bool = True,
) -> Dict[str, Any]:
    item = {
        "id": hyp.id,
        "content": hyp.content,
        "prior": float(prior),
        "similarity": float(similarity),
        "semantic_rank": rank,
    }
    if include_category:
        item["category"] = hyp.category
    return item


def _deduplicate_retrieved_items(items: List[Dict[str, Any]], context: TracerContext) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen_content = set()
    for item in items:
        normalized = _normalize_content(item["content"])
        if normalized in seen_content:
            continue
        is_near_duplicate = False
        for existing in selected:
            try:
                similarity = context.hypothesis_set.get_similarity(item["id"], existing["id"])
            except Exception:
                similarity = None
            if similarity is not None and similarity >= context.tracer_config.similarity_threshold:
                is_near_duplicate = True
                break
        if is_near_duplicate:
            continue
        seen_content.add(normalized)
        selected.append(item)
    return selected


def _format_retrieved_hypotheses(items: List[Dict[str, Any]]) -> str:
    formatted = []
    for item in items:
        parts = [f"ID: {item['id']}"]
        if "category" in item:
            parts.append(f"Category: {item['category']}")
        parts.extend([
            f"Prior: {item['prior']:.3f}",
            f"Retrieval similarity: {item['similarity']:.3f}",
            f"Content: {item['content']}",
        ])
        formatted.append("\n".join(parts))
    return "\n\n".join(formatted)


def retrieve_hypothesis_items(
    query: str,
    context: TracerContext,
    top_k: int,
) -> List[Dict[str, Any]]:
    hypotheses, priors, scores = context.hypothesis_set.retrieve_hypotheses_with_scores(
        query,
        top_k=top_k,
    )
    retrieved_items = [
        _retrieved_item(
            hyp,
            prior,
            score,
            rank,
            include_category=context.tracer_config.use_hypothesis_topics,
        )
        for rank, (hyp, prior, score) in enumerate(zip(hypotheses, priors, scores), start=1)
    ]
    retrieved_items.sort(key=lambda item: (-item["similarity"], -item["prior"], item["semantic_rank"]))
    return retrieved_items


def _belief_retrieval_status(
    query: str,
    retrieved_items: List[Dict[str, Any]],
    context: TracerContext,
) -> Dict[str, Any]:
    selected_items = retrieved_items[:context.tracer_config.belief_retrieve_top_k]
    return {
        "success": bool(selected_items),
        "source": "shared_inference_retrieval",
        "query": query,
        "retrieved": selected_items,
        "pool_count": len(retrieved_items),
    }


def retrieve_inference_profile(
    conversation_history: List[Turn],
    context: TracerContext,
    fallback_profile: str = "",
    include_belief_retrieval: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    query = build_inference_retrieval_query(
        conversation_history,
        max_history_turns=context.tracer_config.max_history_turns,
    )
    if not context.hypothesis_set.hypotheses:
        diagnostics = {
            "source": "working_fallback",
            "reason": "empty_hypothesis_set",
            "query": query,
            "retrieved": [],
        }
        if include_belief_retrieval:
            diagnostics["belief_retrieval"] = {
                "success": False,
                "source": "shared_inference_retrieval",
                "query": query,
                "retrieved": [],
                "pool_count": 0,
            }
        return fallback_profile, diagnostics

    pool_k = max(
        context.tracer_config.inference_retrieve_pool_k,
        context.tracer_config.inference_retrieve_top_k,
        context.tracer_config.belief_retrieve_pool_k if include_belief_retrieval else 0,
        context.tracer_config.belief_retrieve_top_k if include_belief_retrieval else 0,
    )
    retrieved_items = retrieve_hypothesis_items(query, context, top_k=pool_k)
    belief_retrieval = (
        _belief_retrieval_status(query, retrieved_items, context)
        if include_belief_retrieval
        else None
    )
    deduped_items = _deduplicate_retrieved_items(retrieved_items, context)
    high_prior_items = [
        item
        for item in deduped_items
        if item["prior"] >= context.tracer_config.inference_min_prior
    ]
    used_prior_filter = len(high_prior_items) >= MIN_FILTERED_RETRIEVALS
    candidate_items = high_prior_items if used_prior_filter else deduped_items
    selected_items = candidate_items[:context.tracer_config.inference_retrieve_top_k]
    if not selected_items:
        diagnostics = {
            "source": "working_fallback",
            "reason": "no_retrieved_hypotheses",
            "query": query,
            "retrieved": [],
            "pool_count": len(retrieved_items),
            "deduped_count": len(deduped_items),
            "used_prior_filter": used_prior_filter,
        }
        if belief_retrieval is not None:
            diagnostics["belief_retrieval"] = belief_retrieval
        return fallback_profile, diagnostics

    prompt = context.prompts.profile.format(
        hypotheses=_format_retrieved_hypotheses(selected_items)
    )
    summary_overrides = context.get_generation_overrides("summary_override")
    output = context.model.generate(
        prompt,
        cfg=context.generation_config,
        max_tokens=SUMMARY_BUDGET,
        **summary_overrides,
    )["output"]
    diagnostics = {
        "source": "retrieved",
        "query": query,
        "retrieved": selected_items,
        "pool_count": len(retrieved_items),
        "deduped_count": len(deduped_items),
        "used_prior_filter": used_prior_filter,
        "fallback_profile_used": False,
    }
    if belief_retrieval is not None:
        diagnostics["belief_retrieval"] = belief_retrieval
    return output, diagnostics


def summarize_profile(context: TracerContext) -> str:
    if context.tracer_config.hypothesis_update_mode == "flat5" and context.current_belief is not None:
        top_hypotheses = context.belief.get_hypotheses()
    else:
        top_hypotheses = context.hypothesis_set.top_p_retrieve(p=context.tracer_config.profile_top_p)
    prompt = context.prompts.profile.format(
        hypotheses="\n\n".join([
            h.format(include_category=context.tracer_config.use_hypothesis_topics)
            for h in top_hypotheses
        ])
    )
    profile_overrides = context.get_generation_overrides("profile_override")
    output = context.model.generate(prompt, cfg=context.generation_config, max_tokens=SUMMARY_BUDGET, **profile_overrides)["output"]
    return output
