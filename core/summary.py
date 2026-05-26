from typing import Any, Dict, List, Tuple

from pydantic import Field, create_model

from data import Turn

from .utils import TracerContext
from prompt.base import PROFILE_PROMPT, SUMMARY_PROMPT


SUMMARY_BUDGET = 256
PROFILE_BUDGET = 512
INFERENCE_PROFILE_BUDGET = 512
INFERENCE_FILTER_BUDGET = 512
MIN_FILTERED_RETRIEVALS = 3
MAX_INFERENCE_PROFILE_ITEMS = 14
MAX_INFERENCE_FILTER_CANDIDATES = 20

PERSONAMEM_PROFILE_TYPE_ORDER = (
    "ownership_boundary",
    "constraint_boundary",
    "preference_update",
    "background_fact",
    "domain_preference",
    "adaptation_rule",
    "communication_preference",
    "other",
)
PERSONAMEM_PROFILE_TYPE_QUOTAS = {
    "ownership_boundary": 6,
    "constraint_boundary": 6,
    "preference_update": 4,
    "background_fact": 5,
    "domain_preference": 8,
    "adaptation_rule": 5,
    "communication_preference": 3,
    "other": 3,
}
PERSONAMEM_PROFILE_MAX_ITEMS = 18

INFERENCE_RELEVANCE_FILTER_PROMPT = """
Role:
You are selecting user-memory cues for response-time personalization.

Goal:
Choose which retrieved memory cues should actually affect the assistant's answer or
offline preference prediction for the current user message.

Rules:
- Classify useful IDs into these roles:
  - content_ids: cues that may change the substantive answer, recommendation objects,
    examples, constraints, or ranking because they match the current request's domain.
  - style_ids: weak cues that may improve tone, format, pacing, level of detail, framing,
    or usage suggestions, but must not narrow the answer's topic, location, culture, or
    recommendation set.
  - boundary_ids: safety, privacy, ownership, sensitive-info, or do-not-remember boundaries
    that apply to the current message.
- Use retrieval rank as the main ordering signal; this filter is only for removing clearly
  irrelevant or unsafe-to-apply cues, not for enforcing per-role quotas.
- Select at most {content_max} content_ids and {style_max} style_ids unless a boundary cue
  must be kept for safety/privacy/ownership handling.
- Use style_ids when a cue is helpful but not strong enough to decide answer content.
- Do not put broad background traits, location/culture/family ties, or generic preferences
  in content_ids unless the current request explicitly asks for that dimension or the cue
  gives concrete domain guidance.
- Keep boundary_ids whenever they apply now, even if they are not content preferences.
- The bracketed hypothesis type, if present (for example [domain_preference],
  [adaptation_rule], [constraint_boundary]), is not the relevance role. Reclassify each
  cue by how it should affect this current message.
- Decision check:
  - content_ids answer "what should be recommended, selected, avoided, or emphasized?"
  - style_ids answer "how should the same basic answer be phrased, paced, or formatted?"
  - boundary_ids answer "what must not be exposed, remembered, attributed, or assumed?"
- Recent user messages are retrieval context only. Do not select a cue just because it matches
  a previous message.
- If the current request is best answered generically, or no cue is directly applicable,
  return empty arrays.

Output JSON only:
{{
  "content_ids": ["h1"],
  "style_ids": ["h2"],
  "boundary_ids": ["h3"],
  "reason": "brief explanation of the relevance decision"
}}

[Current user message]
{current_message}

[Recent user messages]
{recent_messages}

[Candidate memory cues]
{candidates}
"""

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


def build_current_inference_query(conversation_history: List[Turn]) -> str:
    return f"[Current user message]\n{conversation_history[-1].user_message}"


def build_contextual_inference_query(
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


def _format_recent_user_messages(conversation_history: List[Turn], max_history_turns: int) -> str:
    recent_turns = conversation_history[-max_history_turns:]
    previous_user_messages = [turn.user_message for turn in recent_turns[:-1]]
    if not previous_user_messages:
        return "(none)"
    return "\n".join(f"- {message}" for message in previous_user_messages)


def _normalize_content(content: str) -> str:
    return " ".join(content.lower().split())


def _is_personamem_context(context: TracerContext) -> bool:
    return "PersonaMem-v2 guidance" in context.prompts.profile


def _personamem_memory_type(content: str, metadata: Dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    boundary_types = metadata.get("boundary_types", [])
    labels = metadata.get("labels", [])
    if "do_not_remember" in boundary_types or "forbidden_memory" in labels:
        return "constraint_boundary"
    if "other_person_or_unsupported" in boundary_types or "ownership_boundary" in labels:
        return "ownership_boundary"
    if "privacy_or_sensitive" in boundary_types or "privacy_boundary" in labels:
        return "constraint_boundary"
    text = content.lower()
    for memory_type in PERSONAMEM_PROFILE_TYPE_ORDER:
        if f"[{memory_type}]" in text:
            return memory_type
    if any(marker in text for marker in ("who=others", "belongs to someone else", "not the user's own preference", "do not assume")):
        return "ownership_boundary"
    if any(marker in text for marker in ("sensitive", "private", "privacy", "do-not-remember", "do not remember", "redact", "placeholder")):
        return "constraint_boundary"
    if any(marker in text for marker in ("updated preference", "current preference", "previous preference", "supersedes")):
        return "preference_update"
    if any(marker in text for marker in ("communication", "tone", "format", "style")):
        return "communication_preference"
    return "other"


def _exclude_from_personamem_final_profile(hypothesis) -> bool:
    metadata = getattr(hypothesis, "metadata", {}) or {}
    if metadata.get("profile_exclude"):
        return True
    if "forbidden_memory" in metadata.get("labels", []):
        return True
    if "do_not_remember" in metadata.get("boundary_types", []):
        return True
    if "other_person_or_unsupported" in metadata.get("boundary_types", []):
        return True
    text = hypothesis.content.lower()
    return any(
        marker in text
        for marker in (
            "ask_to_forget",
            "do-not-remember",
            "do not remember",
            "forget this",
            "forgotten boundary",
            "forbidden fact",
            "who=others",
            "belongs to someone else",
            "not the user's own preference",
        )
    )


def _format_personamem_profile_hypothesis(hypothesis, include_category: bool = True) -> str:
    metadata = getattr(hypothesis, "metadata", {}) or {}
    boundary_types = metadata.get("boundary_types", [])
    labels = metadata.get("labels", [])
    is_boundary = (
        metadata.get("profile_exclude")
        or metadata.get("negative_only")
        or boundary_types
        or labels
        or _exclude_from_personamem_final_profile(hypothesis)
    )
    if not is_boundary:
        return hypothesis.format(include_category=include_category)

    parts = [f"ID: {hypothesis.id}"]
    if include_category:
        parts.append(f"Category: {hypothesis.category}")
    handling = []
    if metadata.get("negative_only"):
        handling.append("negative_only")
    if metadata.get("do_not_repeat_content"):
        handling.append("do_not_repeat_content")
    if metadata.get("profile_exclude"):
        handling.append("exclude_from_final_profile")
    handling.extend(labels)
    handling.extend(f"boundary:{item}" for item in boundary_types)
    if handling:
        parts.append(f"Memory handling: {', '.join(dict.fromkeys(handling))}")

    if metadata.get("do_not_repeat_content") or "do_not_remember" in boundary_types:
        content = (
            "[constraint_boundary] Do-not-repeat / deletion boundary for this category. "
            "Use this only to suppress matching positive profile claims; do not write the "
            "forbidden content or topic into the final profile."
        )
    elif "other_person_or_unsupported" in boundary_types:
        content = (
            f"[ownership_boundary] Negative-only boundary: {hypothesis.content} "
            "Use only to prevent unsupported ownership attribution; do not rewrite this "
            "as a user-owned preference."
        )
    else:
        content = (
            f"[constraint_boundary] Negative-only boundary: {hypothesis.content} "
            "Use only to suppress unsafe/private/unsupported profile claims; do not "
            "rewrite this as a positive preference."
        )
    parts.append(f"Content: {content}")
    return "\n".join(parts) + "\n"


def _select_personamem_profile_hypotheses(context: TracerContext):
    ranked = [
        (context.hypothesis_set.hypotheses[hid], float(prior))
        for hid, prior in context.hypothesis_set.global_prior.items()
        if hid in context.hypothesis_set.hypotheses
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    selected = []
    selected_ids = set()
    for memory_type in PERSONAMEM_PROFILE_TYPE_ORDER:
        quota = PERSONAMEM_PROFILE_TYPE_QUOTAS[memory_type]
        count = 0
        for hyp, _ in ranked:
            if hyp.id in selected_ids:
                continue
            if _personamem_memory_type(hyp.content, getattr(hyp, "metadata", {}) or {}) != memory_type:
                continue
            selected.append(hyp)
            selected_ids.add(hyp.id)
            count += 1
            if count >= quota or len(selected) >= PERSONAMEM_PROFILE_MAX_ITEMS:
                break
        if len(selected) >= PERSONAMEM_PROFILE_MAX_ITEMS:
            break
    for hyp, _ in ranked:
        if len(selected) >= PERSONAMEM_PROFILE_MAX_ITEMS:
            break
        if hyp.id not in selected_ids:
            selected.append(hyp)
            selected_ids.add(hyp.id)
    return selected


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
    if getattr(hyp, "metadata", None):
        item["metadata"] = hyp.metadata
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
        if "relevance_roles" in item:
            parts.append(f"Relevance roles: {', '.join(item['relevance_roles'])}")
        if "category" in item:
            parts.append(f"Category: {item['category']}")
        if "sources" in item:
            parts.append(f"Retrieval sources: {', '.join(item['sources'])}")
        metadata = item.get("metadata") or {}
        handling = []
        if metadata.get("negative_only"):
            handling.append("negative_only")
        if metadata.get("do_not_repeat_content"):
            handling.append("do_not_repeat_content")
        if metadata.get("profile_exclude"):
            handling.append("exclude_from_final_profile")
        for label in metadata.get("labels", []):
            if label not in handling:
                handling.append(label)
        for boundary_type in metadata.get("boundary_types", []):
            handling.append(f"boundary:{boundary_type}")
        if handling:
            parts.append(f"Memory handling: {', '.join(handling)}")
        parts.extend([
            f"Prior: {item['prior']:.3f}",
            f"Retrieval similarity: {item['similarity']:.3f}",
            f"Content: {item['content']}",
        ])
        formatted.append("\n".join(parts))
    return "\n\n".join(formatted)


def _format_inference_hypotheses(items: List[Dict[str, Any]], current_message: str) -> str:
    retrieval_text = _format_retrieved_hypotheses(items)
    return (
        f"{retrieval_text}\n\n"
        "[Current inference task]\n"
        f"Current user message: {current_message}\n"
        "Build a response-time profile for this message only.\n"
        "Important: the retrieved list may contain unrelated or weakly related memory cues. "
        "Do not force-fit them into the current answer or ranking. Keep only cues that are "
        "directly relevant now plus safety/privacy/ownership boundaries that apply now.\n"
        "Instruction strength: use MUST only for hard safety/privacy/ownership boundaries or "
        "requirements stated explicitly in the current user message. For ordinary retrieved "
        "preference cues, use softer wording such as 'prefer when directly relevant', "
        "'consider', or 'may'. Do not turn a weak or contextual cue into a hard instruction.\n"
        "Faithfulness check: do not analogize or transfer a cue into a new domain unless the "
        "same actionable attribute is explicitly present in the current request or candidate "
        "contrast. For example, an offline-download media preference is not evidence for food "
        "portability, and an emotional-coping cue is not evidence for outdoor-route preferences "
        "unless the current turn explicitly connects those axes.\n"
        "Preserve each kept cue's type and how it should be handled: [background_fact] and "
        "[domain_preference] may affect content only when topic-matched; [preference_update] "
        "supersedes older preferences; [adaptation_rule] changes response strategy; "
        "[communication_preference] affects only tone/format/pacing; [constraint_boundary] "
        "and [ownership_boundary] are hard negative constraints. If a cue is irrelevant, "
        "omit it from the summary instead of explaining how it might fit. style cues may affect only tone, format, pacing, or framing. "
        "Do not let style/context cues narrow topic, location, "
        "culture, or recommendation set unless the current candidate contrast varies on "
        "that same axis. Never convert rejected-only, who=others, or belongs-to-someone-else "
        "evidence into a self-owned user preference; if it must be kept, keep it only as a "
        "negative ownership boundary. If an item has Memory handling negative_only or "
        "do_not_repeat_content, do not quote or restate its specific content in the response-time "
        "profile; write only a generic negative boundary such as 'avoid unsupported/private "
        "personalization cues' when that boundary is relevant now."
    )


def _prediction_profile_from_items(items: List[Dict[str, Any]], current_message: str) -> str:
    if not items:
        return ""
    return (
        "[Prediction-time profile]\n"
        f"Current user message: {current_message}\n"
        "Use these typed cues when ranking candidate responses. Content and boundary cues "
        "are strongest; style cues are weak tie-breakers; context cues are optional and "
        "should matter only when candidates vary on the same axis. Never treat a context "
        "or style cue as permission to assume ownership.\n\n"
        f"{_format_retrieved_hypotheses(items)}"
    )


def _merge_retrieved_item_pools(
    current_items: List[Dict[str, Any]],
    context_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    def add_source(item: Dict[str, Any], source: str) -> None:
        item_id = item["id"]
        source_similarity_key = f"{source}_similarity"
        if item_id not in merged:
            merged_item = dict(item)
            merged_item["sources"] = [source]
            merged_item[source_similarity_key] = item["similarity"]
            merged[item_id] = merged_item
            return

        merged_item = merged[item_id]
        if source not in merged_item["sources"]:
            merged_item["sources"].append(source)
        merged_item[source_similarity_key] = item["similarity"]
        merged_item["prior"] = max(merged_item["prior"], item["prior"])
        if item["similarity"] > merged_item["similarity"]:
            merged_item["similarity"] = item["similarity"]
            merged_item["semantic_rank"] = min(
                merged_item["semantic_rank"],
                item["semantic_rank"],
            )

    for item in current_items:
        add_source(item, "current")
    for item in context_items:
        add_source(item, "context")

    return sorted(
        merged.values(),
        key=lambda item: (
            "current" not in item["sources"],
            -item.get("current_similarity", -1.0),
            -item["prior"],
            -item.get("context_similarity", -1.0),
            item["semantic_rank"],
        ),
    )


def _format_filter_candidates(items: List[Dict[str, Any]]) -> str:
    formatted = []
    for item in items:
        parts = [f"ID: {item['id']}"]
        if "category" in item:
            parts.append(f"category: {item['category']}")
        if "sources" in item:
            parts.append(f"sources: {', '.join(item['sources'])}")
        parts.extend([
            f"prior: {item['prior']:.3f}",
            f"similarity: {item['similarity']:.3f}",
            f"content: {item['content']}",
        ])
        formatted.append("; ".join(parts))
    return "\n".join(formatted)


def _coerce_model_output(output: Any) -> Dict[str, Any]:
    if isinstance(output, dict):
        return output
    model_dump = getattr(output, "model_dump", None)
    if model_dump:
        return model_dump()
    dict_method = getattr(output, "dict", None)
    if dict_method:
        return dict_method()
    return {}


def _valid_id_list(
    raw_ids: Any,
    valid_ids: set[str],
    max_items: int,
) -> List[str]:
    selected = []
    seen_ids = set()
    if not isinstance(raw_ids, list):
        return selected
    for item_id in raw_ids:
        if item_id in valid_ids and item_id not in seen_ids:
            selected.append(item_id)
            seen_ids.add(item_id)
        if len(selected) >= max_items:
            break
    return selected


def _annotate_relevance_roles(
    candidate_items: List[Dict[str, Any]],
    content_ids: List[str],
    style_ids: List[str],
    boundary_ids: List[str],
    max_items: int,
) -> List[Dict[str, Any]]:
    role_by_id: Dict[str, List[str]] = {}
    for role, ids in (
        ("content", content_ids),
        ("style", style_ids),
        ("boundary", boundary_ids),
    ):
        for item_id in ids:
            role_by_id.setdefault(item_id, [])
            if role not in role_by_id[item_id]:
                role_by_id[item_id].append(role)

    selected = []
    for item in candidate_items:
        roles = role_by_id.get(item["id"])
        if not roles:
            continue
        selected_item = dict(item)
        selected_item["relevance_roles"] = roles
        selected.append(selected_item)
        if len(selected) >= max_items:
            break
    return selected


def model_filter_inference_items(
    items: List[Dict[str, Any]],
    conversation_history: List[Turn],
    context: TracerContext,
    max_items: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not items:
        return [], {
            "success": True,
            "selected_ids": [],
            "content_ids": [],
            "style_ids": [],
            "boundary_ids": [],
            "reason": "no candidate items",
            "candidates": [],
        }

    candidate_items = items[:max(context.tracer_config.inference_filter_pool_k, max_items)]
    candidate_items = candidate_items[:MAX_INFERENCE_FILTER_CANDIDATES]
    valid_ids = {item["id"] for item in candidate_items}
    prompt = INFERENCE_RELEVANCE_FILTER_PROMPT.format(
        content_max=context.tracer_config.inference_content_max_k,
        style_max=context.tracer_config.inference_style_max_k,
        max_items=max_items,
        current_message=conversation_history[-1].user_message,
        recent_messages=_format_recent_user_messages(
            conversation_history,
            context.tracer_config.max_history_turns,
        ),
        candidates=_format_filter_candidates(candidate_items),
    )
    schema = create_model(
        "InferenceRelevanceFilterSchema",
        content_ids=(List[str], Field(..., description="IDs of cues that can change answer content, possibly empty")),
        style_ids=(List[str], Field(..., description="IDs of cues that can only affect tone, format, pacing, or framing, possibly empty")),
        boundary_ids=(List[str], Field(..., description="IDs of applicable safety/privacy/ownership/do-not-remember boundaries, possibly empty")),
        reason=(str, Field(..., description="brief explanation of the relevance decision")),
    )
    filter_overrides = context.get_generation_overrides("filter_override")
    try:
        output = context.model.generate(
            prompt,
            schema=schema,
            cfg=context.generation_config,
            max_tokens=INFERENCE_FILTER_BUDGET,
            **filter_overrides,
        )["output"]
        parsed = _coerce_model_output(output)
    except Exception as exc:
        fallback_items = items[:max_items]
        return fallback_items, {
            "success": False,
            "reason": str(exc),
            "candidate_count": len(candidate_items),
            "selected_ids": [item["id"] for item in fallback_items],
            "content_ids": [item["id"] for item in fallback_items],
            "style_ids": [],
            "boundary_ids": [],
            "candidates": candidate_items,
            "fallback": "heuristic_top_items",
        }

    content_ids = _valid_id_list(
        parsed.get("content_ids", []),
        valid_ids,
        max_items,
    )
    style_ids = _valid_id_list(
        parsed.get("style_ids", []),
        valid_ids,
        max_items,
    )
    boundary_ids = _valid_id_list(parsed.get("boundary_ids", []), valid_ids, max_items)
    selected = _annotate_relevance_roles(
        candidate_items,
        content_ids=content_ids,
        style_ids=style_ids,
        boundary_ids=boundary_ids,
        max_items=max_items,
    )
    selected_ids = [item["id"] for item in selected]
    selected_id_set = set(selected_ids)
    content_ids = [item_id for item_id in content_ids if item_id in selected_id_set]
    style_ids = [item_id for item_id in style_ids if item_id in selected_id_set]
    boundary_ids = [item_id for item_id in boundary_ids if item_id in selected_id_set]
    return selected, {
        "success": True,
        "selected_ids": selected_ids,
        "content_ids": content_ids,
        "style_ids": style_ids,
        "boundary_ids": boundary_ids,
        "reason": parsed.get("reason", ""),
        "candidate_count": len(candidate_items),
        "candidates": candidate_items,
    }


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
    query = build_current_inference_query(conversation_history)
    context_query = None
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
        diagnostics["prediction_profile"] = fallback_profile
        return fallback_profile, diagnostics

    pool_k = max(
        context.tracer_config.inference_retrieve_pool_k,
        context.tracer_config.inference_retrieve_top_k,
        context.tracer_config.belief_retrieve_pool_k if include_belief_retrieval else 0,
        context.tracer_config.belief_retrieve_top_k if include_belief_retrieval else 0,
    )
    current_retrieved_items = retrieve_hypothesis_items(query, context, top_k=pool_k)
    context_retrieved_items = current_retrieved_items
    if context.tracer_config.inference_retrieval_query_mode == "current_and_context":
        context_query = build_contextual_inference_query(
            conversation_history,
            max_history_turns=context.tracer_config.max_history_turns,
        )
        if context_query != query:
            context_retrieved_items = retrieve_hypothesis_items(context_query, context, top_k=pool_k)
        retrieved_items = _merge_retrieved_item_pools(
            current_retrieved_items,
            context_retrieved_items,
        )
    else:
        retrieved_items = current_retrieved_items
    belief_retrieval = (
        _belief_retrieval_status(context_query or query, context_retrieved_items, context)
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
    selected_limit = min(
        context.tracer_config.inference_retrieve_top_k,
        MAX_INFERENCE_PROFILE_ITEMS,
    )
    model_filter = None
    if context.tracer_config.inference_model_filter:
        selected_items, model_filter = model_filter_inference_items(
            candidate_items,
            conversation_history,
            context,
            max_items=selected_limit,
        )
        if _is_personamem_context(context) and len(selected_items) < selected_limit:
            selected_ids = {item["id"] for item in selected_items}
            context_added_ids = []
            for item in candidate_items:
                if len(selected_items) >= selected_limit:
                    break
                if item["id"] in selected_ids:
                    continue
                context_item = dict(item)
                context_item["relevance_roles"] = ["context"]
                selected_items.append(context_item)
                selected_ids.add(item["id"])
                context_added_ids.append(item["id"])
            if model_filter is not None:
                model_filter["context_added_ids"] = context_added_ids
                model_filter["selected_ids"] = [item["id"] for item in selected_items]
    else:
        selected_items = candidate_items[:selected_limit]
    if not selected_items:
        source = "retrieved_no_relevant_hypotheses" if model_filter and model_filter.get("success") else "working_fallback"
        profile = "" if context.tracer_config.inference_empty_profile_on_no_relevance else fallback_profile
        diagnostics = {
            "source": source,
            "reason": "no_retrieved_hypotheses",
            "query": query,
            "context_query": context_query,
            "retrieved": [],
            "pool_count": len(retrieved_items),
            "current_pool_count": len(current_retrieved_items),
            "context_pool_count": len(context_retrieved_items),
            "deduped_count": len(deduped_items),
            "used_prior_filter": used_prior_filter,
            "model_filter": model_filter,
            "prediction_profile": profile,
        }
        if belief_retrieval is not None:
            diagnostics["belief_retrieval"] = belief_retrieval
        return profile, diagnostics

    raw_prediction_profile = _prediction_profile_from_items(
        selected_items,
        current_message=conversation_history[-1].user_message,
    )
    prompt = context.prompts.profile.format(
        hypotheses=_format_inference_hypotheses(
            selected_items,
            current_message=conversation_history[-1].user_message,
        )
    )
    summary_overrides = context.get_generation_overrides("summary_override")
    output = context.model.generate(
        prompt,
        cfg=context.generation_config,
        max_tokens=INFERENCE_PROFILE_BUDGET if _is_personamem_context(context) else SUMMARY_BUDGET,
        **summary_overrides,
    )["output"]
    profile_format = "summarized"
    diagnostics = {
        "source": "retrieved",
        "query": query,
        "context_query": context_query,
        "retrieved": selected_items,
        "pool_count": len(retrieved_items),
        "current_pool_count": len(current_retrieved_items),
        "context_pool_count": len(context_retrieved_items),
        "deduped_count": len(deduped_items),
        "used_prior_filter": used_prior_filter,
        "selection_limit": selected_limit,
        "model_filter": model_filter,
        "prediction_profile": output,
        "raw_prediction_profile": raw_prediction_profile,
        "profile_format": profile_format,
        "fallback_profile_used": False,
    }
    if belief_retrieval is not None:
        diagnostics["belief_retrieval"] = belief_retrieval
    return output, diagnostics


def retrieve_legacy_prediction_profile(
    conversation_history: List[Turn],
    context: TracerContext,
    fallback_profile: str = "",
) -> Tuple[str, Dict[str, Any]]:
    query = build_contextual_inference_query(
        conversation_history,
        max_history_turns=context.tracer_config.max_history_turns,
    )
    if not context.hypothesis_set.hypotheses:
        return fallback_profile, {
            "source": "working_fallback",
            "reason": "empty_hypothesis_set",
            "query": query,
            "retrieved": [],
        }

    pool_k = max(
        context.tracer_config.prediction_retrieve_pool_k,
        context.tracer_config.prediction_retrieve_top_k,
    )
    retrieved_items = retrieve_hypothesis_items(query, context, top_k=pool_k)
    deduped_items = _deduplicate_retrieved_items(retrieved_items, context)
    high_prior_items = [
        item
        for item in deduped_items
        if item["prior"] >= context.tracer_config.prediction_min_prior
    ]
    used_prior_filter = len(high_prior_items) >= MIN_FILTERED_RETRIEVALS
    candidate_items = high_prior_items if used_prior_filter else deduped_items
    selected_items = candidate_items[:context.tracer_config.prediction_retrieve_top_k]
    if not selected_items:
        return fallback_profile, {
            "source": "working_fallback",
            "reason": "no_retrieved_hypotheses",
            "query": query,
            "retrieved": [],
            "pool_count": len(retrieved_items),
            "deduped_count": len(deduped_items),
            "used_prior_filter": used_prior_filter,
        }

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
    return output, {
        "source": "legacy_retrieved",
        "query": query,
        "retrieved": selected_items,
        "pool_count": len(retrieved_items),
        "deduped_count": len(deduped_items),
        "used_prior_filter": used_prior_filter,
        "selection_limit": context.tracer_config.prediction_retrieve_top_k,
        "fallback_profile_used": False,
    }


def summarize_profile(context: TracerContext) -> str:
    if context.tracer_config.hypothesis_update_mode == "flat5" and context.current_belief is not None:
        top_hypotheses = context.belief.get_hypotheses()
    elif _is_personamem_context(context):
        top_hypotheses = _select_personamem_profile_hypotheses(context)
    else:
        top_hypotheses = context.hypothesis_set.top_p_retrieve(p=context.tracer_config.profile_top_p)
    prompt = context.prompts.profile.format(
        hypotheses="\n\n".join([
            h.format(include_category=context.tracer_config.use_hypothesis_topics)
            for h in top_hypotheses
        ])
    )
    profile_overrides = context.get_generation_overrides("profile_override")
    output = context.model.generate(prompt, cfg=context.generation_config, max_tokens=PROFILE_BUDGET, **profile_overrides)["output"]
    return output
