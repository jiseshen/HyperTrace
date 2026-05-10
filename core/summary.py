from .utils import TracerContext


SUMMARY_PROMPT = """
Role:
You summarize current user preference beliefs from long-term and short-term hypothesis sources.

Goal:
Write a concise, faithful summary for what the assistant should do now.

Evidence weighting:
- Current-conversation hypotheses are primary evidence.
- Global consolidated hypotheses are secondary background priors.
- If current and global conflict, present the dominant current explanation first, then note global baseline as longer-term tendency.

Output:
- 4-8 bullet points only. No extra text.

Strength mapping (required):
- High-weight current: MUST / STRONGLY / MAINLY
- Medium-weight current: SHOULD / GENERALLY
- Low-weight current: MAY / SLIGHTLY
- Global-only: TENDS TO / OFTEN

Rules:
- Preserve hypothesis meaning; do not invent new preferences.
- Merge overlaps and remove duplicates.
- Prefer actionable tendencies over vague traits.

[Global consolidated hypotheses] (long-term, prior-top; no current-turn weights)
{consolidated_hypotheses}

[Current hypotheses with weights] (for current conversation; weights sum to 1)
{current_hypotheses}
"""


PROFILE_PROMPT = """
Role:
You compile a user preference profile from hypothesis evidence.

Goal:
Produce a concise profile of what the user values and expects from the assistant.

Guidelines:
- Focus on values, style, structure, factuality expectations, safety boundaries, and helpfulness preferences.
- Only include claims supported by evidence in hypotheses.
- Do not speculate about demographics or sensitive attributes without explicit evidence.
- Output plain profile text only.

[Hypotheses]
{hypotheses}
"""

SUMMARY_BUDGET = 256

def summarize_hypotheses(context: TracerContext) -> str:
    top_consolidated_hypotheses = context.hypothesis_set.top_p_retrieve(p=context.tracer_config.profile_top_p, max_k=context.tracer_config.n_hypotheses)
    hypotheses, weights = context.belief[:]
    prompt = SUMMARY_PROMPT.format(
        consolidated_hypotheses="\n\n".join([f"[G{i+1}] {h.content} (prior rank: {i+1})" for i, h in enumerate(top_consolidated_hypotheses)]),
        current_hypotheses="\n\n".join([f"[H{i+1}] {h.content} (weight: {w:.2f})" for i, (h, w) in enumerate(zip(hypotheses, weights))]),
    )
    summary_overrides = context.get_generation_overrides("summary_override")
    output = context.model.generate(prompt, cfg=context.generation_config, max_tokens=SUMMARY_BUDGET, **summary_overrides)["output"]
    return output


def summarize_profile(context: TracerContext) -> str:
    top_hypotheses = context.hypothesis_set.top_p_retrieve(p=context.tracer_config.profile_top_p)
    prompt = PROFILE_PROMPT.format(
        hypotheses="\n\n".join([h.format() for h in top_hypotheses])
    )
    profile_overrides = context.get_generation_overrides("profile_override")
    output = context.model.generate(prompt, cfg=context.generation_config, max_tokens=SUMMARY_BUDGET, **profile_overrides)["output"]
    return output
