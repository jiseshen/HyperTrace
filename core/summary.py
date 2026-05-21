from .utils import TracerContext
from prompt.base import PROFILE_PROMPT, SUMMARY_PROMPT


SUMMARY_BUDGET = 256

def summarize_hypotheses(context: TracerContext) -> str:
    top_consolidated_hypotheses = context.hypothesis_set.top_p_retrieve(p=context.tracer_config.profile_top_p, max_k=context.tracer_config.n_hypotheses)
    hypotheses, weights = context.belief[:]
    prompt = context.prompts.summary.format(
        consolidated_hypotheses="\n\n".join([f"[G{i+1}] {h.content} (prior rank: {i+1})" for i, h in enumerate(top_consolidated_hypotheses)]),
        current_hypotheses="\n\n".join([f"[H{i+1}] {h.content} (weight: {w:.2f})" for i, (h, w) in enumerate(zip(hypotheses, weights))]),
    )
    summary_overrides = context.get_generation_overrides("summary_override")
    output = context.model.generate(prompt, cfg=context.generation_config, max_tokens=SUMMARY_BUDGET, **summary_overrides)["output"]
    return output


def summarize_retrieved_hypotheses(query: str, context: TracerContext) -> str:
    retrieved_hypotheses, _ = context.hypothesis_set.retrieve_hypotheses(
        query,
        top_k=context.tracer_config.n_hypotheses,
    )
    if not retrieved_hypotheses:
        return ""
    prompt = context.prompts.profile.format(
        hypotheses="\n\n".join([h.format() for h in retrieved_hypotheses])
    )
    summary_overrides = context.get_generation_overrides("summary_override")
    output = context.model.generate(prompt, cfg=context.generation_config, max_tokens=SUMMARY_BUDGET, **summary_overrides)["output"]
    return output


def summarize_profile(context: TracerContext) -> str:
    top_hypotheses = context.hypothesis_set.top_p_retrieve(p=context.tracer_config.profile_top_p)
    prompt = context.prompts.profile.format(
        hypotheses="\n\n".join([h.format() for h in top_hypotheses])
    )
    profile_overrides = context.get_generation_overrides("profile_override")
    output = context.model.generate(prompt, cfg=context.generation_config, max_tokens=SUMMARY_BUDGET, **profile_overrides)["output"]
    return output
