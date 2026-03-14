from .utils import TracerContext


SUMMARY_PROMPT = """
You are summarizing the current belief about a user's preferences using two sources:

1) Global consolidated hypotheses (long-term): stable tendencies selected by top prior. 
   - These DO NOT have per-hypothesis weights for the current turn.
   - Treat them as background priors: usually stable but not necessarily active right now.

2) Current-conversation hypotheses (short-term): hypotheses with posterior weights for this conversation.
   - Treat these as primary evidence for what to do NOW.

Goal:
Produce a concise, faithful summary that combines long-term tendencies with current-turn evidence.

How to weight the two sources:
- Use current-conversation hypotheses as the main signal; reflect their relative weights in prominence and wording.
- Use global hypotheses as secondary signal:
  * include a global item if it is consistent with current evidence, OR
  * include it if current hypothesis is general/weak/ambiguous, OR
  * include it as a "stable baseline" that the current turn may temporarily override.
- If global and current conflict, do NOT invent a resolution. State the dominant current explanation first (if current weight is concentrated), then mention the long-term baseline as a possible stable tendency.

Output format:
- 4-8 bullet points total, no other text.

Strength mapping (must follow):
- High-weight current items: MUST / STRONGLY / MAINLY
- Medium-weight current items: SHOULD / GENERALLY
- Low-weight but kept current items: MAY / SLIGHTLY
- Global-only items: TENDS TO / OFTEN

Rules:
- Preserve the meaning of hypotheses; do NOT invent new preferences.
- Avoid duplicates: merge overlapping points into one bullet.
- Prefer actionable tendencies (style/structure/expectations) over vague traits.

[Global consolidated hypotheses] (long-term, prior-top; no current-turn weights)
{consolidated_hypotheses}

[Current hypotheses with weights] (for current conversation; weights sum to 1)
{current_hypotheses}
"""


PROFILE_PROMPT = """
You are compiling user preference profile from interaction evidence.

Given:
- A list of hypotheses about the user's latent preferences/values, each with an associated topic category.

Task:
Summarize the hypotheses into a concise profile that captures the user's core values and expectations for the AI assistant.

Guidelines:
- Analyze the hypotheses and focus on "what the user values", "what the user expects from the assistant", and "what aspects the user cares most about in the interaction".
- The potential aspects include values, creativity, fluency, factuality, diversity, safety, personalisation and helpfulness.
- If the hypotheses about values/preference reflect a strong likelihood of the user being a certain age (young, grown, senior), culture, religion, you MUST speculate and mention them as additional snippets
- Output the raw profile text.

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
    output = context.model.generate(prompt, cfg=context.generation_config, max_tokens=SUMMARY_BUDGET)["output"]
    return output


def summarize_profile(context: TracerContext) -> str:
    top_hypotheses = context.hypothesis_set.top_p_retrieve(p=context.tracer_config.profile_top_p)
    prompt = PROFILE_PROMPT.format(
        hypotheses="\n\n".join([h.format() for h in top_hypotheses])
    )
    output = context.model.generate(prompt, cfg=context.generation_config, max_tokens=SUMMARY_BUDGET)["output"]
    return output