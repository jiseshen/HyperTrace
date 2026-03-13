from .utils import TracerContext


SUMMARY_PROMPT = """
You are summarizing the current belief about a user's preferences.

Goal:
Produce a concise and faithful report describing the user's likely preferences
based on the weighted hypotheses.

Guidelines:
- Reflect the relative likelihood of hypotheses in how much space you allocate.
- Higher-likelihood hypotheses should be described more prominently.
- If hypotheses conflict, mention the dominant explanation first.

Content requirements:
- Summarize the user's likely preferences, values, style, and expectations.
- Preserve the meaning of hypotheses; do NOT invent new preferences.

Output format:
- A short structured summary (4-8 bullet points).
- Each bullet should describe one aspect of the user's preference or tendency.

[Hypotheses]
{hypotheses}
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
    hypotheses, weights = context.belief[:]
    prompt = SUMMARY_PROMPT.format(
        hypotheses="\n".join([f"{h.content} (weight: {w:.2f})" for h, w in zip(hypotheses, weights)]),
    )
    output = context.model.generate(prompt, cfg=context.generation_config, max_tokens=SUMMARY_BUDGET)["output"]
    return output


def summarize_profile(context: TracerContext) -> str:
    top_hypotheses = context.hypothesis_set.top_p_retrieve(p=context.tracer_config.profile_top_p)
    prompt = PROFILE_PROMPT.format(
        hypotheses="\n".join([h.format() for h in top_hypotheses])
    )
    output = context.model.generate(prompt, cfg=context.generation_config, max_tokens=SUMMARY_BUDGET)["output"]
    return output