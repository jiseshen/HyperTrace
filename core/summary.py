from typing import List

from core.utils import TracerContext
from data.base import Turn


SUMMARY_PROMPT = """
You are compiling weighted hypotheses about the user's preferences into a short generation guidance for the NEXT assistant response.

Goal:
Produce a stable, actionable instruction that reflects relevant hypotheses and helps generate the next reply.

Guidelines:
- If two kept hypotheses conflict, emphasize the higher-likelihood one.
- Ignore any hypothesis that is not relevant to the current turn.
- Cold-start: If the hypothesis list is empty, produce a general instruction that would be helpful to guide the response to the current user message.

Weight-to-emphasis mapping (must follow):
- Highest likelihood: express as MUST / ALWAYS / STRICTLY.
- Medium likelihood: express as SHOULD / GENERALLY / PREFER.
- Lower but kept: express as MAY / SLIGHTLY / IF POSSIBLE.
- Reflect weights in *space*: allocate more words to higher-likelihood hypotheses (roughly proportional), but keep total output short.

Output requirements:
- Output 3-7 bullet rules total, no other text.
- Bullets must be actionable constraints on how to respond (value, format, structure, density, rigor, tone), not abstract personality labels.
- Do NOT mention likelihood numbers.

[Hypotheses]
{hypotheses}

[Conversation history]
{prev_turns}

[Current user message]
{user_message}
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


def summarize_hypotheses(conversation_history: List[Turn], context: TracerContext) -> str:
    prev_turns = conversation_history[:-1]
    current_turn = conversation_history[-1]
    hypotheses, weights = context.belief[:]
    prompt = SUMMARY_PROMPT.format(
        hypotheses="\n".join([f"{h.content} (weight: {w:.2f})" for h, w in zip(hypotheses, weights)]),
        prev_turns="\n".join([turn.format(include_candidates=False) for turn in prev_turns[-context.tracer_config.max_history_turns:]]),
        user_message=current_turn.user_message
    )
    output = context.model.generate(prompt, cfg=context.generation_config)["output"]
    return output


def summarize_profile(context: TracerContext) -> str:
    top_hypotheses = context.hypothesis_set.top_p_retrieve(p=context.tracer_config.profile_top_p)
    prompt = PROFILE_PROMPT.format(
        hypotheses="\n".join([h.format() for h in top_hypotheses])
    )
    output = context.model.generate(prompt, cfg=context.generation_config)["output"]
    return output