from dataclasses import replace

from .base import PromptInserts, PromptSet, compose_prompts


PRISM_TRACE_INSERTS = PromptInserts(
    initialization="- Prioritize conversational style/value preferences over topic facts.",
    profile="- Focus on values, style, structure, factuality expectations, safety boundaries, and helpfulness preferences.",
    profile_evaluation="""PRISM survey guidance:
- Ground truth records may include basic demographics, self descriptions, a system string, prioritized aspects, and less-prioritized aspects.
- Demographics are NOT used to infer preferences; use them only as a weak compatibility check for obvious incompatibilities.
- If religion is not provided or is "prefer not to say", do NOT use religion-based reasoning at all.
- Topic-specific interests should be ignored unless they encode stable preference signals.""",
)


PRISM_PROFILE_EVALUATION_PROMPT = """
You are evaluating how well an inferred user preference profile aligns with a user's ground-truth survey.

Inputs:
- Survey records (ground truth, possibly partial):
  - Basic demographics (age, gender; religion/ethnicity may be "prefer not to say")
  - Self descriptions (values/explicit preferences)
  - System string (expectations for AI interaction)
  - Prioritized aspects and less-prioritized aspects (if provided)
- Inferred profile: a user preference profile inferred from conversation history.

Key principles:
1) Partial GT: Do NOT treat missing survey fields as negatives. Extra details in the inferred profile are NOT wrong by default.
2) Core-signal focus: Many surveys are sparse. Scoring should emphasize whether the inferred profile captures the CLEAR, CENTRAL traits that are explicitly present, even if there are only 1-2 such traits.
3) Hard contradictions: Penalize only when the inferred profile clearly contradicts explicit survey information.
4) Demographics as compatibility check (very weak):
   - Demographics are NOT used to "infer" preferences.
   - Use them only to detect obvious incompatibilities or surprising assumptions when survey is silent.
   - If religion is not provided or is "prefer not to say", do NOT use religion-based reasoning at all.
5) Topic-agnostic: Ignore topic-specific interests unless they encode stable preference signals.

Evaluate three aspects (each 0-5):

A) Survey Consistency (explicit signal capture + non-contradiction)
Rubric:
5 = Captures the survey's clearest core preference(s)/expectation(s) and shows no contradictions (even if survey is brief).
4 = Mostly captures the core signal; minor omissions or slight ambiguity; no hard conflicts.
3 = Partial capture: gets some signal right but misses/blur a key core point OR contains an ambiguous tension.
2 = Weak alignment: misses the core survey signal OR includes one clear contradiction to an explicit survey statement.
1 = Very weak: multiple clear contradictions or systematically mischaracterizes the core signal.
0 = Opposite: directly contradicts the main explicit survey preference(s).

B) Key Aspect Match (prioritized aspects, if provided)
Rubric:
5 = Covers most prioritized aspects with correct emphasis; avoids overemphasizing less-prioritized aspects.
4 = Covers several prioritized aspects; emphasis mostly right with small drift.
3 = Covers some prioritized aspects but misses key ones or spreads emphasis too broadly.
2 = Mentions few prioritized aspects; noticeably focuses on less-prioritized aspects.
1 = Barely covers prioritized aspects; emphasis largely misaligned.
0 = Fails to reflect prioritized aspects at all.
If the survey does NOT provide prioritized aspects, set key_aspect_match=3 by default unless there is strong evidence to go higher/lower.

C) Internal Plausibility (demographic compatibility check)
Interpretation:
- This is NOT stereotype-based profiling and NOT used to infer preferences.
- It is only a sanity check for obvious incompatibilities with provided demographics.

Rubric:
5 = No demographic incompatibilities; profile makes compatible claims.
4 = Generally compatible; a small stretch but not clearly incompatible.
3 = Neutral/unknown: demographics provide little usable constraint OR the profile stays generic/compatible.
2 = Some questionable assumptions given demographics (unnecessary leaps), but not outright incompatible.
1 = Clearly incompatible assumptions with provided demographics.
0 = Multiple strong incompatibilities.

Constraints for C:
- Use religion-based compatibility checks ONLY if religion is explicitly provided and not "prefer not to say".
- If demographics are missing/withheld, score C mainly by internal coherence and default toward 3 unless clearly problematic.

Output ONLY the final JSON:
{{
  "aspects_covered": ["aspect1", "aspect2", ...],
  "survey_consistency": 0-5,
  "key_aspect_match": 0-5,
  "internal_plausibility": 0-5,
  "justification": "2-4 sentences explaining the most important core-signal matches/mismatches.",
}}

Now evaluate:

[Survey]
{survey}

[Inferred Profile]
{profile}
"""


PRISM_SKIP_PROMPT = """
Role:
You are the gating module for an LLM personalization system.

Goal:
Decide whether this turn contains usable preference evidence.

Set "skip": true if either condition is true:
- The user message is only greeting/ack/filler and has no meaningful preference, value, stance, boundary, or constraint signal.
- The candidate differences are not preference-relevant and are mainly correctness/completeness/minor wording differences.

Important:
- Do not skip only because the topic is sensitive or controversial.
- Focus on preference signal, not toxicity level.

Output (JSON only):
{{
  "reason": "a brief (1-2 sentences) justification about why to skip or not",
  "skip": boolean
}}

[user_message]
{user_message}

[candidates]
{candidates}
"""


PRISM_PREDICTION_PROMPT = """
You are ranking candidate responses for a user based on a given user preference profile.

Given:
- User preference profile: a concise summary of the user's stable preferences, values, and communication style
- Conversation history (optional)
- Current user message
- Candidate responses. Each candidate has a unique ID in square brackets, such as [C1], [C2].

Internally:
- Compare each candidate response over how likely the user is to prefer each one.
- Evaluate alignment based on explicit signals in the profile.
- If the profile is empty or clearly irrelevant to this turn, rank candidates based on overall quality, clarity, and usefulness.

Then:
Rank candidates from best to worst according to alignment to the user preference.

Output Format:

Return a JSON object:
{{
  "ranking": ["id1", "id2"],
  "justification": "a brief (2-3 sentences) explanation of the ranking"
}}

Rules:
- Output valid, parsable JSON only, without any extra commentary.
- When referring to candidates in your output, use only their IDs (e.g., C1, C2).
- Keep the number in the ranking exactly the same as the number of given candidates.
- Do not invent preference signals not present in the profile.

User preference profile:
{profile}

Conversation history:
{prev_turns}

Current interaction:
{current_turn}

Number of candidates: {c}
"""


def prism_prompts() -> PromptSet:
    return replace(
        compose_prompts(PRISM_TRACE_INSERTS),
        skip=PRISM_SKIP_PROMPT,
        prediction=PRISM_PREDICTION_PROMPT,
        profile_evaluation=PRISM_PROFILE_EVALUATION_PROMPT,
    )
