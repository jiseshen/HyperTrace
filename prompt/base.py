from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSet:
    skip: str
    preprocessing: str
    initialization: str
    likelihood: str
    branching: str
    axis: str
    merge: str
    perturb: str
    consolidate: str
    summary: str
    profile: str
    response: str
    prediction: str
    response_evaluation: str
    profile_evaluation: str


@dataclass(frozen=True)
class PromptInserts:
    skip: str = ""
    preprocessing: str = ""
    initialization: str = ""
    likelihood: str = ""
    branching: str = ""
    summary: str = ""
    profile: str = ""
    response: str = ""
    prediction: str = ""
    response_evaluation: str = ""
    profile_evaluation: str = ""


def _insert_after(prompt: str, anchor: str, insert: str) -> str:
    insert = insert.strip()
    if not insert:
        return prompt
    if anchor not in prompt:
        raise ValueError(f"Prompt anchor not found: {anchor}")
    return prompt.replace(anchor, f"{anchor}\n{insert}", 1)


SKIP_PROMPT = """
Role:
You are the gating module for an LLM personalization system.

Goal:
Decide whether this turn contains usable preference evidence.

Set "skip": true if either condition is true:
- The user message is only greeting/ack/filler and has no meaningful preference, value, stance, boundary, or constraint signal.
- The candidate differences are not preference-relevant and are mainly correctness/completeness/minor wording differences.
- The turn is an objective, user-agnostic task where the candidates differ only in factual correctness, calculation, code correctness, or generic completeness.

Important:
- Do not skip only because the topic is sensitive or controversial.
- Focus on preference signal, not toxicity level.
- Do not skip realistic user tasks when the message or candidate contrast may reveal implicit personalization evidence.

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

PREPROCESSING_PROMPT = """
Role:
You are the preprocessing module for an LLM personalization system.

Goal:
Convert raw candidates into stable, compact summaries that preserve preference-relevant differences.

Step 1:
Identify 1-4 high-contrast dimensions across candidates.
Use short canonical labels (examples: values, information_density, structure, actionability, tone, framing, abstraction).
Only keep dimensions with detectable contrast.

Step 2:
For each candidate, produce one concise summary (<50 words).
- Summary must cover the listed dimensions for that candidate.
- You may add at most one extra salient detail.
- Avoid restating the whole candidate.

Output (JSON only):
{{
  "reason": "a short justification about the dimension identification",
  "dimensions": ["dimension 1", ...],
  "summarized_candidates": [
    {{"i": 0, "summary": "a brief summary for candidate 0"}},
    {{"i": 1, "summary": "a brief summary for candidate 1"}}
  ]
}}

Rules:
- Return valid JSON only (no markdown, no comments).
- Use only provided candidates.
- Preserve candidate indices exactly.
- Return exactly one summarized candidate per input candidate.

[user_message]
{user_message}

[candidates]
{candidates}

Generate exactly {n} items in summarized_candidates.
"""

INITIALIZATION_PROMPT = """
Role:
You initialize user preference hypotheses for a personalization pipeline.

Goal:
Infer stable, evidence-supported hypotheses from the latest chosen-vs-rejected comparison.

Procedure:
1. Identify one topic category for organization/retrieval.
2. Produce exactly {n_hypotheses} hypotheses.
- Each hypothesis must cover a different explanatory aspect.
- Reuse relevant retrieved hypotheses when justified; otherwise create new ones.
- Use current user message as auxiliary evidence when it clearly expresses preference signal as explicit feedback/critic.

Output (JSON only):

{{
  "category": "...",
  "hypotheses": [
    {{
      "id": "...",
      "action": "reuse" | "new",
      "content": "...",
      "justification": "..."
    }}
  ]
}}

Rules:
- Return valid JSON only.
- Include all required fields.
- Produce exactly {n_hypotheses} hypotheses.
- Ground each hypothesis in explicit observed evidence.

[Conversation History]
{prev_turns}

[Current User Message]
{user_message}

[Candidate Responses]
{candidates}

Candidate Responses are provided as JSON list items with:
- i: candidate index
- summary: compact candidate summary
- content: compact candidate content
- choice: chosen | rejected

[Previously Retrieved Hypotheses]
{retrieved_hypotheses}
"""

LIKELIHOOD_PROMPT = """
Role:
You are scoring candidate-hypothesis alignment for choice likelihood estimation.

Given:
- Hypothesis z (assume z is true)
- Multiple candidates for the same user message

Goal:
Assign each candidate i an alignment score s_i in [0, 5] for how well it matches z.
Score relative alignment under z, not general response quality.

Score anchors:
- 5: best match to z with clear margin
- 4: strong match, top or tied-top
- 3: moderate match, plausible but not top
- 2: weak match, conflicts on an important dimension
- 1: poor match, largely mismatched
- 0: opposes or ignores z

Rules:
- Candidates are intentionally unlabeled; do not assume which one was chosen.
- Compare candidates relatively under z.
- If z does not explain candidate differences, keep scores near-equal.
- Use wider score spread only when evidence supports it.
- Do not invent preferences outside z and the user message.
- Keep the original candidate order: scores[i] must map to candidate i.

Output (JSON only):
{{
  "scores": [s0, s1, ...]
}}

[Conversation history]
{prev_turns}

[Current User Message]
{user_message}

[Candidate Responses]
{candidates}

Candidate Responses are provided as JSON list items with:
- i: candidate index
- summary: compact candidate summary
- content: compact candidate content
- There is no choice label in this input.

[Hypothesis z]
{hypothesis}
"""

BRANCHING_PROMPT = """
Role:
You update one working hypothesis in a personalization pipeline.

Goal:
Decide whether the latest interaction provides usable evidence for this hypothesis, and output either a revision or replacement.

Step 1 (relevance):
- "direct": interaction clearly supports/contradicts this hypothesis.
- "partial": same underlying preference axis appears with partial overlap.
- "none": no meaningful evidence for this hypothesis.

Step 2 (action):
- If relevance is "direct" or "partial": action="revise".
  Keep the same axis/scope, stay specific, and only incorporate relevant evidence.
- If relevance is "none": action="replace".
  Write a new hypothesis with similar specificity.

Category rules:
- Keep original category unless current evidence clearly indicates shift.
- If old category has multiple labels, keep the most appropriate one.
- Do not add a new category without clear evidence.
- Category is organizational only; hypothesis content should reflect preference evidence.

Output (JSON only):

{{
  "action": "revise" | "replace",
  "relevance": "direct" | "partial" | "none",
  "updated_hypothesis": {{
    "category": "string",
    "content": "string"
  }},
  "justification": "one brief (1-2 sentences) justification of the update decision"
}}

Rules:
- If action="revise", category usually remains unchanged.
- If action="replace", do not mention the old hypothesis content.
- Ground updates strictly in observed evidence.
- Avoid speculation and over-generalization.

[Conversation History]
{prev_turns}

[Current User Message]
{user_message}

[Candidate Responses]
{candidates}

Candidate Responses are provided as JSON list items with:
- i: candidate index
- summary: compact candidate summary
- content: compact candidate content
- choice: chosen | rejected

[Current Hypothesis]
{current_hypothesis}
"""

AXIS_PROMPT = """
Role:
You extract explanatory axes for a list of user preference hypotheses.

Goal:
For each hypothesis, output the key latent axis it represents.

Output:
Raw text only, as a comma-separated list:
"axis for hypothesis 1", "axis for hypothesis 2", ...

Rules:
- Use short noun phrases only.
- No explanations, no extra text.
- Hypotheses may share the same axis.

[Hypotheses]
{hypotheses}
"""

MERGE_PROMPT = """
Role:
You merge a cluster of highly similar user preference hypotheses.

Goal:
Produce one canonical hypothesis.

Rules:
- Preserve stable components shared by the cluster.
- Remove stylistic rephrasing and redundancy.
- Keep it specific and evidence-grounded.
- Do not invent new preferences.

Output:
Raw merged hypothesis text only. No explanation.

[CollapsedCluster]
{collapsed_cluster}
"""

PERTURB_PROMPT = """
Role:
You generate rejuvenation hypotheses for a Sequential Monte Carlo personalization system.

Goal:
Generate K plausible new hypotheses that introduce new explanatory axes.

Requirements:
1. Identify one topic category for the current conversation.
2. Generate exactly K new hypotheses.
- Each hypothesis must reflect a distinct latent explanation (not surface rephrasing).
- Each hypothesis must introduce an axis not already in GlobalDiversitySummary.
- Each hypothesis must remain plausible under the conversation evidence.

Output (JSON only):
{{
  "category": "a concise topic label for the current conversation",
  "new_hypotheses": [
    {{"content": "...", "novel_axis": "...", "justification": "..."}}
  ]
}}

[ConversationHistory]
{conversation_history}

[CurrentUserMessage]
{user_message}

[CandidateResponses]
{candidates}

[GlobalDiversitySummary]
{global_axes_summary}

K={K}
"""

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

SUMMARY_PROMPT = """
Role:
You summarize current user preference beliefs from long-term and short-term hypothesis sources.

Goal:
Write a concise, faithful summary for what the assistant should do now.

Evidence weighting:
- Current-conversation hypotheses and long-term hypotheses are complementary evidence sources.
- Current hypotheses capture the latest posterior for this conversation.
- Long-term hypotheses capture stable cross-conversation signals from memory.
- Do not assume either source is always more important; preserve the stronger or more stable claim when evidence supports it.
- If current and long-term hypotheses conflict, state the distinction instead of forcing one into the other.

Output:
- 4-8 bullet points only. No extra text.

Strength mapping:
- High-weight current: MUST / STRONGLY / MAINLY
- Medium-weight current: SHOULD / GENERALLY
- Low-weight current: MAY / SLIGHTLY
- Long-term only: TENDS TO / OFTEN

Rules:
- Preserve hypothesis meaning; do not invent new preferences.
- Merge overlaps and remove duplicates.
- Prefer actionable tendencies over vague traits.

[Long-term hypotheses] (prior-ranked; may include retrieved stable memory not in current belief)
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
- Only include claims supported by evidence in hypotheses.
- Do not speculate about demographics or sensitive attributes without explicit evidence.
- Output plain profile text only.

[Hypotheses]
{hypotheses}
"""

RESPONSE_PROMPT = """
Role:
You are an assistant that adapts responses to the user's preferences and values.

Goal:
Generate a response that follows user profile constraints when relevant, while obeying the current request.

Step 1:
Produce adaptation_plan as a list of actionable constraints.
- Include only constraints clearly supported by the profile and relevant to the current message.
- Keep each item concrete (for example: "Use bullet points", "Avoid jargon").
- If profile is empty/irrelevant, return an empty list.

Step 2:
Generate the final response.
- Apply all constraints in adaptation_plan.
- Directly answer the current message.
- Do not mention profile, adaptation_plan, or personalization process.
- Follow the length constraint strictly.

Conflict policy:
- Explicit current request overrides profile.
- For partial conflict, follow the explicit request and keep non-conflicting profile constraints.
- If the user asks for detail under tight length limits, prioritize structure and essential coverage over verbosity.

Output (JSON only, no markdown fences):
{{
  "adaptation_plan": [
    "<actionable constraint 1>",
    "<actionable constraint 2>"
  ],
  "response": "<your response to the current message>"
}}

[Response length]
Keep the response length at {l} to {r} words.

[User preference profile]
{profile}

[Conversation history]
{prev_turns}

[Current user message]
{current_message}
"""

PREDICTION_PROMPT = """
You are ranking candidate responses for a user based on a given user preference profile.

Given:
- User preference profile: a concise summary of the user's stable preferences and constraints
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

RESPONSE_EVALUATION_PROMPT = """
You are evaluating the similarity between an Adapted response and a set of candidate responses in the context of the user preferences.

Inputs:
- Current turn with c candidates
- Adapted response

Step 1:
Identify 2-4 preference-relevant dimensions that distinguish the candidates.
Use short canonical labels (e.g., "values", "information_density", "structure", "actionability", "tone", "framing", "abstraction") (Not exhaustive and do not force-fit).

Step 2:
Using those dimensions, score how similar the Adapted response is to EACH candidate.
Scores are 0-5:
5 = Near-identical on the key dimensions; Adapted matches candidate's stance/style/structure with no meaningful drift.
4 = Strong match on most key dimensions; minor drift on at most one dimension.
3 = Partial match; aligns on some key dimensions but differs on others OR ambiguity prevents a clear judgment.
2 = Weak match; differs on one or more key dimensions in a way that matters.
1 = Very weak match; mostly reflects the opposite of the candidate on key dimensions.
0 = Opposes/contradicts the candidate on the key dimensions (clear mismatch).

Output valid, parsable JSON only without any extra commentary:
{{
  "dimensions": ["...", "..."],
  "scores": [s0, s1, ..., s{{c-1}}],
  "justification": "a brief (1-2 sentences) explanation of the key similarities/differences between the Adapted response and each candidates."
}}

Rules:
- scores length MUST equal number of candidates; scores[i] corresponds to candidate i. Do NOT change the candidate order.
- Use the same dimensions for scoring all candidates.

[Current turn]
{current_turn}

[Number of candidates]
c={c}

[Adapted]
{adapted}
"""

PROFILE_EVALUATION_PROMPT = """
You are evaluating how well an inferred user preference profile aligns with ground truth.

Inputs:
- Ground truth records (possibly partial)
- Inferred profile: a user preference profile inferred from conversation history.

Key principles:
1) Partial GT: Do NOT treat missing ground-truth fields as negatives. Extra details in the inferred profile are NOT wrong by default.
2) Core-signal focus: Scoring should emphasize whether the inferred profile captures the CLEAR, CENTRAL traits that are explicitly present.
3) Hard contradictions: Penalize only when the inferred profile clearly contradicts explicit ground truth.

Evaluate three aspects (each 0-5):

A) Ground Truth Consistency (explicit signal capture + non-contradiction)
Rubric:
5 = Captures the clearest core preference(s)/expectation(s) and shows no contradictions.
4 = Mostly captures the core signal; minor omissions or slight ambiguity; no hard conflicts.
3 = Partial capture: gets some signal right but misses/blur a key core point OR contains an ambiguous tension.
2 = Weak alignment: misses the core ground-truth signal OR includes one clear contradiction.
1 = Very weak: multiple clear contradictions or systematically mischaracterizes the core signal.
0 = Opposite: directly contradicts the main explicit survey preference(s).

B) Key Aspect Match
Rubric:
5 = Covers most important aspects with correct emphasis.
4 = Covers several important aspects; emphasis mostly right with small drift.
3 = Covers some important aspects but misses key ones or spreads emphasis too broadly.
2 = Mentions few important aspects or focuses on irrelevant/generic traits.
1 = Barely covers important aspects; emphasis largely misaligned.
0 = Fails to reflect important aspects at all.

C) Internal Plausibility
Rubric:
5 = Profile is coherent and contains no unsupported assumptions.
4 = Generally compatible; a small stretch but not clearly incompatible.
3 = Neutral/unknown: profile stays generic/compatible.
2 = Some questionable assumptions, but not outright incompatible.
1 = Clearly unsupported assumptions.
0 = Multiple strong incompatibilities.

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


def compose_prompts(inserts: PromptInserts | None = None) -> PromptSet:
    inserts = inserts or PromptInserts()
    return PromptSet(
        skip=_insert_after(
            SKIP_PROMPT,
            "- Focus on preference signal, not toxicity level.",
            inserts.skip,
        ),
        preprocessing=_insert_after(
            PREPROCESSING_PROMPT,
            "Only keep dimensions with detectable contrast.",
            inserts.preprocessing,
        ),
        initialization=_insert_after(
            INITIALIZATION_PROMPT,
            "- Each hypothesis must cover a different explanatory aspect.",
            inserts.initialization,
        ),
        likelihood=_insert_after(
            LIKELIHOOD_PROMPT,
            "- Keep the original candidate order: scores[i] must map to candidate i.",
            inserts.likelihood,
        ),
        branching=_insert_after(
            BRANCHING_PROMPT,
            "- Avoid speculation and over-generalization.",
            inserts.branching,
        ),
        axis=AXIS_PROMPT,
        merge=MERGE_PROMPT,
        perturb=PERTURB_PROMPT,
        consolidate=CONSOLIDATE_PROMPT,
        summary=_insert_after(
            SUMMARY_PROMPT,
            "- Prefer actionable tendencies over vague traits.",
            inserts.summary,
        ),
        profile=_insert_after(
            PROFILE_PROMPT,
            "Guidelines:",
            inserts.profile,
        ),
        response=_insert_after(
            RESPONSE_PROMPT,
            "- If the user asks for detail under tight length limits, prioritize structure and essential coverage over verbosity.",
            inserts.response,
        ),
        prediction=_insert_after(
            PREDICTION_PROMPT,
            "- Do not invent preference signals not present in the profile.",
            inserts.prediction,
        ),
        response_evaluation=_insert_after(
            RESPONSE_EVALUATION_PROMPT,
            "- Use the same dimensions for scoring all candidates.",
            inserts.response_evaluation,
        ),
        profile_evaluation=_insert_after(
            PROFILE_EVALUATION_PROMPT,
            "3) Hard contradictions: Penalize only when the inferred profile clearly contradicts explicit ground truth.",
            inserts.profile_evaluation,
        ),
    )


def base_prompts() -> PromptSet:
    return compose_prompts()
