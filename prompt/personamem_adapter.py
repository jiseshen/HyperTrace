from dataclasses import replace

from .base import PromptInserts, PromptSet, compose_prompts


PERSONAMEM_TRACE_INSERTS = PromptInserts(
    skip="""PersonaMem-v2 focus:
- Treat subtle implicit persona evidence as usable when candidate differences depend on user-specific context, needs, constraints, preference updates, ownership, or privacy boundaries.
- Do not skip realistic daily chatbot tasks used by PersonaMem-v2, including writing/email refinement, translation, social/chat messages, troubleshooting, health/therapy consultation, knowledge queries, travel/event/fashion/lifestyle questions, or multimodal-style questions, when they may contain implicit user information.
- Skip only when the message is filler/administrative or clearly user-agnostic irrelevant content such as public math/coding benchmark material where the candidate contrast is just correctness and no user-specific information is present.
- Do not skip only because the signal is not a wording/style preference.""",
    preprocessing="""PersonaMem-v2 dimension guidance:
- Prefer dimensions that explain user alignment, not just surface wording.
- Useful labels include user_specific_need, stable_preference, preference_update, ownership, privacy_boundary, task_context, communication_style, and safety_constraint.
- Mark chosen/rejected differences that reveal what information about the user should or should not guide future answers.
- In candidate summaries, preserve visible user-background cues the response uses or avoids: role/work/study context, hobbies/interests, pets/family, health/body constraints, schedule, location/culture/language, food/style preferences, values, and recent changes.
- Prefer phrasing such as "uses background: ...; adapts by ...; misses/avoids ..."; do not collapse these cues into only "personalized", "helpful", "concise", or "detailed".""",
    initialization="""PersonaMem-v2 task guidance:
- Prioritize user-specific preferences, needs, constraints, communication style, values, and safe personalization cues over generic answer quality.
- The chosen vs rejected candidate labels are online tracing evidence and must be used to infer what better aligns with this user.
- Treat do-not-remember and sensitive/private signals as boundaries or safe abstractions, not facts to retain verbatim.
- Hypotheses should be PersonaMem memory units, not surface response-style guesses. Prefer content that starts with one of: [background_fact], [domain_preference], [constraint_boundary], [preference_update], [ownership_boundary], or [adaptation_rule].
- Extract background from the user message and the chosen-vs-rejected contrast: the chosen response may reveal which background cue is useful, while rejected responses may reveal wrong assumptions, irrelevant details, or unsafe personalization.
- Generate different memory slots rather than several paraphrases of concision/practicality. Only make a style hypothesis when wording, tone, format, or level of detail is the main contrast.
- Include how the memory should guide future behavior when possible, e.g. "use when planning meals/travel/emails" or "avoid applying unless the current request is related".""",
    likelihood="""PersonaMem-v2 scoring guidance:
- Score relative personalization alignment under z, not generic response quality.
- Reward candidates that satisfy the user's inferred needs, constraints, stable or updated preferences, and safe memory boundaries.
- Do not reward demographic stereotypes, forbidden memory retention, ownership mistakes, or exact sensitive/private details.
- If z is a background fact or adaptation rule, ask whether the candidate uses it correctly for this current request. Penalize candidates that ignore a clearly relevant memory cue or apply an irrelevant cue just to sound personalized.""",
    branching="""PersonaMem-v2 task guidance:
- Compare chosen vs rejected candidates for personalization signal, not generic answer quality alone.
- Update toward current evidence when it indicates a preference change.
- Preserve ownership: preferences about another person are not the user's own preferences.
- Treat do-not-remember and sensitive/private signals as boundaries or safe abstractions, not facts to retain verbatim.
- Do not use or infer hidden PersonaMem-v2 benchmark metadata or benchmark truth profiles.
- PersonaMem-v2 intentionally mixes many realistic task scenarios for the same user. A topic change by itself is not evidence that the old memory is false.
- For compatible old and new evidence, prefer action="revise" and preserve the old memory while adding or abstracting the new cue as a separate typed slot in the content. Use action="replace" mainly for direct contradiction, a current preference update, an unsafe/private-memory correction, wrong ownership, or a hypothesis that was merely local to one task and has no durable value.
- Keep memory slots explicit with labels such as [background_fact], [domain_preference], [constraint_boundary], [preference_update], [ownership_boundary], and [adaptation_rule]. Do not merge unrelated cues into a vague trait like "practical" or "low effort" unless that abstraction is the actual chosen-vs-rejected signal.
- When a current candidate demonstrates how known background should affect an answer, revise toward an adaptation rule: "when the user asks about X, use/avoid Y because ...".
- Only make a generic communication-style update when style, tone, format, or verbosity is the main candidate contrast.""",
    summary="""PersonaMem-v2 guidance:
- Summaries should capture actionable user-alignment information: stable preferences, current updated preferences, user needs, constraints, communication style, and safety/privacy boundaries.
- Preserve ownership and do-not-remember boundaries; do not convert them into ordinary positive preferences.
- Organize durable evidence as background facts, domain preferences, constraints/boundaries, preference updates, and adaptation rules. Include when each cue should be used if that is clear.""",
    profile="""PersonaMem-v2 guidance:
- Focus on stable preferences, communication style, values, constraints, updated preferences, safety boundaries, and helpfulness expectations.
- Include only information supported by online chosen/rejected evidence.
- Do not infer preferences from demographics or identity alone.
- Do not attribute preferences about other people to the user.
- Do not retain exact sensitive/private details; use safe abstractions only when relevant.
- Respect do-not-remember boundaries as boundaries, not as positive remembered facts.
- Prefer a profile structure that separates: background facts, domain preferences, constraints/boundaries, preference updates, and adaptation rules. Keep each item actionable for future prediction or response generation.""",
    response="""PersonaMem-v2 guidance:
- Respect do-not-remember boundaries in the profile.
- Do not reveal exact sensitive/private details; use only safe abstractions when relevant.
- Apply user-specific needs and constraints only when relevant to the current message.
- In adaptation_plan, first select the relevant profile cues for the current message, then state how to apply them. If no cue is relevant, keep the answer generic and high quality rather than forcing personalization.""",
    prediction="""PersonaMem-v2 guidance:
- Rank by likely user alignment with the inferred profile, including stable or updated preferences, user-specific constraints, ownership, and safety boundaries.
- Do not reward stereotype-based personalization or forbidden private-detail use.
- First identify which profile cues are relevant to this current turn. Prefer candidates that correctly use those background facts, domain preferences, constraints, or update/boundary rules; penalize candidates that ignore a clearly relevant cue or use irrelevant/private cues.""",
    response_evaluation="""PersonaMem-v2 guidance:
- Similarity dimensions may include user-specific need, stable_or_updated_preference, ownership, privacy_boundary, task_context, communication_style, and safety_constraint.
- Score similarity on personalization behavior, not generic fluency alone.
- Include background_use as a dimension when candidates differ in whether they correctly apply known user background to the current task.""",
)


PERSONAMEM_PROFILE_EVALUATION_PROMPT = """
You are evaluating how well an inferred user preference profile aligns with PersonaMem-v2 ground truth.

Inputs:
- Ground truth profile:
  - Persona context and communication style
  - Self preferences from benchmark rows
  - Updated preferences where the current preference supersedes an old one
  - Memory boundaries / do-not-remember constraints
  - Sensitive/private constraints that should be handled as safe abstractions only
  - Facts marked as not the user's own preferences
- Inferred profile: a profile inferred from online chosen-vs-rejected interactions.

Key principles:
1) Personalization focus: Reward profiles that capture stable user preferences, communication style, values, and relevant background needed for personalized answers.
2) No stereotype inference: Demographics and identity context are compatibility information only. Do not reward unsupported stereotypes.
3) Updates: Reward the current preference when an update is present. Penalize profiles that keep only the superseded preference or present both as current.
4) Memory boundaries: If ground truth says a do-not-remember boundary exists, penalize profiles that retain the forbidden fact as a user preference.
5) Sensitive/private details: Reward safe category-level abstractions. Penalize exact private-detail retention or disclosure.
6) Ownership: Penalize attributing "not the user's own preferences" to the user.

Evaluate three aspects (each 0-5):

A) Survey Consistency
5 = Captures the clearest self preferences and communication style with no contradictions or unsafe retention.
4 = Mostly captures core signals; minor omissions; no serious contradiction.
3 = Partial capture; gets some signals right but misses important preferences or update/boundary nuance.
2 = Weak alignment; misses most core signals or includes one serious contradiction.
1 = Very weak; multiple contradictions, unsafe retention, or repeated wrong ownership.
0 = Opposite of the central preferences or clearly unsafe with memory/sensitive constraints.

B) Key Aspect Match
5 = Covers most benchmark preference categories with correct emphasis, including updates, boundaries, and sensitive constraints.
4 = Covers several important categories with small gaps.
3 = Covers some categories but misses important ones or overgeneralizes.
2 = Covers few categories or focuses heavily on irrelevant/generic traits.
1 = Barely reflects benchmark preferences.
0 = Fails to reflect benchmark preferences.

C) Internal Plausibility
5 = Profile is coherent, compatible with persona context, and avoids unsupported stereotypes.
4 = Generally coherent with minor overreach.
3 = Neutral/partially supported; mostly generic but not contradictory.
2 = Some questionable assumptions or ownership mistakes.
1 = Clear unsupported assumptions or unsafe details.
0 = Multiple strong incompatibilities or unsafe claims.

Output ONLY the final JSON:
{{
  "aspects_covered": ["aspect1", "aspect2", ...],
  "survey_consistency": 0-5,
  "key_aspect_match": 0-5,
  "internal_plausibility": 0-5,
  "justification": "2-4 sentences explaining the most important matches/mismatches.",
}}

Now evaluate:

[PersonaMem-v2 Ground Truth]
{survey}

[Inferred Profile]
{profile}
"""


def personamem_prompts() -> PromptSet:
    return replace(
        compose_prompts(PERSONAMEM_TRACE_INSERTS),
        profile_evaluation=PERSONAMEM_PROFILE_EVALUATION_PROMPT,
    )
