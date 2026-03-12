from model import BaseLM, GenerationConfig
from typing import Dict
from pydantic import BaseModel, Field
from .utils import text_similarity, EmbedConfig


COMPARISON_PROMPT = """
You are evaluating how well an inferred user preference profile aligns with a user's ground-truth survey.

Inputs:
- Survey records (ground truth, possibly partial):
  - Basic demographics (age, gender; religion/ethnicity may be "prefer not to say")
  - Self descriptions (values/explicit preferences)
  - System string (expectations for AI interaction, direct signal of preferences)
  - Prioritized aspects and less-prioritized aspects
- Inferred profile: a user preference profile inferred from conversation history.

Key principles:
1) Partial GT: Do NOT treat missing survey fields as negatives. Additional details in the inferred profile are NOT wrong
   by default; they may be true but unobserved in the survey.
2) Hard contradictions: Penalize when the inferred profile clearly contradicts explicit survey information.
3) Demographic priors (soft inference):
   - You MAY use demographics (age, gender, and religion ONLY if explicitly provided) as statistical priors to judge
    plausibility of preferences/values when survey is silent.
   - Demographics are a WEAK signal: they should not override explicit survey statements.
   - If religion/ethnicity is "prefer not to say", do NOT infer religion-based values.
4) Topic-agnostic: Ignore topic-specific interests unless they clearly encode stable preference signals (e.g., "prefers unbiased macro analysis").

Evaluate four aspects (each 1 to 10):
A) Survey Consistency
- Agreement with explicit survey facts: user's stated values and expectations.
- Penalize direct conflicts.

B) Key Aspect Match (prioritized aspects)
- List the aspects covered in the inferred profile in "aspects covered".
- Score reflects how well the inferred profile captures user's prioritized aspects.
- Penalize overemphasis on non-prioritized aspects.

C) Internal Plausibility
- Using age, gender and religion, judge whether the inferred preferences are broadly plausible in a statistical sense.
- e.g. a teen is more likely to prioritize creativity; a religious person may align with certain values.
- This is NOT a correctness check; it is a plausibility check. Just ask: "Given the demographics, how plausible are these preferences"
- Do NOT use religion-based inference if religion is not provided or is "prefer not to say".

Scoring rubric:
The score ranges from 1 (poor) to 10 (excellent) for each aspect.

Output ONLY the final JSON:
{{
  "reason": "A brief explanation of scoring, citing concrete matches/mismatches and analyzing internal plausibility.",
  "aspects_covered": ["aspect1", "aspect2", ...],
  "survey_consistency": 1-10,
  "key_aspect_match": 1-10,
  "internal_plausibility": 1-10,
}}

Now evaluate:

[Survey]
{survey}

[Inferred Profile]
{profile}
"""

class ProfileEvalSchema(BaseModel):
    survey_consistency: float = Field(ge=1.0, le=10.0)
    key_aspect_match: float = Field(ge=1.0, le=10.0)
    internal_plausibility: float = Field(ge=1.0, le=10.0)


def profile_score(eval_model: BaseLM, profile: str, survey: str, embed_cfg: EmbedConfig, evaluation_cfg: GenerationConfig = None) -> Dict[str, float]:
    similarity = text_similarity(profile, survey, embed_cfg)
    prompt = COMPARISON_PROMPT.format(profile=profile, survey=survey)

    try:
        response = eval_model.generate(prompt, schema=ProfileEvalSchema, generation_cfg=evaluation_cfg)["output"]
    except Exception as e:
        return {"survey_consistency": None, "key_aspect_match": None, "internal_plausibility": None, "overall": None, "similarity": similarity, "error": str(e)}
    overall_score = 0.4 * response["survey_consistency"] + 0.4 * response["key_aspect_match"] + 0.2 * response["internal_plausibility"]
    return {
        "survey_consistency": response["survey_consistency"],
        "key_aspect_match": response["key_aspect_match"],
        "internal_plausibility": response["internal_plausibility"],
        "overall": overall_score,
        "similarity": similarity
    }
    