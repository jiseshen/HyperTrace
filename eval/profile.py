from model import BaseLM, GenerationConfig
from typing import Dict
from pydantic import BaseModel, Field
from .utils import text_similarity, EmbedConfig
import logging

COMPARISON_PROMPT = """
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
2) Core-signal focus: Many surveys are sparse. Scoring should emphasize whether the inferred profile captures the CLEAR, CENTRAL traits that are explicitly present, even if there are only 1–2 such traits.
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


PROFILE_EVAL_BUDGET = 384
logger = logging.getLogger(__name__)

class ProfileEvalSchema(BaseModel):
    aspects_covered: list[str] = Field(..., description="List of key aspects that are covered in the profile")
    survey_consistency: float = Field(ge=0, le=5, description="0-5 score for how well the profile captures the user's claimed preferences and expectations in the survey")
    key_aspect_match: float = Field(ge=0, le=5, description="0-5 score for how well the profile covers the user's prioritized aspects")
    internal_plausibility: float = Field(ge=0, le=5, description="0-5 score for the consistency and plausibility of the profile given user demographics")
    justification: str = Field(..., description="a brief explanation")

def profile_score(eval_model: BaseLM, profile: str, survey: str, embed_cfg: EmbedConfig, evaluation_cfg: GenerationConfig = None) -> Dict[str, float]:
    similarity = text_similarity(profile, survey, embed_cfg)
    prompt = COMPARISON_PROMPT.format(profile=profile, survey=survey)
    try:
        response = eval_model.generate(prompt, schema=ProfileEvalSchema, cfg=evaluation_cfg, max_tokens=PROFILE_EVAL_BUDGET)["output"]
    except Exception as e:
        logger.exception("Profile evaluation failed")
        return {"survey_consistency": None, "key_aspect_match": None, "internal_plausibility": None, "overall": None, "similarity": similarity, "error": str(e)}
    overall_score = 0.4 * response["survey_consistency"] + 0.4 * response["key_aspect_match"] + 0.2 * response["internal_plausibility"]
    return {
        "survey_consistency": response["survey_consistency"],
        "key_aspect_match": response["key_aspect_match"],
        "internal_plausibility": response["internal_plausibility"],
        "overall": overall_score,
        "similarity": similarity
    }
    
