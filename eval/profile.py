from model import BaseLM, GenerationConfig
from typing import Dict
from pydantic import BaseModel, Field
from .utils import text_similarity, EmbedConfig
import logging
from prompt import PromptSet, prism_prompts
from prompt.base import PROFILE_EVALUATION_PROMPT
from prompt.personamem_adapter import PERSONAMEM_PROFILE_EVALUATION_PROMPT

COMPARISON_PROMPT = PROFILE_EVALUATION_PROMPT
PERSONAMEM_COMPARISON_PROMPT = PERSONAMEM_PROFILE_EVALUATION_PROMPT


PROFILE_EVAL_BUDGET = 384
logger = logging.getLogger(__name__)

class ProfileEvalSchema(BaseModel):
    aspects_covered: list[str] = Field(..., description="List of key aspects that are covered in the profile")
    survey_consistency: float = Field(ge=0, le=5, description="0-5 score for how well the profile captures the user's claimed preferences and expectations in the survey")
    key_aspect_match: float = Field(ge=0, le=5, description="0-5 score for how well the profile covers the user's prioritized aspects")
    internal_plausibility: float = Field(ge=0, le=5, description="0-5 score for the consistency and plausibility of the profile given user demographics")
    justification: str = Field(..., description="a brief explanation")

def profile_score(
    eval_model: BaseLM,
    profile: str,
    survey: str,
    embed_cfg: EmbedConfig,
    evaluation_cfg: GenerationConfig = None,
    prompts: PromptSet = None,
) -> Dict[str, float]:
    similarity = text_similarity(profile, survey, embed_cfg)
    prompt_set = prompts or prism_prompts()
    prompt = prompt_set.profile_evaluation.format(profile=profile, survey=survey)
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
    
