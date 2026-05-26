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


class PersonaMemProfileEvalSchema(BaseModel):
    aspects_covered: list[str] = Field(..., description="List of PersonaMem memory aspects covered by the inferred profile")
    preference_coverage: float = Field(ge=0, le=5, description="0-5 score for coverage of evidence-supported self-owned preferences")
    personalization_utility: float = Field(ge=0, le=5, description="0-5 score for usefulness as personalization memory for future in-situ queries")
    update_and_boundary_handling: float = Field(ge=0, le=5, description="0-5 score for updates, do-not-remember, sensitive/private, and ownership boundaries")
    memory_quality: float = Field(ge=0, le=5, description="0-5 score for compact, coherent, human-readable, evidence-grounded memory quality")
    justification: str = Field(..., description="a brief explanation")


def _is_personamem_profile_eval(prompt_set: PromptSet) -> bool:
    return prompt_set.profile_evaluation.strip() == PERSONAMEM_PROFILE_EVALUATION_PROMPT.strip()


def _empty_profile_scores(similarity: float, error: str, personamem: bool = False) -> Dict[str, float]:
    if personamem:
        return {
            "preference_coverage": None,
            "personalization_utility": None,
            "update_and_boundary_handling": None,
            "memory_quality": None,
            "overall": None,
            "similarity": similarity,
            "error": error,
        }
    return {
        "survey_consistency": None,
        "key_aspect_match": None,
        "internal_plausibility": None,
        "overall": None,
        "similarity": similarity,
        "error": error,
    }


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
    is_personamem = _is_personamem_profile_eval(prompt_set)
    schema = PersonaMemProfileEvalSchema if is_personamem else ProfileEvalSchema
    try:
        response = eval_model.generate(prompt, schema=schema, cfg=evaluation_cfg, max_tokens=PROFILE_EVAL_BUDGET)["output"]
    except Exception as e:
        logger.exception("Profile evaluation failed")
        return _empty_profile_scores(similarity, str(e), personamem=is_personamem)
    if is_personamem:
        overall_score = (
            0.30 * response["preference_coverage"]
            + 0.30 * response["personalization_utility"]
            + 0.25 * response["update_and_boundary_handling"]
            + 0.15 * response["memory_quality"]
        )
        return {
            "preference_coverage": response["preference_coverage"],
            "personalization_utility": response["personalization_utility"],
            "update_and_boundary_handling": response["update_and_boundary_handling"],
            "memory_quality": response["memory_quality"],
            "overall": overall_score,
            "similarity": similarity,
            "aspects_covered": response.get("aspects_covered", []),
            "justification": response.get("justification", ""),
        }
    overall_score = 0.4 * response["survey_consistency"] + 0.4 * response["key_aspect_match"] + 0.2 * response["internal_plausibility"]
    return {
        "survey_consistency": response["survey_consistency"],
        "key_aspect_match": response["key_aspect_match"],
        "internal_plausibility": response["internal_plausibility"],
        "overall": overall_score,
        "similarity": similarity,
        "aspects_covered": response.get("aspects_covered", []),
        "justification": response.get("justification", ""),
    }
    
