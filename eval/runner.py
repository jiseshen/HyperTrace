import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm

from data import Turn, UserData
from model import BaseLM, EmbedConfig, GenerationConfig
from prompt import PromptSet, prism_prompts

from .prediction import predict_choice
from .profile import profile_score
from .response import evaluate_generation


def _flatten_turn_contexts(user: UserData) -> List[List[Turn]]:
    contexts: List[List[Turn]] = []
    for conversation in user.conversations:
        history: List[Turn] = []
        for turn in conversation.turns:
            history.append(turn)
            contexts.append(list(history))
    return contexts


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _load_record(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=4)


def _prediction_profile_for_turn(turn_record: Dict[str, Any], profile_before_turn: str) -> str:
    if "prediction_profile" in turn_record:
        return turn_record.get("prediction_profile") or ""
    if "inference_profile" in turn_record:
        return turn_record.get("inference_profile") or ""
    return profile_before_turn


def _metrics_cache_matches(
    metrics: Dict[str, Any],
    eval_model_name: str = None,
    prediction_model_name: str = None,
    eval_scope: str = None,
) -> bool:
    if eval_scope is not None and metrics.get("eval_scope") != eval_scope:
        return False
    if eval_model_name is not None and metrics.get("eval_model") != eval_model_name:
        return False
    if prediction_model_name is not None and metrics.get("prediction_model") != prediction_model_name:
        return False
    return True


def evaluate_user_record(
    user_data: UserData,
    record: Dict[str, Any],
    eval_model: BaseLM,
    eval_cfg: GenerationConfig,
    embed_cfg: EmbedConfig,
    prompts: PromptSet = None,
    prediction_model: BaseLM = None,
    prediction_cfg: GenerationConfig = None,
    eval_model_name: str = None,
    prediction_model_name: str = None,
    prediction_only: bool = False,
) -> Dict[str, Any]:
    prompt_set = prompts or prism_prompts()
    prediction_model = prediction_model or eval_model
    prediction_cfg = prediction_cfg or eval_cfg
    eval_scope = "prediction_only" if prediction_only else "full"
    turn_contexts = _flatten_turn_contexts(user_data)
    record_turns = record.get("turns", [])
    n_eval_turns = min(len(record_turns), len(turn_contexts))

    metrics: Dict[str, Any] = {
        "user": record.get("user", user_data.user_id),
        "n_record_turns": len(record_turns),
        "n_dataset_turns": len(turn_contexts),
        "turns": [],
        "eval_model": eval_model_name,
        "prediction_model": prediction_model_name or eval_model_name,
        "eval_scope": eval_scope,
    }
    if len(record_turns) != len(turn_contexts):
        metrics["warning"] = (
            f"Record has {len(record_turns)} turns but dataset has {len(turn_contexts)} turns; "
            f"evaluating first {n_eval_turns} turns."
        )

    profile_before_turn = ""
    for idx in range(n_eval_turns):
        turn_record = record_turns[idx]
        conversation_history = turn_contexts[idx]
        adapted = turn_record.get("adapted", {})
        adapted_response = adapted.get("response") if adapted.get("success") else None

        turn_metrics: Dict[str, Any] = {
            "turn_index": idx,
            "turn_id": conversation_history[-1].turn_id,
            "adapted_success": bool(adapted.get("success")),
        }

        turn_metrics["prediction"] = predict_choice(
            model=prediction_model,
            conversation_history=conversation_history,
            profile=_prediction_profile_for_turn(turn_record, profile_before_turn),
            generation_cfg=prediction_cfg,
            prompts=prompt_set,
        )

        if prediction_only:
            pass
        elif adapted_response:
            turn_metrics["adaptation"] = evaluate_generation(
                eval_model=eval_model,
                conversation_history=conversation_history,
                adapted_response=adapted_response,
                embed_cfg=embed_cfg,
                evaluation_cfg=eval_cfg,
                prompts=prompt_set,
            )
            turn_metrics["adaptation"]["success"] = True
        else:
            turn_metrics["adaptation"] = {
                "success": False,
                "error": "Adapted response missing or unsuccessful.",
                "gpt_score": None,
                "relative_gpt_score": None,
                "relative_mean_gpt_score": None,
                "gpt_scores": [],
                "similarity_score": None,
                "relative_score": None,
                "relative_mean_score": None,
                "similarity_scores": [],
            }

        metrics["turns"].append(turn_metrics)
        if turn_record.get("summary"):
            profile_before_turn = turn_record["summary"]

    final_profile = record.get("final_profile")
    if prediction_only:
        metrics["profile_alignment"] = {"skipped": True, "reason": "prediction_only"}
    elif final_profile:
        metrics["profile_alignment"] = profile_score(
            eval_model=eval_model,
            profile=final_profile,
            survey=user_data.gt_profile,
            embed_cfg=embed_cfg,
            evaluation_cfg=eval_cfg,
            prompts=prompt_set,
        )
    else:
        metrics["profile_alignment"] = {"error": "Final profile missing."}

    metrics["summary"] = summarize_user_metrics(metrics)
    return metrics


def summarize_user_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    turns = metrics.get("turns", [])
    predictions = [turn.get("prediction", {}) for turn in turns]
    valid_predictions = [p for p in predictions if p.get("success")]
    adaptations = [turn.get("adaptation", {}) for turn in turns]
    valid_adaptations = [a for a in adaptations if a.get("success")]
    profile_alignment = metrics.get("profile_alignment", {})

    return {
        "n_turns": len(turns),
        "n_prediction_turns": len(valid_predictions),
        "n_adaptation_turns": len(valid_adaptations),
        "prediction_success_rate": _mean(1.0 if p.get("success") else 0.0 for p in predictions),
        "prediction_accuracy": _mean(p.get("accuracy") for p in valid_predictions),
        "prediction_ranking_score": _mean(p.get("ranking_score") for p in valid_predictions),
        "adapt_gpt_score": _mean(a.get("gpt_score") for a in valid_adaptations),
        "adapt_relative_gpt_score": _mean(a.get("relative_gpt_score") for a in valid_adaptations),
        "adapt_relative_mean_gpt_score": _mean(a.get("relative_mean_gpt_score") for a in valid_adaptations),
        "adapt_similarity_score": _mean(a.get("similarity_score") for a in valid_adaptations),
        "adapt_relative_score": _mean(a.get("relative_score") for a in valid_adaptations),
        "adapt_relative_mean_score": _mean(a.get("relative_mean_score") for a in valid_adaptations),
        "profile_overall": profile_alignment.get("overall"),
        "profile_similarity": profile_alignment.get("similarity"),
        "profile_survey_consistency": profile_alignment.get("survey_consistency"),
        "profile_key_aspect_match": profile_alignment.get("key_aspect_match"),
        "profile_internal_plausibility": profile_alignment.get("internal_plausibility"),
        "profile_preference_coverage": profile_alignment.get("preference_coverage"),
        "profile_personalization_utility": profile_alignment.get("personalization_utility"),
        "profile_update_and_boundary_handling": profile_alignment.get("update_and_boundary_handling"),
        "profile_memory_quality": profile_alignment.get("memory_quality"),
    }


def summarize_metrics(user_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    profile_metrics = [metrics.get("profile_alignment", {}) for metrics in user_metrics]
    all_turns = [
        turn
        for metrics in user_metrics
        for turn in metrics.get("turns", [])
    ]
    all_predictions = [turn.get("prediction", {}) for turn in all_turns]
    valid_predictions = [p for p in all_predictions if p.get("success")]
    all_adaptations = [turn.get("adaptation", {}) for turn in all_turns]
    valid_adaptations = [a for a in all_adaptations if a.get("success")]
    max_turns = max((len(metrics.get("turns", [])) for metrics in user_metrics), default=0)
    online_turns = []
    for turn_index in range(max_turns):
        turn_metrics = [
            metrics["turns"][turn_index]
            for metrics in user_metrics
            if turn_index < len(metrics.get("turns", []))
        ]
        predictions = [turn.get("prediction", {}) for turn in turn_metrics]
        valid_turn_predictions = [p for p in predictions if p.get("success")]
        adaptations = [turn.get("adaptation", {}) for turn in turn_metrics]
        valid_turn_adaptations = [a for a in adaptations if a.get("success")]
        online_turns.append({
            "turn_index": turn_index,
            "n_users": len(turn_metrics),
            "n_prediction_users": len(valid_turn_predictions),
            "n_adaptation_users": len(valid_turn_adaptations),
            "prediction_success_rate": _mean(1.0 if p.get("success") else 0.0 for p in predictions),
            "prediction_accuracy": _mean(p.get("accuracy") for p in valid_turn_predictions),
            "prediction_ranking_score": _mean(p.get("ranking_score") for p in valid_turn_predictions),
            "adapt_gpt_score": _mean(a.get("gpt_score") for a in valid_turn_adaptations),
            "adapt_relative_gpt_score": _mean(a.get("relative_gpt_score") for a in valid_turn_adaptations),
            "adapt_relative_mean_gpt_score": _mean(a.get("relative_mean_gpt_score") for a in valid_turn_adaptations),
            "adapt_similarity_score": _mean(a.get("similarity_score") for a in valid_turn_adaptations),
            "adapt_relative_score": _mean(a.get("relative_score") for a in valid_turn_adaptations),
            "adapt_relative_mean_score": _mean(a.get("relative_mean_score") for a in valid_turn_adaptations),
        })

    average_adaptation = {
        "adapt_gpt_score": _mean(turn.get("adapt_gpt_score") for turn in online_turns),
        "adapt_relative_gpt_score": _mean(turn.get("adapt_relative_gpt_score") for turn in online_turns),
        "adapt_relative_mean_gpt_score": _mean(turn.get("adapt_relative_mean_gpt_score") for turn in online_turns),
        "adapt_similarity_score": _mean(turn.get("adapt_similarity_score") for turn in online_turns),
        "adapt_relative_score": _mean(turn.get("adapt_relative_score") for turn in online_turns),
        "adapt_relative_mean_score": _mean(turn.get("adapt_relative_mean_score") for turn in online_turns),
    }

    overall_prediction = {
        "prediction_success_rate": _mean(1.0 if p.get("success") else 0.0 for p in all_predictions),
        "prediction_accuracy": _mean(p.get("accuracy") for p in valid_predictions),
        "prediction_ranking_score": _mean(p.get("ranking_score") for p in valid_predictions),
    }

    overall_adaptation = {
        "adapt_gpt_score": _mean(a.get("gpt_score") for a in valid_adaptations),
        "adapt_relative_gpt_score": _mean(a.get("relative_gpt_score") for a in valid_adaptations),
        "adapt_relative_mean_gpt_score": _mean(a.get("relative_mean_gpt_score") for a in valid_adaptations),
        "adapt_similarity_score": _mean(a.get("similarity_score") for a in valid_adaptations),
        "adapt_relative_score": _mean(a.get("relative_score") for a in valid_adaptations),
        "adapt_relative_mean_score": _mean(a.get("relative_mean_score") for a in valid_adaptations),
    }

    profile_alignment = {
        "profile_overall": _mean(profile.get("overall") for profile in profile_metrics),
        "profile_similarity": _mean(profile.get("similarity") for profile in profile_metrics),
        "profile_survey_consistency": _mean(profile.get("survey_consistency") for profile in profile_metrics),
        "profile_key_aspect_match": _mean(profile.get("key_aspect_match") for profile in profile_metrics),
        "profile_internal_plausibility": _mean(profile.get("internal_plausibility") for profile in profile_metrics),
        "profile_preference_coverage": _mean(profile.get("preference_coverage") for profile in profile_metrics),
        "profile_personalization_utility": _mean(profile.get("personalization_utility") for profile in profile_metrics),
        "profile_update_and_boundary_handling": _mean(profile.get("update_and_boundary_handling") for profile in profile_metrics),
        "profile_memory_quality": _mean(profile.get("memory_quality") for profile in profile_metrics),
    }

    return {
        "n_users": len(user_metrics),
        "n_turns": sum(len(metrics.get("turns", [])) for metrics in user_metrics),
        "eval_scope": next((metrics.get("eval_scope") for metrics in user_metrics if metrics.get("eval_scope")), None),
        "eval_model": next((metrics.get("eval_model") for metrics in user_metrics if metrics.get("eval_model")), None),
        "prediction_model": next((metrics.get("prediction_model") for metrics in user_metrics if metrics.get("prediction_model")), None),
        "online_turns": online_turns,
        "overall_prediction": overall_prediction,
        "overall_adaptation": overall_adaptation,
        "average_adaptation": average_adaptation,
        "profile_alignment": profile_alignment,
        "users": {
            metrics.get("user"): metrics.get("summary", {})
            for metrics in user_metrics
        },
    }


def evaluate_records(
    users: List[UserData],
    records_path: Path,
    metrics_path: Path,
    eval_model: BaseLM,
    eval_cfg: GenerationConfig,
    embed_cfg: EmbedConfig,
    prompts: PromptSet = None,
    prediction_model: BaseLM = None,
    prediction_cfg: GenerationConfig = None,
    eval_model_name: str = None,
    prediction_model_name: str = None,
    prediction_only: bool = False,
) -> Dict[str, Any]:
    prompt_set = prompts or prism_prompts()
    users_by_id = {user.user_id: user for user in users}
    record_files = sorted(records_path.glob("*.json"))
    user_metrics: List[Dict[str, Any]] = []

    pbar = tqdm(record_files, desc="Evaluating records", unit="user")
    for record_file in pbar:
        record = _load_record(record_file)
        user_id = record.get("user", record_file.stem)
        pbar.set_postfix(user=user_id)

        user_data = users_by_id.get(user_id)
        if user_data is None:
            continue

        user_metrics_path = metrics_path / "users" / f"{user_id}.json"
        if user_metrics_path.exists():
            cached_metrics = _load_record(user_metrics_path)
            if _metrics_cache_matches(
                cached_metrics,
                eval_model_name=eval_model_name,
                prediction_model_name=prediction_model_name,
                eval_scope="prediction_only" if prediction_only else "full",
            ):
                user_metrics.append(cached_metrics)
                continue

        metrics = evaluate_user_record(
            user_data=user_data,
            record=record,
            eval_model=eval_model,
            eval_cfg=eval_cfg,
            embed_cfg=embed_cfg,
            prompts=prompt_set,
            prediction_model=prediction_model,
            prediction_cfg=prediction_cfg,
            eval_model_name=eval_model_name,
            prediction_model_name=prediction_model_name,
            prediction_only=prediction_only,
        )
        _write_json(user_metrics_path, metrics)
        user_metrics.append(metrics)

    summary = summarize_metrics(user_metrics)
    _write_json(metrics_path / "summary.json", summary)
    return summary
