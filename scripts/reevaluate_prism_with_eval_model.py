import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, create_model
from tqdm import tqdm

from data import UserData, load_data
from eval.profile import PROFILE_EVAL_BUDGET
from eval.prediction import predict_choice
from eval.response import EVAL_BUDGET
from eval.runner import (
    _flatten_turn_contexts,
    _load_record,
    _metrics_cache_matches,
    _prediction_profile_for_turn,
    _write_json,
    summarize_metrics,
    summarize_user_metrics,
)
from model import EmbedConfig, GenerationConfig, load_model
from model.openrouter_model import OpenRouterModel
from prompt import PromptSet, load_prompt_adapter
from run import dump_provider_report, model_name


DEFAULT_BASELINES = [
    "cot_gpt5_openrouter",
    "rag_gpt5_openrouter",
    "cheatsheet_gpt5_openrouter",
    "hypogenic_gpt5_openrouter",
]

RERANK_ONLY_BASELINES = {"hydra_reranker_prism"}
BASELINE_PROFILE_SOURCES = {
    "cheatsheet_gpt5_openrouter": "cheatsheets",
    "hypogenic_gpt5_openrouter": "final_hypotheses",
}


class LooseProfileEvalSchema(BaseModel):
    aspects_covered: list[str] = Field(..., description="List of key aspects that are covered in the profile")
    survey_consistency: float = Field(..., description="0-5 score for how well the profile captures the user's claimed preferences and expectations in the survey")
    key_aspect_match: float = Field(..., description="0-5 score for how well the profile covers the user's prioritized aspects")
    internal_plausibility: float = Field(..., description="0-5 score for the consistency and plausibility of the profile given user demographics")
    justification: str = Field(..., description="a brief explanation")


def load_yaml_config(config_root: Path, path: str) -> Dict[str, Any]:
    OmegaConf.register_new_resolver("include", lambda p: OmegaConf.load(config_root / p), replace=True)
    cfg = OmegaConf.load(config_root / path)
    OmegaConf.resolve(cfg)
    return OmegaConf.to_container(cfg, resolve=True)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, indent=4)
    tmp_path.replace(path)


def load_eval_model_config(config_root: Path, eval_model_config: str) -> Dict[str, Any]:
    return load_yaml_config(config_root, eval_model_config)


def provider_report(model) -> Optional[Dict[str, Any]]:
    if isinstance(model, OpenRouterModel):
        return model.provider_report()
    return None


def load_profile_text(path: Path) -> str:
    if path.suffix == ".txt":
        return path.read_text(encoding="utf-8").strip()
    data = _load_record(path)
    hypotheses = data.get("final_hypotheses", [])
    if not hypotheses:
        return json.dumps(data, ensure_ascii=False, indent=2)
    parts = []
    for hyp in hypotheses:
        parts.append("\n".join([
            f"Hypothesis: {hyp.get('hypothesis', '')}",
            f"Evidence: {hyp.get('evidence', '')}",
            f"Confidence: {hyp.get('confidence', '')}",
        ]))
    return "\n\n".join(parts)


def baseline_profile_source(baseline_dir: Path, user_id: str) -> Optional[Path]:
    source_subdir = BASELINE_PROFILE_SOURCES.get(baseline_dir.name)
    if source_subdir is None:
        return None
    for suffix in (".txt", ".json"):
        source = baseline_dir / source_subdir / f"{user_id}{suffix}"
        if source.exists():
            return source
    return None


def _score_0_to_5(value: Any) -> Optional[float]:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(5.0, score))


def _embedding_fields(source_adaptation: Dict[str, Any], chosen_idx: int) -> Dict[str, Any]:
    similarity_scores = source_adaptation.get("similarity_scores") or []
    rejected_similarity_scores = source_adaptation.get("rejected_similarity_scores")
    if rejected_similarity_scores is None and similarity_scores:
        rejected_similarity_scores = [
            score for idx, score in enumerate(similarity_scores)
            if idx != chosen_idx
        ]
    return {
        "similarity_score": source_adaptation.get("similarity_score"),
        "relative_score": source_adaptation.get("relative_score"),
        "relative_mean_score": source_adaptation.get("relative_mean_score"),
        "similarity_scores": similarity_scores,
        "chosen_idx": source_adaptation.get("chosen_idx", chosen_idx),
        "rejected_similarity_scores": rejected_similarity_scores or [],
    }


def evaluate_generation_judge_only(
    eval_model,
    conversation_history,
    adapted_response: str,
    source_adaptation: Dict[str, Any],
    eval_cfg: GenerationConfig,
    prompts: PromptSet,
) -> Dict[str, Any]:
    current_turn = conversation_history[-1]
    c = len(current_turn.candidates)
    result = _embedding_fields(source_adaptation, current_turn.chosen_idx)
    if c < 2:
        result.update({
            "success": True,
            "gpt_score": 2.5,
            "relative_gpt_score": 0,
            "relative_mean_gpt_score": 0,
            "gpt_scores": [],
            "rejected_gpt_scores": [],
            "error": "LLM judge skipped due to insufficient candidates",
        })
        return result

    prompt = prompts.response_evaluation.format(
        current_turn=current_turn.format(include_candidates=True, include_choice=False),
        adapted=adapted_response,
        c=c,
    )
    EvalSchema = create_model(
        "EvalSchema",
        dimensions=(list[str], Field(..., description="List of key dimensions used for evaluation")),
        scores=(
            list[float],
            Field(..., description=f"List of {c} similarity scores in 0-5 for each candidate response"),
        ),
        justification=(str, Field(..., description="a brief explanation")),
    )
    try:
        evaluate_output = eval_model.generate(
            prompt=prompt,
            schema=EvalSchema,
            cfg=eval_cfg,
            max_tokens=EVAL_BUDGET,
        )["output"]
    except Exception as e:
        raise RuntimeError("Response judge evaluation failed") from e

    scores = [_score_0_to_5(score) for score in evaluate_output["scores"]]
    if any(score is None for score in scores):
        raise ValueError(f"Response judge returned non-numeric scores: {evaluate_output['scores']}")
    if len(scores) != c:
        raise ValueError(f"Response judge returned {len(scores)} scores, expected {c}: {scores}")
    gpt_score = scores[current_turn.chosen_idx]
    rejected_gpt_scores = [
        score for idx, score in enumerate(scores)
        if idx != current_turn.chosen_idx
    ]
    result.update({
        "success": True,
        "gpt_score": gpt_score,
        "relative_gpt_score": gpt_score - max(rejected_gpt_scores),
        "relative_mean_gpt_score": gpt_score - (sum(rejected_gpt_scores) / len(rejected_gpt_scores)),
        "gpt_scores": scores,
        "rejected_gpt_scores": rejected_gpt_scores,
    })
    return result


def evaluate_profile_judge_only(
    eval_model,
    final_profile: str,
    survey: str,
    source_profile_alignment: Dict[str, Any],
    eval_cfg: GenerationConfig,
    prompts: PromptSet,
) -> Dict[str, Any]:
    similarity = source_profile_alignment.get("similarity")
    prompt = prompts.profile_evaluation.format(profile=final_profile, survey=survey)
    try:
        response = eval_model.generate(
            prompt,
            schema=LooseProfileEvalSchema,
            cfg=eval_cfg,
            max_tokens=PROFILE_EVAL_BUDGET,
        )["output"]
    except Exception as e:
        raise RuntimeError("Profile judge evaluation failed") from e
    survey_consistency = _score_0_to_5(response["survey_consistency"])
    key_aspect_match = _score_0_to_5(response["key_aspect_match"])
    internal_plausibility = _score_0_to_5(response["internal_plausibility"])
    if survey_consistency is None or key_aspect_match is None or internal_plausibility is None:
        raise ValueError(f"Profile judge returned non-numeric scores: {response}")
    overall_score = 0.4 * survey_consistency + 0.4 * key_aspect_match + 0.2 * internal_plausibility
    return {
        "survey_consistency": survey_consistency,
        "key_aspect_match": key_aspect_match,
        "internal_plausibility": internal_plausibility,
        "overall": overall_score,
        "similarity": similarity,
        "aspects_covered": response.get("aspects_covered", []),
        "justification": response.get("justification", ""),
    }


def evaluate_main_user(
    record_file: Path,
    source_metrics_file: Path,
    user_data: UserData,
    eval_model_config: Dict[str, Any],
    embed_cfg: EmbedConfig,
    prompts: PromptSet,
) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
    eval_cfg = GenerationConfig(**eval_model_config)
    eval_model = load_model(backend=eval_model_config["backend"], default_cfg=eval_cfg)
    try:
        record = _load_record(record_file)
        source_metrics = _load_record(source_metrics_file)
        contexts = _flatten_turn_contexts(user_data)
        record_turns = record.get("turns", [])
        source_turns = source_metrics.get("turns", [])
        n_eval_turns = min(len(record_turns), len(source_turns), len(contexts))
        metrics: Dict[str, Any] = {
            "user": record.get("user", user_data.user_id),
            "n_record_turns": len(record_turns),
            "n_dataset_turns": len(contexts),
            "turns": [],
            "eval_model": model_name(eval_model_config),
            "prediction_model": model_name(eval_model_config),
            "eval_scope": "full",
            "reused_embedding": True,
            "reused_prediction": False,
        }
        profile_before_turn = ""
        for idx in range(n_eval_turns):
            turn_record = record_turns[idx]
            source_turn = source_turns[idx]
            adapted = turn_record.get("adapted", {})
            adapted_response = adapted.get("response") if adapted.get("success") else None
            turn_metrics: Dict[str, Any] = {
                "turn_index": idx,
                "turn_id": contexts[idx][-1].turn_id,
                "adapted_success": bool(adapted.get("success")),
            }
            turn_metrics["prediction"] = predict_choice(
                model=eval_model,
                conversation_history=contexts[idx],
                profile=_prediction_profile_for_turn(turn_record, profile_before_turn),
                generation_cfg=eval_cfg,
                prompts=prompts,
                loose_schema=True,
            )
            if not turn_metrics["prediction"].get("success"):
                raise RuntimeError(
                    f"Prediction failed for {metrics['user']} turn {idx}: "
                    f"{turn_metrics['prediction'].get('reason')}"
                )
            if adapted_response:
                turn_metrics["adaptation"] = evaluate_generation_judge_only(
                    eval_model=eval_model,
                    conversation_history=contexts[idx],
                    adapted_response=adapted_response,
                    source_adaptation=source_turn.get("adaptation", {}),
                    eval_cfg=eval_cfg,
                    prompts=prompts,
                )
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
        if final_profile:
            metrics["profile_alignment"] = evaluate_profile_judge_only(
                eval_model=eval_model,
                final_profile=final_profile,
                survey=user_data.gt_profile,
                source_profile_alignment=source_metrics.get("profile_alignment", {}),
                eval_cfg=eval_cfg,
                prompts=prompts,
            )
        else:
            metrics["profile_alignment"] = {"error": "Final profile missing."}
        metrics["summary"] = summarize_user_metrics(metrics)
        return metrics.get("user", user_data.user_id), metrics, provider_report(eval_model)
    finally:
        close = getattr(eval_model, "close", None)
        if close:
            close()


def evaluate_main_run(
    run_path: Path,
    metrics_name: str,
    users: List[UserData],
    eval_model_config: Dict[str, Any],
    embed_cfg: EmbedConfig,
    prompts: PromptSet,
    workers: int,
) -> None:
    users_by_id = {user.user_id: user for user in users}
    records_path = run_path / "records"
    metrics_path = run_path / metrics_name
    metrics_users_path = metrics_path / "users"
    metrics_users_path.mkdir(parents=True, exist_ok=True)

    source_metrics_root = run_path / "metrics" / "users"
    eval_jobs: List[Tuple[Path, Path, UserData]] = []
    user_metrics: List[Dict[str, Any]] = []
    for record_file in sorted(records_path.glob("*.json")):
        record = _load_record(record_file)
        user_id = record.get("user", record_file.stem)
        user_data = users_by_id.get(user_id)
        if user_data is None:
            continue
        user_metrics_path = metrics_users_path / f"{user_id}.json"
        if user_metrics_path.exists():
            cached = _load_record(user_metrics_path)
            if _metrics_cache_matches(
                cached,
                eval_model_name=model_name(eval_model_config),
                prediction_model_name=model_name(eval_model_config),
                eval_scope="full",
            ) and cached.get("reused_embedding") is True and cached.get("reused_prediction") is False:
                user_metrics.append(cached)
                continue
        source_metrics_file = source_metrics_root / f"{user_id}.json"
        if not source_metrics_file.exists():
            print(f"Warning: skipping {user_id}; missing source metrics for embedding reuse: {source_metrics_file}")
            continue
        eval_jobs.append((record_file, source_metrics_file, user_data))

    print(f"[main] {run_path}: {len(user_metrics)} cached, {len(eval_jobs)} to evaluate")
    reports: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(evaluate_main_user, record_file, source_metrics_file, user_data, eval_model_config, embed_cfg, prompts): user_data.user_id
            for record_file, source_metrics_file, user_data in eval_jobs
        }
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Evaluating main run", unit="user")
        for future in pbar:
            user_id = futures[future]
            pbar.set_postfix(user=user_id)
            completed_user_id, metrics, report = future.result()
            _write_json(metrics_users_path / f"{completed_user_id}.json", metrics)
            user_metrics.append(metrics)
            if report:
                reports.append(report)

    summary = summarize_metrics(user_metrics)
    _write_json(metrics_path / "summary.json", summary)
    if reports:
        dump_provider_report(metrics_path / "eval_provider_report.json", reports)
    print(f"[main] wrote {metrics_path / 'summary.json'} ({summary['n_users']} users, {summary['n_turns']} turns)")


def evaluate_baseline_user(
    user_file: Path,
    user_data: UserData,
    eval_model_config: Dict[str, Any],
    embed_cfg: EmbedConfig,
    prompts: PromptSet,
) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
    eval_cfg = GenerationConfig(**eval_model_config)
    eval_model = load_model(backend=eval_model_config["backend"], default_cfg=eval_cfg)
    try:
        source = _load_record(user_file)
        contexts = _flatten_turn_contexts(user_data)
        source_turns = source.get("turns", [])
        n_eval_turns = min(len(source_turns), len(contexts))
        metrics: Dict[str, Any] = {
            "user": source.get("user", user_file.stem),
            "n_record_turns": len(source_turns),
            "n_dataset_turns": len(contexts),
            "turns": [],
            "eval_model": model_name(eval_model_config),
            "prediction_model": source.get("prediction_model"),
            "eval_scope": "full",
            "reused_embedding": True,
            "reused_prediction": True,
        }
        if len(source_turns) != len(contexts):
            metrics["warning"] = (
                f"Baseline file has {len(source_turns)} turns but dataset has {len(contexts)} turns; "
                f"evaluating first {n_eval_turns} turns."
            )

        for idx in range(n_eval_turns):
            source_turn = source_turns[idx]
            adapted = source_turn.get("adapted", {})
            adapted_response = adapted.get("response") if adapted.get("success") else None
            turn_metrics: Dict[str, Any] = {
                "turn_index": idx,
                "turn_id": contexts[idx][-1].turn_id,
                "adapted_success": bool(adapted.get("success")),
                "prediction": source_turn.get("prediction", {"success": False}),
            }
            if adapted_response:
                adaptation = evaluate_generation_judge_only(
                    eval_model=eval_model,
                    conversation_history=contexts[idx],
                    adapted_response=adapted_response,
                    source_adaptation=source_turn.get("adaptation", {}),
                    eval_cfg=eval_cfg,
                    prompts=prompts,
                )
                turn_metrics["adaptation"] = adaptation
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

        metrics["profile_alignment"] = {"skipped": True, "reason": "baseline_has_no_final_profile"}
        metrics["summary"] = summarize_user_metrics(metrics)
        return metrics["user"], metrics, provider_report(eval_model)
    finally:
        close = getattr(eval_model, "close", None)
        if close:
            close()


def evaluate_baseline_dir(
    baseline_dir: Path,
    metrics_name: str,
    users: List[UserData],
    eval_model_config: Dict[str, Any],
    embed_cfg: EmbedConfig,
    prompts: PromptSet,
    workers: int,
) -> None:
    if baseline_dir.name in RERANK_ONLY_BASELINES:
        print(f"[baseline] skipping {baseline_dir}: rerank-only baseline has no adapted responses to re-judge")
        return

    users_by_id = {user.user_id: user for user in users}
    source_users_path = baseline_dir / "users"
    metrics_path = baseline_dir / metrics_name
    metrics_users_path = metrics_path / "users"
    metrics_users_path.mkdir(parents=True, exist_ok=True)

    eval_jobs: List[Tuple[Path, UserData]] = []
    user_metrics: List[Dict[str, Any]] = []
    for user_file in sorted(source_users_path.glob("*.json")):
        source = _load_record(user_file)
        user_id = source.get("user", user_file.stem)
        user_data = users_by_id.get(user_id)
        if user_data is None:
            continue
        metrics_user_path = metrics_users_path / f"{user_id}.json"
        if metrics_user_path.exists():
            cached = _load_record(metrics_user_path)
            if _metrics_cache_matches(
                cached,
                eval_model_name=model_name(eval_model_config),
                eval_scope="full",
            ):
                user_metrics.append(cached)
                continue
        eval_jobs.append((user_file, user_data))

    print(f"[baseline] {baseline_dir}: {len(user_metrics)} cached, {len(eval_jobs)} to evaluate")
    reports: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(evaluate_baseline_user, user_file, user_data, eval_model_config, embed_cfg, prompts): user_data.user_id
            for user_file, user_data in eval_jobs
        }
        pbar = tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {baseline_dir.name}", unit="user")
        for future in pbar:
            user_id = futures[future]
            pbar.set_postfix(user=user_id)
            completed_user_id, metrics, report = future.result()
            _write_json(metrics_users_path / f"{completed_user_id}.json", metrics)
            user_metrics.append(metrics)
            if report:
                reports.append(report)

    summary = summarize_metrics(user_metrics)
    source_summary_path = baseline_dir / "summary.json"
    if source_summary_path.exists():
        source_summary = _load_record(source_summary_path)
        summary["source_model_info"] = source_summary.get("model_info")
        summary["source_alignment_info"] = source_summary.get("alignment_info")
    summary["reeval_model"] = model_name(eval_model_config)
    _write_json(metrics_path / "summary.json", summary)
    if reports:
        dump_provider_report(metrics_path / "eval_provider_report.json", reports)
    evaluate_baseline_profiles(
        baseline_dir=baseline_dir,
        metrics_name=metrics_name,
        users=users,
        eval_model_config=eval_model_config,
        prompts=prompts,
        workers=workers,
    )
    print(f"[baseline] wrote {metrics_path / 'summary.json'} ({summary['n_users']} users, {summary['n_turns']} turns)")


def evaluate_baseline_profile_user(
    baseline_dir: Path,
    user_data: UserData,
    eval_model_config: Dict[str, Any],
    prompts: PromptSet,
) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
    source = baseline_profile_source(baseline_dir, user_data.user_id)
    if source is None:
        raise FileNotFoundError(f"Missing profile source for {baseline_dir.name}/{user_data.user_id}")
    source_profile_path = baseline_dir / "profile_metrics" / "users" / f"{user_data.user_id}.json"
    source_profile_alignment = {}
    if source_profile_path.exists():
        source_profile_alignment = _load_record(source_profile_path).get("profile_alignment", {})

    eval_cfg = GenerationConfig(**eval_model_config)
    eval_model = load_model(backend=eval_model_config["backend"], default_cfg=eval_cfg)
    try:
        profile_alignment = evaluate_profile_judge_only(
            eval_model=eval_model,
            final_profile=load_profile_text(source),
            survey=user_data.gt_profile,
            source_profile_alignment=source_profile_alignment,
            eval_cfg=eval_cfg,
            prompts=prompts,
        )
        return (
            user_data.user_id,
            {
                "user": user_data.user_id,
                "profile_alignment": profile_alignment,
                "source": str(source),
                "eval_model": model_name(eval_model_config),
            },
            provider_report(eval_model),
        )
    finally:
        close = getattr(eval_model, "close", None)
        if close:
            close()


def evaluate_baseline_profiles(
    baseline_dir: Path,
    metrics_name: str,
    users: List[UserData],
    eval_model_config: Dict[str, Any],
    prompts: PromptSet,
    workers: int,
) -> None:
    if baseline_dir.name not in BASELINE_PROFILE_SOURCES:
        return
    output_dir = baseline_dir / f"profile_metrics_{metrics_name}" / "users"
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    cached = 0
    for user in users:
        if baseline_profile_source(baseline_dir, user.user_id) is None:
            continue
        output_path = output_dir / f"{user.user_id}.json"
        if output_path.exists():
            existing = _load_record(output_path)
            if existing.get("eval_model") == model_name(eval_model_config):
                cached += 1
                continue
        jobs.append(user)
    print(f"[baseline-profile] {baseline_dir}: {cached} cached, {len(jobs)} to evaluate")
    reports: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(evaluate_baseline_profile_user, baseline_dir, user, eval_model_config, prompts): user.user_id
            for user in jobs
        }
        pbar = tqdm(as_completed(futures), total=len(futures), desc=f"Profile {baseline_dir.name}", unit="user")
        for future in pbar:
            user_id = futures[future]
            pbar.set_postfix(user=user_id)
            completed_user_id, metrics, report = future.result()
            _write_json(output_dir / f"{completed_user_id}.json", metrics)
            if report:
                reports.append(report)
    if reports:
        dump_provider_report(baseline_dir / f"profile_metrics_{metrics_name}" / "eval_provider_report.json", reports)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate PRISM main and baseline artifacts with a different eval model.")
    parser.add_argument("--config-root", type=str, default="config")
    parser.add_argument("--run-config", type=str, default="run/openrouter_gpt5_trace_gemini3_flash_eval.yaml")
    parser.add_argument("--eval-model-config", type=str, default="model/openrouter-minimax-m2.7.yaml")
    parser.add_argument("--main-run", type=str, default="result/openrouter-gpt5-trace-gemini3-flash-eval")
    parser.add_argument("--baseline-root", type=str, default="baseline_results")
    parser.add_argument("--baseline", action="append", default=None, help="Baseline directory name under --baseline-root; repeatable.")
    parser.add_argument("--metrics-name", type=str, default="metrics_minimax_m27")
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--skip-main", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    config_root = Path(args.config_root)
    run_config = load_yaml_config(config_root, args.run_config)
    eval_model_config = load_eval_model_config(config_root, args.eval_model_config)
    run_config["eval_model"] = eval_model_config

    users = load_data(run_config["dataset"], n_users=run_config["n_users"], seed=run_config["seed"])
    embed_cfg = EmbedConfig(**run_config["embed"])
    prompts = load_prompt_adapter(run_config.get("prompt_adapter", run_config["dataset"]))
    print(f"Loaded {len(users)} users from {run_config['dataset']}")
    print(f"Eval model: {model_name(eval_model_config)}")
    print(f"Metrics name: {args.metrics_name}")

    if not args.skip_main:
        evaluate_main_run(
            run_path=Path(args.main_run),
            metrics_name=args.metrics_name,
            users=users,
            eval_model_config=eval_model_config,
            embed_cfg=embed_cfg,
            prompts=prompts,
            workers=args.eval_workers,
        )

    if not args.skip_baselines:
        baselines = args.baseline or DEFAULT_BASELINES
        for baseline_name in baselines:
            evaluate_baseline_dir(
                baseline_dir=Path(args.baseline_root) / baseline_name,
                metrics_name=args.metrics_name,
                users=users,
                eval_model_config=eval_model_config,
                embed_cfg=embed_cfg,
                prompts=prompts,
                workers=args.eval_workers,
            )


if __name__ == "__main__":
    main()
