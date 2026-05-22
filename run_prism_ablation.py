import json
from argparse import ArgumentParser
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from core.utils import OverrideConfig, TracerConfig
from data import UserData, load_data
from eval.runner import _load_record, _metrics_cache_matches, _write_json, summarize_metrics
from model import EmbedConfig
from prompt import load_prompt_adapter
from run import (
    dump_provider_report,
    evaluate_one_user,
    format_active,
    load_run_config,
    model_config,
    model_name,
    model_slug,
    trace_one_user,
    write_json,
)


PRISM_ABLATION_USER_IDS = [
    "user1010",
    "user1017",
    "user1108",
    "user1221",
    "user1356",
    "user1387",
    "user1459",
    "user1483",
    "user1487",
    "user1488",
    "user151",
    "user174",
    "user21",
    "user248",
    "user272",
    "user358",
    "user362",
    "user431",
    "user508",
    "user512",
    "user516",
    "user52",
    "user627",
    "user70",
    "user773",
    "user839",
    "user874",
    "user94",
]


def select_prism_ablation_users(users: List[UserData]) -> List[UserData]:
    users_by_id = {user.user_id: user for user in users}
    missing = [user_id for user_id in PRISM_ABLATION_USER_IDS if user_id not in users_by_id]
    if missing:
        raise ValueError(f"Missing PRISM ablation users: {', '.join(missing)}")
    return [users_by_id[user_id] for user_id in PRISM_ABLATION_USER_IDS]


def target_ablation_trace_users(users: List[UserData], records_path: Path) -> Tuple[List[UserData], set[str], set[str]]:
    ablation_ids = {user.user_id for user in users}
    finished_ids = {path.stem for path in records_path.glob("*.json")}
    finished_target_ids = {user.user_id for user in users if user.user_id in finished_ids}
    stray_finished_ids = finished_ids - ablation_ids
    target_users = [user for user in users if user.user_id not in finished_target_ids]
    return target_users, finished_target_ids, stray_finished_ids


def ablation_record_files(records_path: Path) -> List[Path]:
    return [
        path
        for user_id in PRISM_ABLATION_USER_IDS
        if (path := records_path / f"{user_id}.json").exists()
    ]


def main() -> None:
    parser = ArgumentParser(description="Run PRISM ablations with fixed GPT-5-filtered users.")
    parser.add_argument("--config", type=str, required=True, help="Path to an ablation config under config/")
    parser.add_argument("--config-root", type=str, default="config", help="Root directory for config files")
    parser.add_argument("--result", type=str, default=None, help="Path to save results; defaults to run name")
    parser.add_argument("--result-root", type=str, default="result", help="Root directory for results")
    parser.add_argument("--trace-workers", type=int, default=4, help="Number of users to trace concurrently")
    parser.add_argument("--eval-workers", type=int, default=4, help="Number of user records to evaluate concurrently")
    parser.add_argument("--eval-only", action="store_true", help="Skip tracing and evaluate existing records only")
    parser.add_argument("--prediction-only", action="store_true", help="Only run offline preference prediction metrics")
    parser.add_argument("--prediction-model-from-main", action="store_true", help="Use main_model as the prediction model")
    parser.add_argument("--metrics-name", type=str, default=None, help="Metrics directory name under the run path")
    args = parser.parse_args()

    config_root = Path(args.config_root)
    config = load_run_config(config_root, args.config)
    if config.get("dataset") != "prism":
        raise ValueError("run_prism_ablation.py only supports dataset: prism")
    if args.prediction_model_from_main:
        config["prediction_model"] = config["main_model"]
    print(f"Loaded config: {json.dumps(config, indent=4)}")

    run_path = Path(args.result_root) / (args.result if args.result else config["name"])
    records_path = run_path / "records"
    records_path.mkdir(parents=True, exist_ok=True)
    print(f"Records will be saved to: {records_path}")

    all_users = load_data(config["dataset"], n_users=None, seed=config["seed"])
    target_sample = select_prism_ablation_users(all_users)
    users_by_id = {user.user_id: user for user in target_sample}
    print(f"Loaded {len(all_users)} PRISM users; selected {len(target_sample)} ablation users")

    tracer_cfg = TracerConfig(**config["tracer"])
    tracer_cfg.override = OverrideConfig(**config.get("override", {}))
    embed_cfg = EmbedConfig(**config["embed"])
    prompt_adapter_name = config.get("prompt_adapter", config["dataset"])
    prompts = load_prompt_adapter(prompt_adapter_name)
    print(f"Loaded prompt adapter: {prompt_adapter_name}")

    eval_model_config = config["eval_model"]
    prediction_model_config = model_config(config, "prediction_model")
    metrics_name = args.metrics_name
    if metrics_name is None:
        metrics_name = (
            f"metrics_prediction_{model_slug(model_name(prediction_model_config))}"
            if args.prediction_only
            else "metrics"
        )
    metrics_path = run_path / metrics_name
    print(f"Eval model: {model_name(eval_model_config)}")
    print(f"Prediction model: {model_name(prediction_model_config)}")
    print(f"Eval scope: {'prediction_only' if args.prediction_only else 'full'}")

    if not args.eval_only:
        target_users, finished_target_ids, stray_finished_ids = target_ablation_trace_users(
            target_sample,
            records_path,
        )
        if finished_target_ids:
            print(f"Skipping {len(finished_target_ids)} finished ablation users")
        if stray_finished_ids:
            print(f"Ignoring {len(stray_finished_ids)} records outside the fixed ablation user set")
        print(f"Running preference tracing for {len(target_users)} users to reach {len(target_sample)} total users")

        trace_reports: List[Dict[str, Any]] = []
        active_progress: Dict[str, Tuple[int, int]] = {}
        progress_lock = Lock()
        with ThreadPoolExecutor(max_workers=max(1, args.trace_workers)) as executor:
            futures = {
                executor.submit(
                    trace_one_user,
                    user,
                    config,
                    tracer_cfg,
                    embed_cfg,
                    prompts,
                    active_progress,
                    progress_lock,
                ): user.user_id
                for user in target_users
            }
            pending = set(futures)
            pbar = tqdm(total=len(futures), desc="Tracing PRISM ablation", unit="user")
            while pending:
                done, pending = wait(pending, timeout=2.0, return_when=FIRST_COMPLETED)
                with progress_lock:
                    active = format_active(active_progress)
                pbar.set_postfix(active=active)
                for future in done:
                    completed_user_id, records, report = future.result()
                    write_json(records_path / f"{completed_user_id}.json", records)
                    with progress_lock:
                        active_progress.pop(completed_user_id, None)
                        active = format_active(active_progress)
                    pbar.set_postfix(done=completed_user_id, active=active)
                    pbar.update(1)
                    if report:
                        trace_reports.append(report)
            pbar.close()
        if trace_reports:
            dump_provider_report(run_path / "provider_report.json", trace_reports)
            print(f"Provider report saved to: {run_path / 'provider_report.json'}")

    metrics_users_path = metrics_path / "users"
    metrics_users_path.mkdir(parents=True, exist_ok=True)
    user_metrics: List[Dict[str, Any]] = []
    eval_jobs = []
    for record_file in ablation_record_files(records_path):
        record = _load_record(record_file)
        user_id = record.get("user", record_file.stem)
        user_data = users_by_id.get(user_id)
        if user_data is None:
            continue
        user_metrics_path = metrics_users_path / f"{user_id}.json"
        if user_metrics_path.exists():
            cached_metrics = _load_record(user_metrics_path)
            if _metrics_cache_matches(
                cached_metrics,
                eval_model_name=None if args.prediction_only else model_name(eval_model_config),
                prediction_model_name=model_name(prediction_model_config),
                eval_scope="prediction_only" if args.prediction_only else "full",
            ):
                user_metrics.append(cached_metrics)
            else:
                eval_jobs.append((record_file, user_data))
        else:
            eval_jobs.append((record_file, user_data))

    print(f"Metrics will be saved to: {metrics_path}")
    eval_reports: List[Dict[str, Any]] = []
    prediction_reports: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.eval_workers)) as executor:
        futures = {
            executor.submit(
                evaluate_one_user,
                record_file,
                user_data,
                config,
                embed_cfg,
                prompts,
                args.prediction_only,
            ): user_data.user_id
            for record_file, user_data in eval_jobs
        }
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Evaluating PRISM ablation", unit="user")
        for future in pbar:
            user_id = futures[future]
            pbar.set_postfix(user=user_id)
            completed_user_id, metrics, eval_report, prediction_report = future.result()
            _write_json(metrics_users_path / f"{completed_user_id}.json", metrics)
            user_metrics.append(metrics)
            if eval_report:
                eval_reports.append(eval_report)
            if prediction_report:
                prediction_reports.append(prediction_report)
    if eval_reports:
        dump_provider_report(run_path / "eval_provider_report.json", eval_reports)
        print(f"Eval provider report saved to: {run_path / 'eval_provider_report.json'}")
    if prediction_reports:
        dump_provider_report(run_path / "prediction_provider_report.json", prediction_reports)
        print(f"Prediction provider report saved to: {run_path / 'prediction_provider_report.json'}")

    summary = summarize_metrics(user_metrics)
    _write_json(metrics_path / "summary.json", summary)
    print(f"Evaluated {summary['n_users']} users and {summary['n_turns']} turns")


if __name__ == "__main__":
    main()
