import json
from argparse import ArgumentParser
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import OmegaConf
from tqdm import tqdm

from core.preference_tracer import PreferenceTracer
from core.utils import OverrideConfig, TracerConfig
from data import UserData, load_data
from eval.runner import _load_record, _write_json, evaluate_user_record, summarize_metrics
from model import EmbedConfig, GenerationConfig, load_model
from model.openrouter_model import OpenRouterModel
from prompt import PromptSet, load_prompt_adapter


def load_run_config(config_root: Path, config_path: str) -> Dict[str, Any]:
    OmegaConf.register_new_resolver("include", lambda path: OmegaConf.load(config_root / path), replace=True)
    config = OmegaConf.load(config_root / config_path)
    OmegaConf.resolve(config)
    return OmegaConf.to_container(config, resolve=True)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, indent=4)
    tmp_path.replace(path)


def target_trace_users(users: List[UserData], records_path: Path, users_per_run: int) -> Tuple[List[UserData], List[UserData], set[str]]:
    finished_ids = {path.stem for path in records_path.glob("*.json")}
    target_user_count = min(users_per_run, len(users))
    target_sample = users[:target_user_count]
    target_users = [user for user in target_sample if user.user_id not in finished_ids]
    finished_target_ids = {user.user_id for user in target_sample if user.user_id in finished_ids}
    return target_sample, target_users, finished_target_ids


def dump_provider_report(path: Path, reports: List[Dict[str, Any]]) -> None:
    attempts = [
        attempt
        for report in reports
        for attempt in report.get("attempts", [])
    ]
    by_provider: Dict[str, Dict[str, Any]] = {}
    for attempt in attempts:
        provider = attempt.get("provider", "unknown")
        bucket = by_provider.setdefault(
            provider,
            {
                "attempts": 0,
                "successes": 0,
                "errors": 0,
                "models": {},
                "error_types": {},
            },
        )
        bucket["attempts"] += 1
        model = attempt.get("model", "unknown")
        bucket["models"][model] = bucket["models"].get(model, 0) + 1
        if attempt.get("success"):
            bucket["successes"] += 1
        else:
            bucket["errors"] += 1
            error_type = attempt.get("error_type") or "UnknownError"
            bucket["error_types"][error_type] = bucket["error_types"].get(error_type, 0) + 1
    for bucket in by_provider.values():
        bucket["error_rate"] = bucket["errors"] / bucket["attempts"] if bucket["attempts"] else None
    write_json(
        path,
        {
            "total_attempts": len(attempts),
            "providers": by_provider,
            "attempts": attempts,
        },
    )


def format_active(progress: Dict[str, Tuple[int, int]], limit: int = 4) -> str:
    items = sorted(progress.items())
    parts = [f"{user_id}:{turn}/{total}" for user_id, (turn, total) in items[:limit]]
    if len(items) > limit:
        parts.append(f"+{len(items) - limit}")
    return " ".join(parts)


def trace_one_user(
    user: UserData,
    config: Dict[str, Any],
    tracer_cfg: TracerConfig,
    embed_cfg: EmbedConfig,
    prompts: PromptSet,
    progress: Optional[Dict[str, Tuple[int, int]]] = None,
    progress_lock: Optional[Lock] = None,
) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
    gen_cfg = GenerationConfig(**config["main_model"])
    gen_model = load_model(backend=config["main_model"]["backend"], default_cfg=gen_cfg)
    def progress_hook(user_id: str, turn_index: int, total_turns: int) -> None:
        if progress is None or progress_lock is None:
            return
        with progress_lock:
            progress[user_id] = (turn_index, total_turns)

    try:
        tracer = PreferenceTracer(
            model=gen_model,
            generation_cfg=gen_cfg,
            tracer_cfg=tracer_cfg,
            embed_cfg=embed_cfg,
            prompts=prompts,
            progress_hook=progress_hook if progress is not None else None,
        )
        records = tracer.trace(user)
        report = gen_model.provider_report() if isinstance(gen_model, OpenRouterModel) else None
        return user.user_id, records, report
    finally:
        close = getattr(gen_model, "close", None)
        if close:
            close()


def evaluate_one_user(
    record_file: Path,
    user_data: UserData,
    config: Dict[str, Any],
    embed_cfg: EmbedConfig,
    prompts: PromptSet,
) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
    eval_cfg = GenerationConfig(**config["eval_model"])
    eval_model = load_model(backend=config["eval_model"]["backend"], default_cfg=eval_cfg)
    try:
        record = _load_record(record_file)
        metrics = evaluate_user_record(
            user_data=user_data,
            record=record,
            eval_model=eval_model,
            eval_cfg=eval_cfg,
            embed_cfg=embed_cfg,
            prompts=prompts,
        )
        report = eval_model.provider_report() if isinstance(eval_model, OpenRouterModel) else None
        return metrics.get("user", user_data.user_id), metrics, report
    finally:
        close = getattr(eval_model, "close", None)
        if close:
            close()


def main() -> None:
    parser = ArgumentParser(description="Run preference tracing and eval with per-user parallelism.")
    parser.add_argument("--config", type=str, default="run/main.yaml", help="Path to the main config file")
    parser.add_argument("--config-root", type=str, default="config", help="Root directory for config files")
    parser.add_argument("--result", type=str, default=None, help="Path to save the results; defaults to run name")
    parser.add_argument("--result-root", type=str, default="result", help="Root directory for results")
    parser.add_argument("--trace-workers", type=int, default=4, help="Number of users to trace concurrently")
    parser.add_argument("--eval-workers", type=int, default=4, help="Number of user records to evaluate concurrently")
    parser.add_argument("--eval-only", action="store_true", help="Skip tracing and evaluate existing records only")
    args = parser.parse_args()

    config_root = Path(args.config_root)
    config = load_run_config(config_root, args.config)
    print(f"Loaded config: {json.dumps(config, indent=4)}")

    run_path = Path(args.result_root) / (args.result if args.result else config["name"])
    records_path = run_path / "records"
    metrics_path = run_path / "metrics"
    records_path.mkdir(parents=True, exist_ok=True)
    print(f"Records will be saved to: {records_path}")

    users = load_data(config["dataset"], n_users=config["n_users"], seed=config["seed"])
    users_by_id = {user.user_id: user for user in users}
    print(f"Loaded {len(users)} users from dataset {config['dataset']}")

    tracer_cfg = TracerConfig(**config["tracer"])
    tracer_cfg.override = OverrideConfig(**config.get("override", {}))
    embed_cfg = EmbedConfig(**config["embed"])
    prompt_adapter_name = config.get("prompt_adapter", config["dataset"])
    prompts = load_prompt_adapter(prompt_adapter_name)
    print(f"Loaded prompt adapter: {prompt_adapter_name}")

    if not args.eval_only:
        target_sample, target_users, finished_target_ids = target_trace_users(
            users,
            records_path,
            config["users_per_run"],
        )
        finished_ids = {path.stem for path in records_path.glob("*.json")}
        if finished_target_ids:
            print(f"Skipping {len(finished_target_ids)} finished users in target sample")
        if len(finished_ids) > len(finished_target_ids):
            print(f"Ignoring {len(finished_ids) - len(finished_target_ids)} finished users outside target sample")
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
            pbar = tqdm(total=len(futures), desc="Tracing preferences", unit="user")
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

    record_files = sorted(records_path.glob("*.json"))
    metrics_users_path = metrics_path / "users"
    metrics_users_path.mkdir(parents=True, exist_ok=True)
    user_metrics: List[Dict[str, Any]] = []
    eval_jobs = []
    for record_file in record_files:
        record = _load_record(record_file)
        user_id = record.get("user", record_file.stem)
        user_data = users_by_id.get(user_id)
        if user_data is None:
            continue
        user_metrics_path = metrics_users_path / f"{user_id}.json"
        if user_metrics_path.exists():
            user_metrics.append(_load_record(user_metrics_path))
        else:
            eval_jobs.append((record_file, user_data))

    print(f"Metrics will be saved to: {metrics_path}")
    eval_reports: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.eval_workers)) as executor:
        futures = {
            executor.submit(evaluate_one_user, record_file, user_data, config, embed_cfg, prompts): user_data.user_id
            for record_file, user_data in eval_jobs
        }
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Evaluating records", unit="user")
        for future in pbar:
            user_id = futures[future]
            pbar.set_postfix(user=user_id)
            completed_user_id, metrics, report = future.result()
            _write_json(metrics_users_path / f"{completed_user_id}.json", metrics)
            user_metrics.append(metrics)
            if report:
                eval_reports.append(report)
    if eval_reports:
        dump_provider_report(run_path / "eval_provider_report.json", eval_reports)
        print(f"Eval provider report saved to: {run_path / 'eval_provider_report.json'}")

    summary = summarize_metrics(user_metrics)
    _write_json(metrics_path / "summary.json", summary)
    print(f"Evaluated {summary['n_users']} users and {summary['n_turns']} turns")


if __name__ == "__main__":
    main()
