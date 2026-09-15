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
from eval.runner import _load_record, _metrics_cache_matches, _write_json as write_json, evaluate_user_record, summarize_metrics
from model import EmbedConfig, GenerationConfig, load_model
from model.openrouter_model import OpenRouterModel
from model.embed import close_embedding_clients
from prompt import PromptSet, load_prompt_adapter


def load_run_config(config_root: Path, config_path: str) -> Dict[str, Any]:
    OmegaConf.register_new_resolver("include", lambda path: OmegaConf.load(config_root / path), replace=True)
    config = OmegaConf.load(config_root / config_path)
    if set(config.keys()) == {"include"}:
        config = OmegaConf.load(config_root / config["include"])
    OmegaConf.resolve(config)
    return OmegaConf.to_container(config, resolve=True)


def target_trace_users(users: List[UserData], records_path: Path, users_per_run: int) -> Tuple[List[UserData], List[UserData], set[str]]:
    finished_ids = {path.stem for path in records_path.glob("*.json")}
    target_user_count = min(users_per_run, len(users))
    target_sample = users[:target_user_count]
    target_users = [user for user in target_sample if user.user_id not in finished_ids]
    finished_target_ids = {user.user_id for user in target_sample if user.user_id in finished_ids}
    return target_sample, target_users, finished_target_ids


def select_users_by_id_file(users: List[UserData], user_ids_path: Path) -> List[UserData]:
    user_ids = [
        line.strip()
        for line in user_ids_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not user_ids or len(user_ids) != len(set(user_ids)):
        raise ValueError("User ID file must contain a nonempty list of unique IDs")
    users_by_id = {user.user_id: user for user in users}
    missing = [user_id for user_id in user_ids if user_id not in users_by_id]
    if missing:
        raise ValueError(f"Missing users from {user_ids_path}: {', '.join(missing)}")
    return [users_by_id[user_id] for user_id in user_ids]


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


def model_config(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    return config.get(key, config["eval_model"])


def model_name(config: Dict[str, Any]) -> str:
    return config.get("model", "unknown")


def model_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in name.lower()).strip("-")


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
        close_embedding_clients()


def evaluate_one_user(
    record_file: Path,
    user_data: UserData,
    config: Dict[str, Any],
    embed_cfg: EmbedConfig,
    prompts: PromptSet,
    prediction_only: bool = False,
) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    eval_model_config = config["eval_model"]
    prediction_model_config = model_config(config, "prediction_model")
    eval_cfg = GenerationConfig(**eval_model_config)
    prediction_cfg = GenerationConfig(**prediction_model_config)
    eval_model = None
    prediction_model = None
    try:
        if not prediction_only or prediction_model_config == eval_model_config:
            eval_model = load_model(backend=eval_model_config["backend"], default_cfg=eval_cfg)
        prediction_model = eval_model
        if prediction_model_config != eval_model_config:
            prediction_model = load_model(backend=prediction_model_config["backend"], default_cfg=prediction_cfg)
        record = _load_record(record_file)
        metrics = evaluate_user_record(
            user_data=user_data,
            record=record,
            eval_model=eval_model,
            eval_cfg=eval_cfg,
            embed_cfg=embed_cfg,
            prompts=prompts,
            prediction_model=prediction_model,
            prediction_cfg=prediction_cfg,
            eval_model_name=None if prediction_only else model_name(eval_model_config),
            prediction_model_name=model_name(prediction_model_config),
            prediction_only=prediction_only,
        )
        eval_report = eval_model.provider_report() if isinstance(eval_model, OpenRouterModel) else None
        prediction_report = (
            prediction_model.provider_report()
            if prediction_model is not eval_model and isinstance(prediction_model, OpenRouterModel)
            else None
        )
        return metrics.get("user", user_data.user_id), metrics, eval_report, prediction_report
    finally:
        if prediction_model is not eval_model:
            close = getattr(prediction_model, "close", None)
            if close:
                close()
        if eval_model is not None:
            close = getattr(eval_model, "close", None)
            if close:
                close()
        close_embedding_clients()


def main(argv=None) -> None:
    parser = ArgumentParser(description="Run preference tracing and eval with per-user parallelism.")
    parser.add_argument("--config", type=str, default="run/main.yaml", help="Path to the main config file")
    parser.add_argument("--config-root", type=str, default=str(Path(__file__).resolve().parent / "config"), help="Root directory for config files")
    parser.add_argument("--result", type=str, default=None, help="Path to save the results; defaults to run name")
    parser.add_argument("--result-root", type=str, default="result", help="Root directory for results")
    parser.add_argument("--trace-workers", type=int, default=4, help="Number of users to trace concurrently")
    parser.add_argument("--eval-workers", type=int, default=4, help="Number of user records to evaluate concurrently")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--trace-only", action="store_true", help="Trace users without offline evaluation")
    modes.add_argument("--eval-only", action="store_true", help="Skip tracing and evaluate existing records only")
    parser.add_argument("--prediction-only", action="store_true", help="Only run offline preference prediction metrics; skip response/profile evaluation")
    parser.add_argument("--prediction-model-from-main", action="store_true", help="Use main_model as the offline preference prediction model")
    parser.add_argument("--metrics-name", type=str, default=None, help="Metrics directory name under the run path; useful for prediction-only comparisons")
    parser.add_argument("--user-ids-file", type=str, default=None, help="Optional newline-delimited user id file to run/evaluate a fixed cohort")
    parser.add_argument("--seed", type=int, help="Override dataset sampling seed")
    parser.add_argument("--n-users", type=int, help="Override sampled user count")
    parser.add_argument("--users-per-run", type=int, help="Override run cohort size")
    parser.add_argument("--use-batch", action="store_true", help="Trace with the OpenAI Batch API")
    parser.add_argument("--batch-workers", type=int, default=16, help="Concurrent batch trace workers")
    args = parser.parse_args(argv)
    if args.trace_only and (args.prediction_only or args.prediction_model_from_main or args.metrics_name):
        parser.error("--trace-only cannot be combined with evaluation options")
    if args.eval_only and args.use_batch:
        parser.error("--use-batch cannot be combined with --eval-only")
    for key in ("trace_workers", "eval_workers", "batch_workers", "n_users", "users_per_run"):
        value = getattr(args, key)
        if value is not None and value <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")

    config_root = Path(args.config_root)
    config = load_run_config(config_root, args.config)
    for key in ("seed", "n_users", "users_per_run"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    for key in ("n_users", "users_per_run"):
        if not isinstance(config.get(key), int) or isinstance(config[key], bool) or config[key] <= 0:
            parser.error(f"config {key} must be a positive integer")
    if args.use_batch and config["main_model"]["backend"] != "openai":
        parser.error("--use-batch requires an OpenAI main_model configuration")
    if args.prediction_model_from_main:
        config["prediction_model"] = config["main_model"]
    tracer_cfg = TracerConfig(**config["tracer"])
    tracer_cfg.override = OverrideConfig(**config.get("override", {}))
    embed_cfg = EmbedConfig(**config["embed"])
    prompt_adapter_name = config.get("prompt_adapter", config["dataset"])
    prompts = load_prompt_adapter(prompt_adapter_name)
    print(f"Loaded prompt adapter: {prompt_adapter_name}")
    for key in ("main_model", "eval_model", "prediction_model"):
        if key in config:
            generation = GenerationConfig(**config[key])
            if generation.backend not in ("openai", "openrouter"):
                parser.error(f"Unsupported {key} backend: {generation.backend}")
            if generation.max_retries < 1:
                parser.error(f"{key} max_retries must be positive")
    if embed_cfg.backend not in ("openai", "openrouter", "gemini", "transformer"):
        parser.error(f"Unsupported embedding backend: {embed_cfg.backend}")
    if embed_cfg.max_retries < 1 or embed_cfg.dim < 1:
        parser.error("Embedding retries and dimension must be positive")
    print(f"Loaded config: {json.dumps(config, indent=4)}")

    run_path = Path(args.result_root) / (args.result if args.result else config["name"])
    records_path = run_path / "records"
    if args.eval_only and not records_path.is_dir():
        parser.error(f"No records directory at {records_path}; run tracing first")
    records_path.mkdir(parents=True, exist_ok=True)
    print(f"Records will be saved to: {records_path}")

    users = load_data(config["dataset"], n_users=config["n_users"], seed=config["seed"])
    if args.user_ids_file:
        users = select_users_by_id_file(users, Path(args.user_ids_file))
    users = users[:config["users_per_run"]]
    if not users:
        parser.error("The selected cohort is empty")
    users_by_id = {user.user_id: user for user in users}
    print(f"Selected {len(users)} users from dataset {config['dataset']}")

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

        if args.use_batch:
            from core.batch_run import trace_batch
            records_by_user = trace_batch(target_users, config, tracer_cfg, embed_cfg, prompts, args.batch_workers)
            for user_id, records in records_by_user.items():
                write_json(records_path / f"{user_id}.json", records)
        else:
            trace_reports: List[Dict[str, Any]] = []
            active_progress: Dict[str, Tuple[int, int]] = {}
            progress_lock = Lock()
            with ThreadPoolExecutor(max_workers=args.trace_workers) as executor:
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

    if args.trace_only:
        return
    record_files = [records_path / f"{user.user_id}.json" for user in users]
    missing = [path.stem for path in record_files if not path.is_file()]
    if missing:
        parser.error(f"Missing records for selected users: {', '.join(missing)}; trace this cohort first")
    metrics_users_path = metrics_path / "users"
    metrics_users_path.mkdir(parents=True, exist_ok=True)
    user_metrics: List[Dict[str, Any]] = []
    eval_jobs = []
    for record_file in record_files:
        record = _load_record(record_file)
        user_id = record.get("user", record_file.stem)
        if user_id != record_file.stem:
            raise ValueError(f"Record user {user_id!r} does not match {record_file}")
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
    with ThreadPoolExecutor(max_workers=args.eval_workers) as executor:
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
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Evaluating records", unit="user")
        for future in pbar:
            user_id = futures[future]
            pbar.set_postfix(user=user_id)
            completed_user_id, metrics, eval_report, prediction_report = future.result()
            write_json(metrics_users_path / f"{completed_user_id}.json", metrics)
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
    write_json(metrics_path / "summary.json", summary)
    print(f"Evaluated {summary['n_users']} users and {summary['n_turns']} turns")


if __name__ == "__main__":
    main()
