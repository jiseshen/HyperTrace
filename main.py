from core.preference_tracer import PreferenceTracer
from core.batch_tracer import BatchPreferenceTracer
from data import load_data
from argparse import ArgumentParser
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
from core.utils import TracerConfig, OverrideConfig
from model import load_model, GenerationConfig, EmbedConfig
from model.batch_queue_model import BatchQueueModel
from model.openai_model import OpenAIModel
from eval.runner import evaluate_records
from prompt import load_prompt_adapter
import json


def main():
    parser = ArgumentParser(description="Run preference tracing on a configured dataset")
    parser.add_argument("--config", type=str, default="run/main.yaml", help="Path to the main config file")
    parser.add_argument("--config-root", type=str, default="config", help="Root directory for config files, used for resolving relative paths in the main config")
    parser.add_argument("--result", type=str, default=None, help="Path to save the results, if not provided use run name with seed")
    parser.add_argument("--result-root", type=str, default="result", help="Root directory for results")
    parser.add_argument("--use-batch", action="store_true", help="Use queueing model wrapper that batches requests via OpenAI Batch API")
    parser.add_argument("--batch-workers", type=int, default=16, help="Number of parallel user traces when --use-batch is enabled")
    parser.add_argument("--eval-only", action="store_true", help="Skip tracing and evaluate existing records only")
    args = parser.parse_args()
    
    config_root = Path(args.config_root)
    OmegaConf.register_new_resolver("include", lambda path: OmegaConf.load(config_root / path))
    config = OmegaConf.load(config_root / args.config)
    OmegaConf.resolve(config)
    config = OmegaConf.to_container(config, resolve=True)
    print(f"Loaded config: {json.dumps(config, indent=4)}")
    
    result_root = Path(args.result_root)
    run_path = result_root / (args.result if args.result else f"{config['name']}")
    result_path = run_path / "records"
    metrics_path = run_path / "metrics"
    result_path.mkdir(parents=True, exist_ok=True)
    print(f"Records will be saved to: {result_path}")
    
    user_data = load_data(config['dataset'], n_users=config['n_users'], seed=config['seed'])
    print(f"Loaded {len(user_data)} users from dataset {config['dataset']}")

    tracer_cfg = TracerConfig(**config["tracer"])
    tracer_cfg.override = OverrideConfig(**config.get("override", {}))
    embed_cfg = EmbedConfig(**config["embed"])
    prompt_adapter_name = config.get("prompt_adapter", config["dataset"])
    prompts = load_prompt_adapter(prompt_adapter_name)
    print(f"Loaded prompt adapter: {prompt_adapter_name}")

    if not args.eval_only:
        gen_cfg = GenerationConfig(**config["main_model"])
        gen_model = load_model(backend=config["main_model"]["backend"], default_cfg=gen_cfg)
        finished_ids = {p.stem for p in result_path.glob("*.json")}
        target_users = [ud for ud in user_data if ud.user_id not in finished_ids][:config['users_per_run']]
        if finished_ids:
            print(f"Skipping {len(finished_ids)} finished users")
        print(f"Running preference tracing for {len(target_users)} users")
        if args.use_batch:
            if isinstance(gen_model, BatchQueueModel):
                queue_model = gen_model
            elif isinstance(gen_model, OpenAIModel):
                queue_model = BatchQueueModel(
                    base_model=gen_model,
                    completion_window=gen_cfg.completion_window,
                    retry_delay=gen_cfg.retry_delay,
                )
            else:
                raise ValueError("--use-batch currently requires OpenAIModel or BatchQueueModel.")
            preference_tracer = BatchPreferenceTracer(
                model=queue_model,
                generation_cfg=gen_cfg,
                tracer_cfg=tracer_cfg,
                embed_cfg=embed_cfg,
                stage_workers=args.batch_workers,
                prompts=prompts,
            )
            try:
                records_by_user = preference_tracer.trace_users(target_users)
                for user in target_users:
                    records = records_by_user[user.user_id]
                    with open(result_path / f"{user.user_id}.json", "w") as f:
                        json.dump(records, f, indent=4)
            finally:
                queue_model.shutdown()
        else:
            preference_tracer = PreferenceTracer(
                model=gen_model,
                generation_cfg=gen_cfg,
                tracer_cfg=tracer_cfg,
                embed_cfg=embed_cfg,
                prompts=prompts,
            )
            pbar = tqdm(target_users, desc="Tracing preferences", unit="user")
            for user in pbar:
                pbar.set_postfix(user=user.user_id)
                records = preference_tracer.trace(user)
                with open(result_path / f"{user.user_id}.json", "w") as f:
                    json.dump(records, f, indent=4)

    eval_cfg = GenerationConfig(**config["eval_model"])
    eval_model = load_model(backend=config["eval_model"]["backend"], default_cfg=eval_cfg)
    print(f"Metrics will be saved to: {metrics_path}")
    summary = evaluate_records(
        users=user_data,
        records_path=result_path,
        metrics_path=metrics_path,
        eval_model=eval_model,
        eval_cfg=eval_cfg,
        embed_cfg=embed_cfg,
        prompts=prompts,
    )
    print(f"Evaluated {summary['n_users']} users and {summary['n_turns']} turns")


if __name__ == "__main__":
    main()
