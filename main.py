from core import PreferenceTracer
from data import load_data
from argparse import ArgumentParser
from omegaconf import OmegaConf
from pathlib import Path
from glob import glob
from tqdm import tqdm
from core import TracerConfig, EmbedConfig
from model import load_model, GenerationConfig
import json

def main():
    parser = ArgumentParser(description="Run preference tracing on a configured dataset")
    parser.add_argument("--config", type=str, default="run/main.yaml", help="Path to the main config file")
    parser.add_argument("--config-root", type=str, default="config", help="Root directory for config files, used for resolving relative paths in the main config")
    parser.add_argument("--result", type=str | None, default=None, help="Path to save the results, if not provided use run name with seed")
    parser.add_argument("--result-root", type=str, default="result", help="Root directory for results")
    args = parser.parse_args()
    
    config_root = Path(args.config_root)
    OmegaConf.register_new_resolver("include", lambda path: OmegaConf.load(config_root / path))
    config = OmegaConf.load(config_root / args.config)
    OmegaConf.resolve(config)
    print(f"Loaded config: {config}")
    
    result_root = Path(args.result_root)
    result_path = result_root / (args.result if args.result else f"{config['name']}_{config['seed']}")
    result_path.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {result_path}")
    
    user_data = load_data(config['dataset'], n_users=config['n_users'], seed=config['seed'])
    print(f"Loaded {len(user_data)} users from dataset {config['dataset']}")

    tracer_cfg = TracerConfig(**config["tracer"])
    gen_cfg = GenerationConfig(**config["gen_model"])
    eval_cfg = GenerationConfig(**config["eval_model"])
    embed_cfg = EmbedConfig(**config["embed"])
    
    gen_model = load_model(backend=config["gen_model"]["backend"], default_cfg=gen_cfg)
    eval_model = load_model(backend=config["eval_model"]["backend"], default_cfg=eval_cfg)
    preference_tracer = PreferenceTracer(
        model=gen_model,
        generation_cfg=gen_cfg,
        tracer_cfg=tracer_cfg,
        embed_cfg=embed_cfg,
        evaluation_model=eval_model,
        evaluation_cfg=eval_cfg
    )

    finished_ids = set(Path(f).stem for f in glob(result_path / "*.json"))
    if finished_ids:
        target_users = [ud for ud in user_data if ud.user_id not in finished_ids][:config['users_per_run']]
        print(f"Skipping {len(finished_ids)} finished users")
    print(f"Running preference tracing for {len(target_users)} users")
    for user in tqdm(target_users, desc="Tracing preferences", unit="user"):
        records = preference_tracer.trace(user)
        with open(result_path / f"{user.user_id}.json", "w") as f:
            json.dump(records, f, indent=4)