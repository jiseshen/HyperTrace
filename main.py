from core import PreferenceTracer
from data import load_data
from argparse import ArgumentParser
import yaml


def main():
    parser = ArgumentParser(description="Run preference tracing on a dataset")
    parser.add_argument("--dataset", type=str, default="prism", help="Dataset name (default: prism)")
    parser.add_argument("--n_users", type=int, default=None, help="Number of users to process (default: all)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file (default: config.yaml)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    tracer_config = config.get("tracer_config", {})
    evaluation_config = config.get("evaluation_config", {})
    preference_tracer = PreferenceTracer(tracer_cfg=tracer_config, evaluation_cfg=evaluation_config)

    user_data_list = load_data(args.dataset, n_users=args.n_users)
    
    all_records = []
    for user_data in user_data_list:
        records = preference_tracer.trace(user_data)
        all_records.append(records)
    
    # Save or process all_records as needed
    print(all_records)