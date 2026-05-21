import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.preference_tracer import PreferenceTracer
from core.utils import OverrideConfig, TracerConfig
from data.base import Conversation, UserData
from data.personamem_v2 import load_personamem_v2
from model import EmbedConfig, GenerationConfig, load_model
from prompt import load_prompt_adapter


BOUNDARY_MARKERS = {
    "updates": "Updated preferences:",
    "memory_boundaries": "Memory boundaries / do-not-remember:",
    "sensitive_constraints": "Sensitive/private constraints:",
    "non_user_preferences": "Not the user's own preferences:",
}


def load_config(config_root: Path, config_path: str) -> Dict[str, Any]:
    OmegaConf.register_new_resolver("include", lambda path: OmegaConf.load(config_root / path), replace=True)
    config = OmegaConf.load(config_root / config_path)
    OmegaConf.resolve(config)
    return OmegaConf.to_container(config, resolve=True)


def truncate_user(user: UserData, max_turns: int) -> UserData:
    turns = user.conversations[0].turns[:max_turns]
    return UserData(
        user_id=user.user_id,
        conversations=[Conversation(conversation_id=user.conversations[0].conversation_id, turns=turns)],
        gt_profile=user.gt_profile,
    )


def collect_hypothesis_text(record: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for turn in record.get("turns", []):
        hypotheses = turn.get("hypotheses", [])
        if isinstance(hypotheses, list):
            for item in hypotheses:
                if isinstance(item, dict):
                    chunks.append(str(item.get("content", "")))
        elif isinstance(hypotheses, dict):
            chunks.append(json.dumps(hypotheses, ensure_ascii=False))
        if turn.get("summary"):
            chunks.append(str(turn["summary"]))
    if record.get("final_profile"):
        chunks.append(str(record["final_profile"]))
    return "\n".join(chunks)


def summarize_checks(user: UserData, record: Dict[str, Any]) -> Dict[str, Any]:
    hypothesis_text = collect_hypothesis_text(record).lower()
    alignment_terms = [
        "prefer",
        "preference",
        "needs",
        "wants",
        "values",
        "constraint",
        "boundary",
        "sensitive",
        "remember",
        "style",
    ]
    present_cases = {
        name: marker in user.gt_profile
        for name, marker in BOUNDARY_MARKERS.items()
    }
    return {
        "hypotheses_or_profile_contain_alignment_terms": any(term in hypothesis_text for term in alignment_terms),
        "present_gt_case_sections": present_cases,
        "case_notes": {
            name: "present in sampled user's eval-only gt_profile; inspect generated hypotheses/profile for correct handling"
            if present
            else "not present in sampled user's gt_profile"
            for name, present in present_cases.items()
        },
        "final_profile_present": bool(record.get("final_profile")),
        "turns_traced": len(record.get("turns", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small real-model PersonaMem-v2 tracing smoke test.")
    parser.add_argument("--config", type=str, default="run/main.yaml", help="Config path under --config-root.")
    parser.add_argument("--config-root", type=Path, default=Path("config"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-users", type=int, default=1)
    parser.add_argument("--user-index", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--split", type=str, default="benchmark_text")
    parser.add_argument("--prompt-adapter", type=str, default="personamem_v2")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config_root, args.config)
    config["dataset"] = "personamem_v2"
    config["prompt_adapter"] = args.prompt_adapter
    config["n_users"] = args.n_users
    config["seed"] = args.seed

    users = load_personamem_v2(n_users=args.n_users, seed=args.seed, split=args.split)
    if not users:
        raise ValueError("No PersonaMem-v2 users loaded.")
    user = truncate_user(users[args.user_index], max_turns=args.max_turns)

    tracer_cfg = TracerConfig(**config["tracer"])
    tracer_cfg.override = OverrideConfig(**config.get("override", {}))
    embed_cfg = EmbedConfig(**config["embed"])
    gen_cfg = GenerationConfig(**config["main_model"])
    model = load_model(backend=config["main_model"]["backend"], default_cfg=gen_cfg)
    prompts = load_prompt_adapter(args.prompt_adapter)

    try:
        tracer = PreferenceTracer(
            model=model,
            generation_cfg=gen_cfg,
            tracer_cfg=tracer_cfg,
            embed_cfg=embed_cfg,
            prompts=prompts,
        )
        record = tracer.trace(user)
    finally:
        close = getattr(model, "close", None)
        if close:
            close()

    report = {
        "dataset": "personamem_v2",
        "split": args.split,
        "adapter": args.prompt_adapter,
        "user_id": user.user_id,
        "turn_ids": [turn.turn_id for turn in user.conversations[0].turns],
        "checks": summarize_checks(user, record),
        "gt_profile_preview": user.gt_profile[:1600],
        "record": record,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote smoke report to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
