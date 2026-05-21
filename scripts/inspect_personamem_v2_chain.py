import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.preprocess import compact_text
from data.personamem_v2 import load_personamem_v2
from prompt import load_prompt_adapter


TRACE_PROMPT_NAMES = [
    "skip",
    "preprocessing",
    "initialization",
    "likelihood",
    "branching",
    "axis",
    "merge",
    "perturb",
    "consolidate",
    "summary",
    "profile",
    "response",
    "prediction",
    "response_evaluation",
]

GT_METADATA_FIELD_TOKENS = [
    "expanded_persona",
    "pref_type",
    "prev_pref",
    "sensitive_info",
    "gt_profile",
]


def preview(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def candidate_rows(turn) -> List[Dict[str, Any]]:
    rows = []
    for i, candidate in enumerate(turn.candidates):
        rows.append(
            {
                "i": i,
                "summary": compact_text(candidate, head=120, tail=80),
                "choice": "chosen" if i == turn.chosen_idx else "rejected",
                "content": compact_text(candidate),
            }
        )
    return rows


def render_prompts(turn, gt_profile: str, adapter: str) -> Dict[str, str]:
    prompts = load_prompt_adapter(adapter)
    rows = candidate_rows(turn)
    candidates_with_choice = "[CandidateSet]\n" + json.dumps(rows, ensure_ascii=False, indent=2)
    candidates_without_choice = "[CandidateSet]\n" + json.dumps(
        [{"i": item["i"], "summary": item["summary"], "content": item["content"]} for item in rows],
        ensure_ascii=False,
        indent=2,
    )
    current_turn_text = turn.format(include_candidates=True, include_choice=False)
    kwargs = {
        "user_message": turn.user_message,
        "candidates": "\n".join([f"[{i}] {c}" for i, c in enumerate(turn.candidates)]),
        "n": len(turn.candidates),
        "n_hypotheses": 3,
        "prev_turns": "",
        "retrieved_hypotheses": "",
        "hypothesis": "The user prefers practical, user-specific personalization.",
        "current_hypothesis": "h1: The user prefers concise, practical help.",
        "hypotheses": "h1: The user prefers concise, practical help.",
        "collapsed_cluster": "The user prefers concise, practical help.",
        "conversation_history": turn.format(include_candidates=False),
        "global_axes_summary": "communication style, user-specific constraints",
        "K": 2,
        "consolidated_hypotheses": "The user tends to prefer practical help.",
        "current_hypotheses": "0.7 The user prefers practical help.\n0.3 The user prefers concise answers.",
        "profile": "The user prefers practical, concise responses.",
        "l": 20,
        "r": 80,
        "current_message": turn.user_message,
        "current_turn": current_turn_text,
        "c": len(turn.candidates),
        "adapted": "Here is a concise, practical answer tailored to the request.",
        "survey": gt_profile,
    }
    rendered = {}
    for name in TRACE_PROMPT_NAMES + ["profile_evaluation"]:
        prompt = getattr(prompts, name)
        if name in {"initialization", "branching", "perturb"}:
            local_kwargs = dict(kwargs, candidates=candidates_with_choice)
        elif name == "likelihood":
            local_kwargs = dict(kwargs, candidates=candidates_without_choice)
        else:
            local_kwargs = kwargs
        rendered[name] = prompt.format(**local_kwargs)
    return rendered


def build_report(args) -> Dict[str, Any]:
    users = load_personamem_v2(n_users=args.n_users, seed=args.seed, split=args.split)
    if not users:
        raise ValueError("No PersonaMem-v2 users loaded.")
    user = users[args.user_index]
    turns = user.conversations[0].turns
    turn = turns[args.turn_index]
    prompts = render_prompts(turn, user.gt_profile, args.adapter)
    tracing_prompts = {name: prompts[name] for name in TRACE_PROMPT_NAMES}
    leaked_tokens = sorted(
        token
        for token in GT_METADATA_FIELD_TOKENS
        if any(token in prompt for prompt in tracing_prompts.values())
    )
    return {
        "dataset": "personamem_v2",
        "split": args.split,
        "adapter": args.adapter,
        "user_id": user.user_id,
        "turn_id": turn.turn_id,
        "online_tracing_evidence": {
            "user_message": turn.user_message,
            "chosen_idx": turn.chosen_idx,
            "candidates": [
                {
                    "i": i,
                    "choice": "chosen" if i == turn.chosen_idx else "rejected",
                    "content_preview": preview(candidate, args.preview_chars),
                }
                for i, candidate in enumerate(turn.candidates)
            ],
            "visible_to_tracing": [
                "user_message",
                "candidate content",
                "choice: chosen | rejected for update prompts",
            ],
        },
        "offline_eval_only_ground_truth": {
            "gt_profile_preview": preview(user.gt_profile, args.gt_preview_chars),
            "not_visible_to_tracing": [
                "PersonaMem-v2 benchmark metadata",
                "derived gt_profile",
            ],
        },
        "checks": {
            "tracing_prompts_include_choice_signal": all(
                "choice: chosen | rejected" in prompts[name]
                for name in ("initialization", "branching")
            ),
            "gt_metadata_tokens_in_tracing_prompts": leaked_tokens,
        },
        "rendered_prompts": prompts if args.include_prompts else {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PersonaMem-v2 data and rendered tracing prompts.")
    parser.add_argument("--n-users", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="benchmark_text")
    parser.add_argument("--adapter", type=str, default="personamem_v2")
    parser.add_argument("--user-index", type=int, default=0)
    parser.add_argument("--turn-index", type=int, default=0)
    parser.add_argument("--preview-chars", type=int, default=300)
    parser.add_argument("--gt-preview-chars", type=int, default=1200)
    parser.add_argument("--include-prompts", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = build_report(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote inspection report to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
