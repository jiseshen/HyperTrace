import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AFTER_TURN = 20
MIN_USERS_PER_TURN = 10
SMOOTH_WIN = 5

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

RUNS = [
    ("GPT-5", Path("result/openrouter-gpt5-trace-gemini3-flash-eval")),
    ("Hybrid", Path("result/openrouter-hybrid-trace-gemini3-flash-eval")),
    ("No-skip", Path("result/openrouter-hybrid-trace-prism-ablate-no-skip-gemini3-flash-eval")),
    ("Flat5", Path("result/openrouter-hybrid-trace-prism-ablate-flat5-gemini3-flash-eval")),
    ("No-topic", Path("result/openrouter-hybrid-trace-prism-ablate-no-topic-gemini3-flash-eval")),
    ("Retrieve-replace", Path("result/openrouter-hybrid-trace-prism-ablate-retrieve-replace-gemini3-flash-eval")),
]

ONLINE_METRICS = [
    ("prediction_accuracy", "Prediction Accuracy"),
    ("adapt_relative_gpt_score", "Relative GPT Score"),
    ("adapt_relative_score", "Relative Embedding Score"),
]

def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def smooth(values: List[float], win: int = SMOOTH_WIN) -> List[float]:
    if win <= 1 or len(values) < win:
        return list(values)
    arr = np.array(values, dtype=float)
    kernel = np.ones(win) / win
    padded = np.pad(arr, win // 2, mode="edge")
    return list(np.convolve(padded, kernel, mode="valid")[: len(values)])


def turn_list(user: Dict[str, Any]) -> List[Dict[str, Any]]:
    return user.get("turns") or user.get("turn_results") or []


def load_aligned_users(result_root: Path, user_ids: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    users_dir = result_root / "metrics" / "users"
    users: List[Dict[str, Any]] = []
    missing: List[str] = []
    for uid in user_ids:
        path = users_dir / f"{uid}.json"
        if not path.exists():
            missing.append(uid)
            continue
        user = load_json(path)
        user.setdefault("user", uid)
        users.append(user)
    return users, missing


def metric_value(turn: Dict[str, Any], metric: str) -> Optional[float]:
    if metric == "prediction_accuracy":
        prediction = turn.get("prediction") or {}
        if prediction.get("success") is False:
            return None
        return safe_float(prediction.get("accuracy"))
    adaptation = turn.get("adaptation") or {}
    if adaptation.get("success") is False:
        return None
    return safe_float(adaptation.get(metric.replace("adapt_", "", 1)))


def build_turn_series(
    users: List[Dict[str, Any]],
    metric: str,
    min_users_per_turn: int,
) -> Dict[str, List[float]]:
    max_turns = max((len(turn_list(user)) for user in users), default=0)
    xs: List[int] = []
    ys: List[float] = []
    counts: List[int] = []
    for turn_idx in range(max_turns):
        vals = [
            value
            for user in users
            if turn_idx < len(turn_list(user))
            for value in [metric_value(turn_list(user)[turn_idx], metric)]
            if value is not None
        ]
        if len(vals) < min_users_per_turn:
            continue
        avg = mean(vals)
        if avg is None:
            continue
        xs.append(turn_idx)
        ys.append(avg)
        counts.append(len(vals))
    return {"x": xs, "y": ys, "counts": counts}


def per_user_after_turn_mean(users: List[Dict[str, Any]], metric: str, after_turn: int) -> Optional[float]:
    per_user = []
    for user in users:
        vals = [
            metric_value(turn, metric)
            for idx, turn in enumerate(turn_list(user))
            if idx >= after_turn
        ]
        vals = [v for v in vals if v is not None]
        if vals:
            per_user.append(sum(vals) / len(vals))
    return mean(per_user)


def profile_overall(users: List[Dict[str, Any]]) -> Optional[float]:
    vals = []
    for user in users:
        profile = user.get("profile_alignment") or {}
        summary = user.get("summary") or {}
        value = safe_float(profile.get("overall"))
        if value is None:
            value = safe_float(summary.get("profile_overall"))
        if value is not None:
            vals.append(value)
    return mean(vals)


def run_turn_count(result_root: Path, users: List[Dict[str, Any]]) -> int:
    return sum(len(turn_list(user)) for user in users)


def attempt_cost(attempt: Dict[str, Any]) -> Optional[float]:
    usage = attempt.get("usage") or {}
    return safe_float(usage.get("cost"))


def usage_per_turn(result_root: Path, n_turns: int) -> Tuple[Optional[float], Optional[float]]:
    report_path = result_root / "provider_report.json"
    if not report_path.exists() or n_turns <= 0:
        return None, None
    report = load_json(report_path)
    completion_tokens = 0.0
    costs = []
    for attempt in report.get("attempts", []):
        if not attempt.get("success"):
            continue
        usage = attempt.get("usage") or {}
        completion_tokens += safe_float(usage.get("completion_tokens")) or safe_float(usage.get("output_tokens")) or 0.0
        cost = attempt_cost(attempt)
        if cost is not None:
            costs.append(cost)
    dollar_total = sum(costs) if costs else None
    return completion_tokens / n_turns, (dollar_total / n_turns if dollar_total is not None else None)


def format_num(value: Optional[float], digits: int = 4) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def collect_runs(
    run_specs: Sequence[Tuple[str, Path]],
    user_ids: Sequence[str],
    after_turn: int,
) -> List[Dict[str, Any]]:
    runs = []
    for label, path in run_specs:
        users, missing = load_aligned_users(path, user_ids)
        n_turns = run_turn_count(path, users)
        completion_per_turn, dollar_per_turn = usage_per_turn(path, n_turns)
        runs.append({
            "label": label,
            "path": path,
            "users": users,
            "missing": missing,
            "n_turns": n_turns,
            "accuracy_after": per_user_after_turn_mean(users, "prediction_accuracy", after_turn),
            "relative_gpt_after": per_user_after_turn_mean(users, "adapt_relative_gpt_score", after_turn),
            "relative_embedding_after": per_user_after_turn_mean(users, "adapt_relative_score", after_turn),
            "profile_overall": profile_overall(users),
            "completion_tokens_per_turn": completion_per_turn,
            "dollar_per_turn": dollar_per_turn,
        })
    return runs


def write_table(runs: List[Dict[str, Any]], output: Path) -> None:
    rows = []
    for run in runs:
        rows.append({
            "run": run["label"],
            "n_users": str(len(run["users"])),
            "n_turns": str(run["n_turns"]),
            "accuracy_after20": format_num(run["accuracy_after"]),
            "relative_gpt_after20": format_num(run["relative_gpt_after"]),
            "relative_embedding_after20": format_num(run["relative_embedding_after"]),
            "profile_overall": format_num(run["profile_overall"]),
            "completion_tokens_per_turn": format_num(run["completion_tokens_per_turn"], digits=1),
            "dollar_per_turn": format_num(run["dollar_per_turn"], digits=5),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = output.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as f:
        headers = list(rows[0].keys())
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def plot_runs(runs: List[Dict[str, Any]], output: Path, min_users_per_turn: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    axes_list = list(axes.flatten())
    colors = plt.get_cmap("tab10")

    for ax, (metric, title) in zip(axes_list[:3], ONLINE_METRICS):
        for idx, run in enumerate(runs):
            series = build_turn_series(run["users"], metric, min_users_per_turn=min_users_per_turn)
            if not series["x"]:
                continue
            ax.plot(
                series["x"],
                smooth(series["y"]),
                marker="o",
                markersize=3,
                linewidth=2.2,
                color=colors(idx),
                label=run["label"],
            )
        if "Relative" in title:
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.axvline(AFTER_TURN, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_title(f"{title} (n>={min_users_per_turn}, smooth={SMOOTH_WIN})")
        ax.set_xlabel("Turn Index")
        ax.set_ylabel("Smoothed Mean")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    labels = [run["label"] for run in runs]
    x = np.arange(len(labels))

    ax = axes_list[3]
    ax.bar(x, [run["profile_overall"] or math.nan for run in runs], color=[colors(i) for i in range(len(runs))])
    ax.set_title("Overall Profile Score")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3)

    ax = axes_list[4]
    dollars = [run["dollar_per_turn"] for run in runs]
    ax.bar(x, [value if value is not None else 0 for value in dollars], color=[colors(i) for i in range(len(runs))])
    for i, value in enumerate(dollars):
        if value is None:
            ax.text(i, 0, "-", ha="center", va="bottom", fontsize=11)
    ax.set_title("Average Trace Dollar Usage per Turn")
    ax.set_ylabel("USD / turn")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3)

    ax = axes_list[5]
    ax.axis("off")
    table_rows = [
        [
            run["label"],
            format_num(run["accuracy_after"]),
            format_num(run["relative_gpt_after"]),
            format_num(run["relative_embedding_after"]),
            format_num(run["profile_overall"]),
            format_num(run["completion_tokens_per_turn"], digits=1),
            format_num(run["dollar_per_turn"], digits=5),
        ]
        for run in runs
    ]
    table = ax.table(
        cellText=table_rows,
        colLabels=["Run", "Acc@20+", "RelGPT@20+", "RelEmb@20+", "Profile", "CompTok/turn", "$/turn"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.45)
    ax.set_title("Summary Table")

    fig.suptitle("PRISM Ablation Comparison (fixed 28 GPT-5-filtered users)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot fixed-user PRISM ablation comparison.")
    parser.add_argument("--output", type=Path, default=Path("result/plots/prism_ablation_comparison.png"))
    parser.add_argument("--table-output", type=Path, default=Path("result/plots/prism_ablation_comparison_table.csv"))
    parser.add_argument("--after-turn", type=int, default=AFTER_TURN)
    parser.add_argument("--min-users-per-turn", type=int, default=MIN_USERS_PER_TURN)
    args = parser.parse_args()

    runs = collect_runs(RUNS, PRISM_ABLATION_USER_IDS, after_turn=args.after_turn)
    for run in runs:
        if run["missing"]:
            print(f"{run['label']}: matched {len(run['users'])}/{len(PRISM_ABLATION_USER_IDS)}; missing {run['missing'][:5]}")
        else:
            print(f"{run['label']}: matched {len(run['users'])}/{len(PRISM_ABLATION_USER_IDS)}")

    write_table(runs, args.table_output)
    plot_runs(runs, args.output, min_users_per_turn=args.min_users_per_turn)
    print(f"Saved table: {args.table_output}")
    print(f"Saved markdown table: {args.table_output.with_suffix('.md')}")
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
