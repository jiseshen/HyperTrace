import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MAX_SKIP_RATIO = 0.75
SMOOTH_WIN = 5
MIN_USERS_PER_TURN = 10
AFTER_TURN = 20

ONLINE_METRICS = [
    ("prediction_accuracy", "Prediction Accuracy", None),
    ("adapt_relative_gpt_score", "Relative GPT Score", None),
    ("adapt_relative_score", "Relative Embedding Score", None),
]
PROFILE_KEYS = [
    "survey_consistency",
    "key_aspect_match",
    "internal_plausibility",
    "overall",
    "similarity",
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


def skip_ratio(record: Dict[str, Any]) -> float:
    turns = record.get("turns", [])
    if not turns:
        return 0.0
    return sum(1 for turn in turns if turn.get("preprocess", {}).get("skip", False)) / len(turns)


def gpt5_metric_value(turn: Dict[str, Any], metric: str) -> Optional[float]:
    if metric == "prediction_accuracy":
        pred = turn.get("prediction") or {}
        if pred.get("success") is False:
            return None
        return safe_float(pred.get("accuracy"))
    adap = turn.get("adaptation") or {}
    if adap.get("success") is False:
        return None
    return safe_float(adap.get(metric.replace("adapt_", "", 1)))


def baseline_metric_value(turn: Dict[str, Any], metric: str) -> Optional[float]:
    if metric == "prediction_accuracy":
        pred = turn.get("prediction") or {}
        if pred:
            if pred.get("success") is False:
                return None
            return safe_float(pred.get("accuracy"))
        return safe_float(turn.get("accuracy"))

    adap = turn.get("adaptation") or {}
    if adap:
        if adap.get("success") is False:
            return None
        return safe_float(adap.get(metric.replace("adapt_", "", 1)))

    legacy = {
        "adapt_relative_gpt_score": "relative_gpt_score",
        "adapt_relative_score": "relative_similarity_score",
    }
    return safe_float(turn.get(legacy[metric]))


def user_id(user: Dict[str, Any], fallback: Optional[str] = None) -> Optional[str]:
    return user.get("user") or user.get("user_id") or fallback


def load_gpt5_users(result_root: Path) -> List[Dict[str, Any]]:
    records_dir = result_root / "records"
    metrics_dir = result_root / "metrics" / "users"
    users: List[Dict[str, Any]] = []
    for metric_path in sorted(metrics_dir.glob("*.json")):
        record_path = records_dir / metric_path.name
        if record_path.exists() and skip_ratio(load_json(record_path)) > MAX_SKIP_RATIO:
            continue
        user = load_json(metric_path)
        user.setdefault("user", metric_path.stem)
        users.append(user)
    return users


def load_baseline_users(result_root: Path) -> Dict[str, Dict[str, Any]]:
    users_dir = result_root / "users"
    if users_dir.exists():
        users = {}
        for path in sorted(users_dir.glob("*.json")):
            user = load_json(path)
            uid = user_id(user, path.stem)
            if uid is not None:
                user.setdefault("user", uid)
                users[uid] = user
        return users
    data = load_json(result_root)
    if not isinstance(data, list):
        raise ValueError(f"Expected list baseline JSON at {result_root}")
    users = {}
    for idx, user in enumerate(data):
        uid = user_id(user)
        if uid is None:
            uid = str(idx)
        user.setdefault("user", uid)
        users[uid] = user
    return users


def align_users(
    users_by_id: Dict[str, Dict[str, Any]],
    reference_user_ids: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    users = []
    missing = []
    for uid in reference_user_ids:
        user = users_by_id.get(uid)
        if user is None:
            missing.append(uid)
        else:
            users.append(user)
    return users, missing


def turn_list(user: Dict[str, Any]) -> List[Dict[str, Any]]:
    return user.get("turns") or user.get("turn_results") or []


def build_turn_series(
    users: List[Dict[str, Any]],
    metric: str,
    getter,
    min_users_per_turn: int = MIN_USERS_PER_TURN,
) -> Dict[str, List[float]]:
    max_turns = max((len(turn_list(user)) for user in users), default=0)
    xs: List[int] = []
    ys: List[float] = []
    counts: List[int] = []
    for turn_idx in range(max_turns):
        vals = []
        for user in users:
            turns = turn_list(user)
            if turn_idx >= len(turns):
                continue
            value = getter(turns[turn_idx], metric)
            if value is not None:
                vals.append(value)
        if len(vals) < min_users_per_turn:
            continue
        avg = mean(vals)
        if avg is None:
            continue
        xs.append(turn_idx)
        ys.append(avg)
        counts.append(len(vals))
    return {"x": xs, "y": ys, "counts": counts}


def per_user_after20_mean(users: List[Dict[str, Any]], metric: str, getter) -> Optional[float]:
    per_user = []
    for user in users:
        vals = [
            getter(turn, metric)
            for idx, turn in enumerate(turn_list(user))
            if idx >= AFTER_TURN
        ]
        vals = [v for v in vals if v is not None]
        if vals:
            per_user.append(sum(vals) / len(vals))
    return mean(per_user)


def per_user_delta_from_first(users: List[Dict[str, Any]], metric: str, getter) -> Optional[float]:
    deltas = []
    for user in users:
        turns = turn_list(user)
        if not turns:
            continue
        first = getter(turns[0], metric)
        if first is None:
            continue
        later_vals = [
            getter(turn, metric)
            for idx, turn in enumerate(turns)
            if idx >= AFTER_TURN
        ]
        later_vals = [v for v in later_vals if v is not None]
        if later_vals:
            deltas.append(sum(later_vals) / len(later_vals) - first)
    return mean(deltas)


def profile_averages(users: List[Dict[str, Any]]) -> Dict[str, float]:
    buckets = {key: [] for key in PROFILE_KEYS}
    for user in users:
        profile = user.get("profile_alignment") or user.get("profile_metrics") or {}
        summary = user.get("summary") or {}
        for key in PROFILE_KEYS:
            value = safe_float(profile.get(key))
            if value is None:
                value = safe_float(summary.get(f"profile_{key}"))
            if value is not None:
                buckets[key].append(value)
    return {key: float(np.mean(vals)) if vals else math.nan for key, vals in buckets.items()}


def plot_all(
    runs: List[Tuple[str, List[Dict[str, Any]], Any]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    axes_list = list(axes.flatten())
    colors = plt.get_cmap("tab10")

    for ax, (metric, title, ylim) in zip(axes_list[:3], ONLINE_METRICS):
        for idx, (name, users, getter) in enumerate(runs):
            series = build_turn_series(users, metric, getter)
            if not series["x"]:
                continue
            ax.plot(
                series["x"],
                smooth(series["y"]),
                marker="o",
                markersize=3,
                linewidth=2.2,
                color=colors(idx),
                label=name,
            )
        if "Relative" in title:
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.set_xlabel("Turn Index")
        ax.set_ylabel("Smoothed Mean")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    ax = axes_list[3]
    x = np.arange(len(PROFILE_KEYS))
    width = 0.8 / len(runs)
    for idx, (name, users, _) in enumerate(runs):
        prof = profile_averages(users)
        values = [prof[key] for key in PROFILE_KEYS]
        ax.bar(x + (idx - (len(runs) - 1) / 2) * width, values, width=width, label=name, color=colors(idx))
    ax.set_title("Profile Alignment by Dimension")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(["survey", "aspect", "plaus.", "overall", "sim."], rotation=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("GPT-5 Trace vs Baselines (GPT-5-filtered matched users)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def format_num(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def write_table(runs: List[Tuple[str, List[Dict[str, Any]], Any]], output: Path) -> None:
    rows: List[Dict[str, str]] = []
    for name, users, getter in runs:
        prof = profile_averages(users)
        row = {"run": name, "n_users": str(len(users))}
        for metric, _, _ in ONLINE_METRICS:
            short = metric.replace("prediction_", "pred_").replace("adapt_", "")
            row[f"{short}_after20"] = format_num(per_user_after20_mean(users, metric, getter))
            row[f"{short}_delta_after20_vs_turn0"] = format_num(per_user_delta_from_first(users, metric, getter))
        for key in PROFILE_KEYS:
            row[f"profile_{key}"] = format_num(prof[key])
        rows.append(row)

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

    print(f"Saved table: {output}")
    print(f"Saved markdown table: {md_path}")
    print(md_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporary GPT-5 trace vs baseline comparison.")
    parser.add_argument("--gpt5-result", type=Path, default=Path("result/openrouter-gpt5-trace-gemini3-flash-eval"))
    parser.add_argument("--baseline-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("result/plots/temp_gpt5_vs_baselines.png"))
    parser.add_argument("--table-output", type=Path, default=Path("result/plots/temp_gpt5_vs_baselines_table.csv"))
    args = parser.parse_args()

    baseline_root = args.baseline_root
    if baseline_root is None:
        baseline_root = Path("baseline_results") if Path("baseline_results").exists() else Path("baseline_results")

    gpt5_users = load_gpt5_users(args.gpt5_result)
    reference_user_ids = [uid for user in gpt5_users if (uid := user_id(user)) is not None]

    runs: List[Tuple[str, List[Dict[str, Any]], Any]] = [
        ("GPT-5 trace", gpt5_users, gpt5_metric_value),
    ]
    print(f"Reference GPT-5 users after skip filter: {len(reference_user_ids)}")
    for dirname, label in [
        ("cot_gpt5_openrouter", "CoT"),
        ("rag_gpt5_openrouter", "RAG"),
        ("cheatsheet_gpt5_openrouter", "Cheatsheet"),
        ("hydra_reranker_prism", "Hydra"),
        ("hypogenic_gpt5_openrouter", "HyperAlign")
    ]:
        path = baseline_root / dirname
        if path.exists():
            baseline_users, missing = align_users(load_baseline_users(path), reference_user_ids)
            print(
                f"{label}: matched {len(baseline_users)}/{len(reference_user_ids)} "
                f"reference users"
                + (f"; missing examples: {', '.join(missing[:5])}" if missing else "")
            )
            runs.append((label, baseline_users, baseline_metric_value))

    plot_all(runs, args.output)
    write_table(runs, args.table_output)
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
