import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LINE_METRICS = [
    ("prediction_accuracy", "Prediction Accuracy", None),
    ("adapt_relative_gpt_score", "Adapt Relative GPT Score", None),
    ("adapt_relative_score", "Adapt Relative Embedding Score", None),
]
PROFILE_KEYS = [
    "survey_consistency",
    "key_aspect_match",
    "internal_plausibility",
    "overall",
    "similarity",
]
AFTER_TURN = 20


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def centered_moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) < window:
        return list(values)
    kernel = np.ones(window) / window
    padded = np.pad(np.array(values), window // 2, mode="edge")
    return list(np.convolve(padded, kernel, mode="valid")[: len(values)])


def sorted_user_ids(result_dir: Path) -> List[str]:
    return sorted(path.stem for path in (result_dir / "metrics" / "users").glob("*.json"))


def skip_ratio(record: Dict[str, Any]) -> float:
    turns = record.get("turns", [])
    if not turns:
        return 0.0
    skipped = sum(1 for turn in turns if turn.get("preprocess", {}).get("skip", False))
    return skipped / len(turns)


def reference_skip_exclude_ids(result_dir: Path, max_skip_ratio: float) -> Dict[str, float]:
    excluded: Dict[str, float] = {}
    for path in sorted((result_dir / "records").glob("*.json")):
        ratio = skip_ratio(load_json(path))
        if ratio > max_skip_ratio:
            excluded[path.stem] = ratio
    return excluded


def load_users(result_dir: Path, user_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    users = {}
    metrics_dir = result_dir / "metrics" / "users"
    for user_id in user_ids:
        path = metrics_dir / f"{user_id}.json"
        if path.exists():
            users[user_id] = load_json(path)
    return users


def turn_metric(turn: Dict[str, Any], metric: str) -> Optional[float]:
    if metric == "prediction_accuracy":
        prediction = turn.get("prediction", {})
        if not prediction.get("success"):
            return None
        return safe_float(prediction.get("accuracy"))

    adaptation = turn.get("adaptation", {})
    if not adaptation.get("success", True):
        return None

    mapping = {
        "adapt_gpt_score": "gpt_score",
        "adapt_relative_gpt_score": "relative_gpt_score",
        "adapt_relative_mean_gpt_score": "relative_mean_gpt_score",
        "adapt_similarity_score": "similarity_score",
        "adapt_relative_score": "relative_score",
        "adapt_relative_mean_score": "relative_mean_score",
    }
    return safe_float(adaptation.get(mapping[metric]))


def profile_overall(user: Dict[str, Any]) -> Optional[float]:
    return safe_float((user.get("profile_alignment") or {}).get("overall"))


def profile_averages(users: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    buckets = {key: [] for key in PROFILE_KEYS}
    for user in users.values():
        profile = user.get("profile_alignment") or {}
        summary = user.get("summary") or {}
        for key in PROFILE_KEYS:
            value = safe_float(profile.get(key))
            if value is None:
                value = safe_float(summary.get(f"profile_{key}"))
            if value is not None:
                buckets[key].append(value)
    return {
        key: float(np.mean(values)) if values else math.nan
        for key, values in buckets.items()
    }


def per_user_after20_mean(users: Dict[str, Dict[str, Any]], metric: str) -> Optional[float]:
    per_user = []
    for user in users.values():
        values = [
            turn_metric(turn, metric)
            for idx, turn in enumerate(user.get("turns", []))
            if idx >= AFTER_TURN
        ]
        values = [value for value in values if value is not None]
        if values:
            per_user.append(sum(values) / len(values))
    return mean(per_user)


def per_user_delta_from_first(users: Dict[str, Dict[str, Any]], metric: str) -> Optional[float]:
    deltas = []
    for user in users.values():
        turns = user.get("turns", [])
        if not turns:
            continue
        first = turn_metric(turns[0], metric)
        if first is None:
            continue
        later_values = [
            turn_metric(turn, metric)
            for idx, turn in enumerate(turns)
            if idx >= AFTER_TURN
        ]
        later_values = [value for value in later_values if value is not None]
        if later_values:
            deltas.append(sum(later_values) / len(later_values) - first)
    return mean(deltas)


def format_num(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def build_turn_series(
    users: Dict[str, Dict[str, Any]],
    metric: str,
    min_users_per_turn: int,
) -> Dict[str, List[float]]:
    max_turns = max((len(user.get("turns", [])) for user in users.values()), default=0)
    x: List[int] = []
    y: List[float] = []
    counts: List[int] = []

    for turn_index in range(max_turns):
        values = []
        for user in users.values():
            turns = user.get("turns", [])
            if turn_index >= len(turns):
                continue
            value = turn_metric(turns[turn_index], metric)
            if value is not None:
                values.append(value)

        if len(values) < min_users_per_turn:
            continue
        avg = mean(values)
        if avg is None:
            continue
        x.append(turn_index)
        y.append(avg)
        counts.append(len(values))

    return {"x": x, "y": y, "counts": counts}


def parse_result_specs(specs: List[List[str]]) -> List[Tuple[Path, str]]:
    return [(Path(path), name) for path, name in specs]


def matched_user_ids(
    results: List[Tuple[Path, str]],
    reference_result: Path,
    max_skip_ratio: float,
    n_users: int,
) -> Tuple[List[str], Dict[str, float]]:
    id_sets = [set(sorted_user_ids(path)) for path, _ in results]
    if not id_sets:
        raise ValueError("At least one --result PATH NAME pair is required.")
    excluded = reference_skip_exclude_ids(reference_result, max_skip_ratio)
    common = sorted(set.intersection(*id_sets) - set(excluded))
    if n_users > 0:
        common = common[:n_users]
    if not common:
        raise ValueError("No matched users after applying intersection and reference skip filter.")
    return common, excluded


def plot_learning_trajectories(
    result_users: List[Tuple[str, Dict[str, Dict[str, Any]]]],
    output: Path,
    min_users_per_turn: int,
    smooth_window: int,
    show_raw: bool,
    raw_alpha: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    axes_list = list(axes.flatten())
    colors = plt.get_cmap("tab10")

    for ax, (metric, title, ylim) in zip(axes_list[:3], LINE_METRICS):
        for idx, (label, users) in enumerate(result_users):
            series = build_turn_series(users, metric, min_users_per_turn)
            color = colors(idx)
            if not series["x"]:
                continue
            if show_raw:
                ax.plot(
                    series["x"],
                    series["y"],
                    linewidth=1.1,
                    alpha=raw_alpha,
                    color=color,
                    label=f"{label} raw",
                )
            ax.plot(
                series["x"],
                centered_moving_average(series["y"], smooth_window),
                marker="o",
                markersize=3,
                linewidth=2.3,
                color=color,
                label=f"{label} smooth" if show_raw else label,
            )
        if "relative" in metric:
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.set_xlabel("Turn Index")
        ax.set_ylabel("Mean Across Matched Users")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    ax = axes_list[3]
    labels = []
    values = []
    for label, users in result_users:
        vals = [v for v in (profile_overall(user) for user in users.values()) if v is not None]
        labels.append(label)
        values.append(mean(vals) if vals else math.nan)
    bars = ax.bar(labels, values, color=[colors(i) for i in range(len(labels))])
    ax.set_title("Profile Overall Score")
    ax.set_ylabel("Mean Profile Overall")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=15)
    ymax = max((v for v in values if not math.isnan(v)), default=1.0)
    ax.set_ylim(0, max(1.0, ymax * 1.18))
    for bar, value in zip(bars, values):
        if not math.isnan(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02 * max(1.0, ymax),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Learning Trajectory Comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_table(
    result_users: List[Tuple[str, Dict[str, Dict[str, Any]]]],
    output: Path,
) -> None:
    rows: List[Dict[str, str]] = []
    for label, users in result_users:
        prof = profile_averages(users)
        row = {"run": label, "n_users": str(len(users))}
        for metric, _, _ in LINE_METRICS:
            short = metric.replace("prediction_", "pred_").replace("adapt_", "")
            row[f"{short}_after20"] = format_num(per_user_after20_mean(users, metric))
            row[f"{short}_delta_after20_vs_turn0"] = format_num(
                per_user_delta_from_first(users, metric)
            )
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
    parser = argparse.ArgumentParser(description="Compare learning trajectories across multiple result directories.")
    parser.add_argument(
        "--result",
        nargs=2,
        action="append",
        metavar=("PATH", "NAME"),
        required=True,
        help="Result root and display name. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--reference-result",
        type=Path,
        default=None,
        help="Result root whose records define the skip-ratio filter. Defaults to the first --result path.",
    )
    parser.add_argument("--n-users", type=int, default=0, help="Number of matched users to plot; 0 means all matched users.")
    parser.add_argument("--min-users-per-turn", type=int, default=10)
    parser.add_argument("--max-reference-skip-ratio", type=float, default=0.75)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--show-raw", action="store_true")
    parser.add_argument("--raw-alpha", type=float, default=0.22)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("result/plots/learning_trajectory_comparison.png"),
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=Path("result/plots/learning_trajectory_comparison_table.csv"),
    )
    args = parser.parse_args()

    results = parse_result_specs(args.result)
    reference_result = args.reference_result or results[0][0]
    selected_ids, excluded = matched_user_ids(
        results=results,
        reference_result=reference_result,
        max_skip_ratio=args.max_reference_skip_ratio,
        n_users=args.n_users,
    )
    result_users = [
        (name, load_users(path, selected_ids))
        for path, name in results
    ]

    plot_learning_trajectories(
        result_users=result_users,
        output=args.output,
        min_users_per_turn=args.min_users_per_turn,
        smooth_window=args.smooth_window,
        show_raw=args.show_raw,
        raw_alpha=args.raw_alpha,
    )
    write_table(result_users=result_users, output=args.table_output)

    print(f"Reference result: {reference_result}")
    print(f"Users excluded by reference skip ratio > {args.max_reference_skip_ratio}: {len(excluded)}")
    print(f"Matched users plotted: {len(selected_ids)}")
    print(", ".join(selected_ids))
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
