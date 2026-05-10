import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def centered_moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    n = len(values)
    if n == 0:
        return []
    half = window // 2
    smoothed: List[float] = []
    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        smoothed.append(mean(values[left:right]))
    return smoothed


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_prism_turn_metric(turn: Dict[str, Any], metric: str) -> Optional[float]:
    if metric == "accuracy":
        return safe_float(turn.get("choice_metrics", {}).get("accuracy"))
    if metric == "relative_gpt_score":
        return safe_float(turn.get("generation_metrics", {}).get("relative_gpt_score"))
    if metric == "relative_similarity_score":
        # In prism records this metric is named relative_score.
        return safe_float(turn.get("generation_metrics", {}).get("relative_score"))
    if metric == "similarity_score":
        return safe_float(turn.get("generation_metrics", {}).get("similarity_score"))
    return None


def build_turnwise_series(
    baseline_users: List[Dict[str, Any]],
    prism_users: List[Dict[str, Any]],
    metric: str,
    min_users_per_turn: int,
) -> Dict[str, Dict[str, List[float]]]:
    baseline_max_turns = max(len(u.get("turn_results", [])) for u in baseline_users) if baseline_users else 0
    prism_max_turns = max(len(u.get("turns", [])) for u in prism_users) if prism_users else 0
    max_turns = max(baseline_max_turns, prism_max_turns)

    series = {
        "baseline": {"x": [], "y": [], "count": []},
        "prism": {"x": [], "y": [], "count": []},
    }

    for t in range(max_turns):
        baseline_vals: List[float] = []
        prism_vals: List[float] = []

        for u in baseline_users:
            turns = u.get("turn_results", [])
            if t >= len(turns):
                continue
            v = safe_float(turns[t].get(metric))
            if v is not None:
                baseline_vals.append(v)

        for u in prism_users:
            turns = u.get("turns", [])
            if t >= len(turns):
                continue
            v = get_prism_turn_metric(turns[t], metric)
            if v is not None:
                prism_vals.append(v)

        if len(baseline_vals) < min_users_per_turn and len(prism_vals) < min_users_per_turn:
            break

        if len(baseline_vals) >= min_users_per_turn:
            series["baseline"]["x"].append(t)
            series["baseline"]["y"].append(mean(baseline_vals))
            series["baseline"]["count"].append(len(baseline_vals))

        if len(prism_vals) >= min_users_per_turn:
            series["prism"]["x"].append(t)
            series["prism"]["y"].append(mean(prism_vals))
            series["prism"]["count"].append(len(prism_vals))

    return series


def collect_overall_mean(baseline_users: List[Dict[str, Any]], prism_users: List[Dict[str, Any]], metric: str) -> Tuple[float, float]:
    baseline_vals: List[float] = []
    prism_vals: List[float] = []

    for u in baseline_users:
        for turn in u.get("turn_results", []):
            v = safe_float(turn.get(metric))
            if v is not None:
                baseline_vals.append(v)

    for u in prism_users:
        for turn in u.get("turns", []):
            v = get_prism_turn_metric(turn, metric)
            if v is not None:
                prism_vals.append(v)

    baseline_mean = mean(baseline_vals) if baseline_vals else math.nan
    prism_mean = mean(prism_vals) if prism_vals else math.nan
    return baseline_mean, prism_mean


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare COT baseline and gpt-5.4-nano-trace-prism on matched users.")
    parser.add_argument("--baseline", type=Path, default=Path("results.json"), help="Path to baseline results.json")
    parser.add_argument(
        "--prism-records-dir",
        type=Path,
        default=Path("result/gpt5.4-nano-trace-prism/records"),
        help="Directory containing prism user JSON files.",
    )
    parser.add_argument("--top-n", type=int, default=60, help="Take first N users from baseline results.")
    parser.add_argument("--min-users-per-turn", type=int, default=10, help="Minimum users required to keep a turn.")
    parser.add_argument("--smooth-window", type=int, default=5, help="Centered moving-average window size.")
    parser.add_argument("--raw-alpha", type=float, default=0.23, help="Alpha for raw background trend lines.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("result/gpt5.4-nano-trace-prism/plots/cot_baseline_vs_gpt54_nano_trace_prism_top60_smooth5.png"),
        help="Output comparison figure path.",
    )
    args = parser.parse_args()

    baseline_data = load_json(args.baseline)
    if not isinstance(baseline_data, list):
        raise ValueError(f"Expected baseline data to be a list, got: {type(baseline_data).__name__}")

    baseline_subset = baseline_data[: args.top_n]
    baseline_user_ids = [u.get("user_id") for u in baseline_subset if isinstance(u, dict) and u.get("user_id")]

    if not baseline_user_ids:
        raise ValueError("No valid user_id found in baseline subset.")

    matched_baseline: List[Dict[str, Any]] = []
    matched_prism: List[Dict[str, Any]] = []
    missing_user_ids: List[str] = []

    for uid in baseline_user_ids:
        prism_path = args.prism_records_dir / f"{uid}.json"
        if not prism_path.exists():
            missing_user_ids.append(uid)
            continue

        baseline_user = next((u for u in baseline_subset if u.get("user_id") == uid), None)
        if baseline_user is None:
            continue

        prism_user = load_json(prism_path)
        matched_baseline.append(baseline_user)
        matched_prism.append(prism_user)

    if not matched_baseline or not matched_prism:
        raise ValueError("No matched users between baseline and prism records.")

    metric_specs = [
        ("accuracy", "Accuracy"),
        ("relative_gpt_score", "Relative GPT Score"),
        ("relative_similarity_score", "Relative Similarity Score"),
        ("similarity_score", "Similarity Score"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_list = list(axes.flatten())

    overall_summary: Dict[str, Tuple[float, float]] = {}

    for ax, (metric, title) in zip(axes_list, metric_specs):
        turnwise = build_turnwise_series(
            baseline_users=matched_baseline,
            prism_users=matched_prism,
            metric=metric,
            min_users_per_turn=args.min_users_per_turn,
        )

        bx = turnwise["baseline"]["x"]
        by = turnwise["baseline"]["y"]
        px = turnwise["prism"]["x"]
        py = turnwise["prism"]["y"]
        by_smooth = centered_moving_average(by, args.smooth_window)
        py_smooth = centered_moving_average(py, args.smooth_window)

        if bx and by:
            ax.plot(
                bx,
                by,
                linewidth=1.5,
                alpha=args.raw_alpha,
                color="#1f77b4",
                label="COT baseline (raw)",
            )
            ax.plot(
                bx,
                by_smooth,
                marker="o",
                markersize=4,
                linewidth=2.6,
                color="#1f77b4",
                label=f"COT baseline (smooth, w={args.smooth_window})",
            )
        if px and py:
            ax.plot(
                px,
                py,
                linewidth=1.5,
                alpha=args.raw_alpha,
                color="#ff7f0e",
                label="gpt-5.4-nano-trace-prism (raw)",
            )
            ax.plot(
                px,
                py_smooth,
                marker="s",
                markersize=4,
                linewidth=2.6,
                color="#ff7f0e",
                label=f"gpt-5.4-nano-trace-prism (smooth, w={args.smooth_window})",
            )

        if "relative" in metric:
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

        ax.set_title(title)
        ax.set_xlabel("Turn Index")
        ax.set_ylabel("Average Score")
        ax.grid(True, alpha=0.3)
        ax.legend()

        overall_summary[metric] = collect_overall_mean(matched_baseline, matched_prism, metric)

    fig.suptitle(
        (
            f"COT baseline vs gpt-5.4-nano-trace-prism "
            f"(matched by first {args.top_n} baseline users, smooth window={args.smooth_window})"
        ),
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Baseline users requested: {len(baseline_user_ids)}")
    print(f"Matched users: {len(matched_baseline)}")
    print(f"Missing users in prism records: {len(missing_user_ids)}")
    if missing_user_ids:
        print("Missing user IDs:")
        print(", ".join(missing_user_ids))
    print(f"Saved figure: {args.output}")

    print("\nOverall means across all available turns:")
    for metric, (baseline_mean, prism_mean) in overall_summary.items():
        print(f"- {metric}: baseline={baseline_mean:.4f}, prism={prism_mean:.4f}, delta={prism_mean - baseline_mean:+.4f}")


if __name__ == "__main__":
    main()
