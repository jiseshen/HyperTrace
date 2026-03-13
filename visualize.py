import json
import math
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


ROOT = Path("result/gpt5-nano-prism_42")
MIN_USERS_PER_TURN = 10
SAVE_DIR = Path("result") / "gpt5-nano-prism-plots" / "v2"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d: Dict[str, Any], *keys: str) -> Optional[float]:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    if cur is None:
        return None
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_user_files(root: Path) -> List[Path]:
    files = sorted(Path(p) for p in glob(str(root / "user*.json")))
    if not files:
        raise FileNotFoundError(f"No user JSON files found under: {root}")
    return files


def build_turnwise_averages(
    users: List[Dict[str, Any]],
    min_users_per_turn: int = 10,
) -> Dict[str, List[float]]:
    max_turns = max(len(u.get("turns", [])) for u in users)

    turn_indices: List[int] = []
    accuracy_avg: List[float] = []
    ranking_avg: List[float] = []
    gpt_avg: List[float] = []
    sim_avg: List[float] = []
    rel_avg: List[float] = []
    counts: List[int] = []

    for t in range(max_turns):
        acc_vals: List[float] = []
        rank_vals: List[float] = []
        gpt_vals: List[float] = []
        sim_vals: List[float] = []
        rel_vals: List[float] = []

        for user in users:
            turns = user.get("turns", [])
            if t >= len(turns):
                continue

            turn = turns[t]
            v = safe_get(turn, "choice_metrics", "accuracy")
            if v is not None:
                acc_vals.append(v)

            v = safe_get(turn, "choice_metrics", "ranking_score")
            if v is not None:
                rank_vals.append(v)

            v = safe_get(turn, "generation_metrics", "gpt_score")
            if v is not None:
                gpt_vals.append(v)

            v = safe_get(turn, "generation_metrics", "similarity_score")
            if v is not None:
                sim_vals.append(v)

            v = safe_get(turn, "generation_metrics", "relative_score")
            if v is not None:
                rel_vals.append(v)

        # 用“至少有多少人参与了这个 turn”作为截断标准
        # 这里取五类指标里样本数的最大值，避免因为某个字段偶尔缺失被过早截断
        n_present = max(
            len(acc_vals),
            len(rank_vals),
            len(gpt_vals),
            len(sim_vals),
            len(rel_vals),
        )

        if n_present < min_users_per_turn:
            break

        turn_indices.append(t)
        counts.append(n_present)
        accuracy_avg.append(mean(acc_vals) if acc_vals else math.nan)
        ranking_avg.append(mean(rank_vals) if rank_vals else math.nan)
        gpt_avg.append(mean(gpt_vals) if gpt_vals else math.nan)
        sim_avg.append(mean(sim_vals) if sim_vals else math.nan)
        rel_avg.append(mean(rel_vals) if rel_vals else math.nan)

    return {
        "turn": turn_indices,
        "count": counts,
        "accuracy": accuracy_avg,
        "ranking_score": ranking_avg,
        "gpt_score": gpt_avg,
        "similarity_score": sim_avg,
        "relative_score": rel_avg,
    }


def build_profile_averages(users: List[Dict[str, Any]]) -> Dict[str, float]:
    metrics = {
        "survey_consistency": [],
        "key_aspect_match": [],
        "internal_plausibility": [],
        "overall": [],
        "similarity": [],
    }

    for user in users:
        pm = user.get("profile_metrics", {})
        for k in metrics:
            v = safe_get({"profile_metrics": pm}, "profile_metrics", k)
            if v is not None:
                metrics[k].append(v)

    return {k: mean(vs) if vs else math.nan for k, vs in metrics.items()}


def plot_line(
    x: List[int],
    y: List[float],
    counts: List[int],
    title: str,
    ylabel: str,
    save_path: Path,
    ylim: Optional[tuple] = None,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.xlabel("Turn Index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if ylim is not None:
        plt.ylim(*ylim)

    # 在点上标注参与用户数
    for xi, yi, ci in zip(x, y, counts):
        if yi == yi:  # not nan
            plt.annotate(
                f"n={ci}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_profile_bar(profile_avg: Dict[str, float], save_path: Path) -> None:
    labels = [
        "survey_consistency",
        "key_aspect_match",
        "internal_plausibility",
        "overall",
        "similarity",
    ]
    values = [profile_avg[k] for k in labels]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values)
    plt.title("Average Profile Metrics Across Users")
    plt.ylabel("Score")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)

    ymax = max(v for v in values if v == v) if any(v == v for v in values) else 1.0
    plt.ylim(0, max(1.0, ymax * 1.2))

    for bar, v in zip(bars, values):
        if v == v:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02 * max(1.0, ymax),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def print_summary(turnwise: Dict[str, List[float]], profile_avg: Dict[str, float], n_users: int) -> None:
    print(f"Loaded users: {n_users}")
    print(f"Turn curve length after truncation (min {MIN_USERS_PER_TURN} users/turn): {len(turnwise['turn'])}")
    if turnwise["turn"]:
        print(f"Last retained turn index: {turnwise['turn'][-1]}")
        print(f"Users in last retained turn: {turnwise['count'][-1]}")
    print("Average profile metrics:")
    for k, v in profile_avg.items():
        print(f"  {k}: {v:.6f}" if v == v else f"  {k}: nan")


def main() -> None:
    ensure_dir(SAVE_DIR)

    user_files = collect_user_files(ROOT)
    users = [load_json(p) for p in user_files]

    turnwise = build_turnwise_averages(users, min_users_per_turn=MIN_USERS_PER_TURN)
    profile_avg = build_profile_averages(users)

    # 5 张随 turn 变化的图
    plot_line(
        turnwise["turn"],
        turnwise["accuracy"],
        turnwise["count"],
        title="Average Choice Accuracy vs Turn",
        ylabel="Choice Accuracy",
        save_path=SAVE_DIR / "turn_choice_accuracy.png",
        ylim=(0, 1.05),
    )

    plot_line(
        turnwise["turn"],
        turnwise["ranking_score"],
        turnwise["count"],
        title="Average Choice Ranking Score vs Turn",
        ylabel="Ranking Score",
        save_path=SAVE_DIR / "turn_choice_ranking_score.png",
        ylim=(0, 1.05),
    )

    plot_line(
        turnwise["turn"],
        turnwise["gpt_score"],
        turnwise["count"],
        title="Average Generation GPT Score vs Turn",
        ylabel="GPT Score",
        save_path=SAVE_DIR / "turn_generation_gpt_score.png",
    )

    plot_line(
        turnwise["turn"],
        turnwise["similarity_score"],
        turnwise["count"],
        title="Average Generation Similarity Score vs Turn",
        ylabel="Similarity Score",
        save_path=SAVE_DIR / "turn_generation_similarity_score.png",
    )

    plot_line(
        turnwise["turn"],
        turnwise["relative_score"],
        turnwise["count"],
        title="Average Generation Relative Score vs Turn",
        ylabel="Relative Score",
        save_path=SAVE_DIR / "turn_generation_relative_score.png",
    )

    # 额外的 profile metrics 柱状图
    plot_profile_bar(
        profile_avg,
        save_path=SAVE_DIR / "profile_metrics_bar.png",
    )

    print_summary(turnwise, profile_avg, len(users))
    print(f"Saved plots to: {SAVE_DIR.resolve()}")


if __name__ == "__main__":
    main()