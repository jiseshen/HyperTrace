"""
Analysis and visualization for openrouter-gpt5-trace-gemini3-flash-eval.

Filters out users whose skipped-turn ratio > 3/4, then plots:
  1. Prediction accuracy vs turn index  — 95% Wilson score CI
  2. Relative GPT score vs turn index   — 95% normal CI
  3. Relative similarity score vs turn  — 95% normal CI
  4. Profile alignment bar chart (5 metrics)
All line plots use 5-turn centered rolling average; CI bounds smoothed separately.
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── config ──────────────────────────────────────────────────────────────────
RUN = "openrouter-gpt5-trace-gemini3-flash-eval"
RESULT_ROOT = Path("result") / RUN
RECORD_DIR = RESULT_ROOT / "records"
METRIC_DIR = RESULT_ROOT / "metrics" / "users"
SAVE_DIR = RESULT_ROOT / "plots"

MAX_SKIP_RATIO = 0.75
SMOOTH_WIN = 5
MIN_USERS_PER_TURN = 5
SHADOW_ALPHA = 0.22
Z95 = 1.96


# ── helpers ──────────────────────────────────────────────────────────────────
def load_json(p: Path) -> dict:
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def skip_ratio(record: dict) -> float:
    turns = record.get("turns", [])
    if not turns:
        return 0.0
    skipped = sum(1 for t in turns if t.get("preprocess", {}).get("skip", False))
    return skipped / len(turns)


def smooth(arr: np.ndarray, win: int) -> np.ndarray:
    """Centered rolling mean; edges padded with edge values."""
    if win <= 1 or len(arr) < win:
        return arr.copy()
    kernel = np.ones(win) / win
    padded = np.pad(arr, win // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def wilson_ci(successes: np.ndarray, n: int) -> tuple[float, float, float]:
    """
    Wilson score interval for binary data.
    Returns (p_hat, lower, upper) clipped to [0, 1].
    """
    p = successes.mean()
    z2n = Z95 ** 2 / n
    center = (p + z2n / 2) / (1 + z2n)
    half = Z95 * math.sqrt(p * (1 - p) / n + z2n / (4 * n)) / (1 + z2n)
    return p, max(0.0, center - half), min(1.0, center + half)


def normal_ci(vals: np.ndarray) -> tuple[float, float, float]:
    """95% normal CI: mean ± 1.96 * SEM."""
    mu = vals.mean()
    sem = vals.std() / math.sqrt(len(vals))
    return mu, mu - Z95 * sem, mu + Z95 * sem


def collect_turn_series(metric_users: list[dict]) -> dict[str, np.ndarray]:
    """
    Returns per-turn mean, lo, hi for each metric.
    prediction_accuracy uses Wilson score CI; others use normal CI.
    NaN-filled where fewer than MIN_USERS_PER_TURN users present.
    """
    max_t = max(len(u["turns"]) for u in metric_users)
    nan = float("nan")

    store: dict[str, dict[str, list]] = {
        k: {"mean": [], "lo": [], "hi": []}
        for k in ("prediction_accuracy", "relative_gpt_score", "relative_score")
    }
    counts: list[int] = []

    for t in range(max_t):
        buckets: dict[str, list[float]] = {k: [] for k in store}
        for u in metric_users:
            turns = u["turns"]
            if t >= len(turns):
                continue
            turn = turns[t]
            pred = turn.get("prediction") or {}
            adap = turn.get("adaptation") or {}
            for k, src, field in [
                ("prediction_accuracy", pred, "accuracy"),
                ("relative_gpt_score", adap, "relative_gpt_score"),
                ("relative_score", adap, "relative_score"),
            ]:
                v = src.get(field)
                if v is not None and not math.isnan(v):
                    buckets[k].append(float(v))

        n_present = max(len(b) for b in buckets.values())
        counts.append(n_present)

        for k, entry in store.items():
            vals = np.array(buckets[k])
            if len(vals) >= MIN_USERS_PER_TURN:
                if k == "prediction_accuracy":
                    mu, lo, hi = wilson_ci(vals, len(vals))
                else:
                    mu, lo, hi = normal_ci(vals)
                entry["mean"].append(mu)
                entry["lo"].append(lo)
                entry["hi"].append(hi)
            else:
                entry["mean"].append(nan)
                entry["lo"].append(nan)
                entry["hi"].append(nan)

    valid = [i for i, c in enumerate(counts) if c >= MIN_USERS_PER_TURN]
    cutoff = valid[-1] + 1 if valid else 0

    result: dict[str, np.ndarray] = {
        "turn": np.arange(cutoff),
        "counts": np.array(counts[:cutoff]),
    }
    for k, entry in store.items():
        result[f"{k}_mean"] = np.array(entry["mean"][:cutoff])
        result[f"{k}_lo"] = np.array(entry["lo"][:cutoff])
        result[f"{k}_hi"] = np.array(entry["hi"][:cutoff])
    return result


def collect_profile_averages(metric_users: list[dict]) -> dict[str, float]:
    keys = ["survey_consistency", "key_aspect_match", "internal_plausibility", "overall", "similarity"]
    buckets: dict[str, list[float]] = {k: [] for k in keys}
    for u in metric_users:
        pa = u.get("profile_alignment") or {}
        for k in keys:
            v = pa.get(k)
            if v is not None and not math.isnan(v):
                buckets[k].append(float(v))
    return {k: float(np.mean(vs)) if vs else float("nan") for k, vs in buckets.items()}


# ── plotting ──────────────────────────────────────────────────────────────────
STYLE = dict(linewidth=2, marker="o", markersize=3)
COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def plot_turn_metric(
    turns: np.ndarray,
    mean: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    label: str,
    ylabel: str,
    title: str,
    save_path: Path,
    color: str,
    ci_label: str = "95% CI",
    ylim=None,
) -> None:
    sm = smooth(mean, SMOOTH_WIN)
    sl = smooth(lo, SMOOTH_WIN)
    sh = smooth(hi, SMOOTH_WIN)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(turns, sl, sh, alpha=SHADOW_ALPHA, color=color, label=ci_label)
    ax.plot(turns, sm, color=color, label=label, **STYLE)
    ax.set_xlabel("Turn Index", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {save_path}")


def plot_profile_bar(avgs: dict[str, float], save_path: Path) -> None:
    labels = ["survey\nconsistency", "key aspect\nmatch", "internal\nplausibility", "overall", "similarity"]
    keys = ["survey_consistency", "key_aspect_match", "internal_plausibility", "overall", "similarity"]
    values = [avgs[k] for k in keys]

    # normalise "similarity" (0-1 range) onto a 0-5 scale for visual comparison,
    # but show it with a secondary axis note instead — keep raw values, add note in title.
    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = plt.cm.tab10(np.linspace(0, 0.5, len(labels)))
    bars = ax.bar(labels, values, color=bar_colors, edgecolor="white", linewidth=0.8)

    for bar, v in zip(bars, values):
        if not math.isnan(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.04,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=10,
            )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Average Profile Alignment Metrics\n(similarity is cosine 0–1; others are 1–5 LLM rubric)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ymax = max((v for v in values if not math.isnan(v)), default=1.0)
    ax.set_ylim(0, max(1.1, ymax * 1.2))
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # load records to compute skip ratios
    record_files = sorted(RECORD_DIR.glob("*.json"))
    skip_map: dict[str, float] = {}
    for p in record_files:
        rec = load_json(p)
        skip_map[p.stem] = skip_ratio(rec)

    # load metric files, filtering by skip ratio
    metric_users = []
    dropped = []
    for p in sorted(METRIC_DIR.glob("*.json")):
        ratio = skip_map.get(p.stem, 0.0)
        if ratio > MAX_SKIP_RATIO:
            dropped.append((p.stem, ratio))
            continue
        metric_users.append(load_json(p))

    print(f"Users loaded : {len(metric_users)}")
    print(f"Users dropped: {len(dropped)}  (skip ratio > {MAX_SKIP_RATIO})")
    if dropped:
        for uid, r in dropped:
            print(f"  {uid}  skip={r:.2f}")

    # ── turn-wise metrics ────────────────────────────────────────────────────
    series = collect_turn_series(metric_users)
    turns = series["turn"]
    print(f"Turn curve length: {len(turns)}")

    plot_turn_metric(
        turns,
        series["prediction_accuracy_mean"],
        series["prediction_accuracy_lo"],
        series["prediction_accuracy_hi"],
        label="Prediction Accuracy",
        ylabel="Accuracy",
        title=f"Prediction Accuracy vs Turn  (n≥{MIN_USERS_PER_TURN} users, {SMOOTH_WIN}-turn smooth)",
        save_path=SAVE_DIR / "turn_prediction_accuracy.png",
        color=COLORS[0],
        ci_label="95% Wilson CI",
        ylim=(0, 1.05),
    )

    plot_turn_metric(
        turns,
        series["relative_gpt_score_mean"],
        series["relative_gpt_score_lo"],
        series["relative_gpt_score_hi"],
        label="Relative GPT Score",
        ylabel="Relative GPT Score",
        title=f"Relative GPT Score vs Turn  (n≥{MIN_USERS_PER_TURN} users, {SMOOTH_WIN}-turn smooth)",
        save_path=SAVE_DIR / "turn_relative_gpt_score.png",
        color=COLORS[1],
        ci_label="95% Normal CI",
    )

    plot_turn_metric(
        turns,
        series["relative_score_mean"],
        series["relative_score_lo"],
        series["relative_score_hi"],
        label="Relative Similarity Score",
        ylabel="Relative Similarity Score",
        title=f"Relative Similarity Score vs Turn  (n≥{MIN_USERS_PER_TURN} users, {SMOOTH_WIN}-turn smooth)",
        save_path=SAVE_DIR / "turn_relative_similarity_score.png",
        color=COLORS[2],
        ci_label="95% Normal CI",
    )

    # ── profile alignment bar ────────────────────────────────────────────────
    profile_avgs = collect_profile_averages(metric_users)
    print("Profile alignment averages:")
    for k, v in profile_avgs.items():
        print(f"  {k}: {v:.4f}")

    plot_profile_bar(profile_avgs, SAVE_DIR / "profile_alignment_bar.png")

    print(f"\nAll plots saved to: {SAVE_DIR.resolve()}")


if __name__ == "__main__":
    main()
