"""Plot saved evaluation summaries without provider calls or score transformations."""

import argparse
import json
import math
from pathlib import Path

METRICS = {
    "prediction_accuracy": "Prediction accuracy",
    "prediction_ranking_score": "Prediction ranking score",
    "adapt_relative_gpt_score": "Relative judge score",
    "adapt_relative_score": "Relative embedding score",
}


def metric_series(summary, metric):
    """Preserve saved turn indices and values, including gaps in missing metrics."""
    rows = sorted(summary.get("online_turns", []), key=lambda row: row["turn_index"])
    return (
        [row["turn_index"] for row in rows],
        [float(row[metric]) if row.get(metric) is not None else math.nan for row in rows],
    )


def plot_summary(summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    series = {key: metric_series(summary, key) for key in METRICS}
    available = {key: values for key, values in series.items() if any(math.isfinite(y) for y in values[1])}
    if not available:
        raise ValueError("Summary has no finite online metrics to plot; evaluate records first.")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, (turns, values) in available.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(turns, values, marker="o", markersize=3)
        ax.set(xlabel="Turn index", ylabel=METRICS[key], title=METRICS[key])
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = output_dir / f"{key}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    # Expose changing sample sizes alongside the metric curves.
    rows = sorted(summary["online_turns"], key=lambda row: row["turn_index"])
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in (("n_users", "All users"), ("n_prediction_users", "Valid prediction"), ("n_adaptation_users", "Valid adaptation")):
        if any(row.get(key) is not None for row in rows):
            ax.plot([row["turn_index"] for row in rows], [row.get(key, math.nan) for row in rows], label=label)
    ax.set(xlabel="Turn index", ylabel="Users", title="Metric sample sizes")
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    path = output_dir / "sample_sizes.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory containing saved metrics")
    parser.add_argument("--metrics-name", default="metrics", help="Metrics directory inside the run")
    parser.add_argument("--output-dir", type=Path, help="Default: RUN_DIR/plots/METRICS_NAME")
    args = parser.parse_args(argv)
    summary_path = args.run_dir / args.metrics_name / "summary.json"
    if not summary_path.is_file():
        parser.error(f"Missing {summary_path}; run main.py with --eval-only for this cohort first")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        paths = plot_summary(summary, args.output_dir or args.run_dir / "plots" / args.metrics_name)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(f"Invalid metrics in {summary_path}: {exc}")
    except ImportError:
        parser.error("Plotting requires matplotlib: python -m pip install matplotlib")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
