"""Generate a four-panel Bradley-Terry likelihood schematic."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Sequence, Tuple

from plot_schematic_bar_chart import DEFAULT_PALETTE, draw_chart


DEFAULT_SCORE_PAIRS = [(3.0, 5.0), (2.0, 2.0), (2.0, 5.0)]
DEFAULT_HYPOTHESIS_COLORS = ["#cc0000", "#6aa84f", "#bf9000"]
DEFAULT_LINEWIDTH = 5


def parse_score_pairs(raw: str) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [item.strip() for item in chunk.split(",") if item.strip()]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("--score-pairs must look like '3,5;2,2;2,5'")
        try:
            left, right = float(parts[0]), float(parts[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError("--score-pairs must contain numeric scores") from exc
        if left < 0 or right < 0:
            raise argparse.ArgumentTypeError("--score-pairs cannot contain negative scores")
        pairs.append((left, right))
    if not pairs:
        raise argparse.ArgumentTypeError("--score-pairs cannot be empty")
    return pairs


def bradley_terry_probability(scores: Sequence[float], chosen_idx: int, temperature: float) -> float:
    if temperature <= 0:
        raise ValueError("--temperature must be positive")
    scaled = [score / temperature for score in scores]
    offset = max(scaled)
    exps = [math.exp(score - offset) for score in scaled]
    return exps[chosen_idx] / sum(exps)


def normalized(values: Sequence[float]) -> List[float]:
    total = sum(values)
    if total <= 0:
        raise ValueError("Cannot normalize non-positive likelihoods")
    return [value / total for value in values]


def draw_schematic_set(
    *,
    score_pairs: Sequence[Tuple[float, float]],
    output_dir: Path,
    temperature: float,
    chosen_idx: int,
    utility_scale_max: float,
    transparent: bool,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []

    for idx, pair in enumerate(score_pairs, start=1):
        output = output_dir / f"hypothesis_{idx}_utility.svg"
        draw_chart(
            heights=list(pair),
            colors=DEFAULT_PALETTE[:2],
            output=output,
            figsize=(3.0, 2.7),
            dpi=240,
            transparent=transparent,
            show_axis=True,
            axis_color="#000000",
            background="#FFFFFF",
            bar_edge_color="#000000",
            bar_edge_width=DEFAULT_LINEWIDTH,
            bar_bottom_gap=0.0,
            axis_linewidth=DEFAULT_LINEWIDTH,
            scale_max=utility_scale_max,
        )
        outputs.append(output)

    likelihoods = [
        bradley_terry_probability(pair, chosen_idx=chosen_idx, temperature=temperature)
        for pair in score_pairs
    ]
    posterior = normalized(likelihoods)
    output = output_dir / "hypothesis_posterior.svg"
    draw_chart(
        heights=posterior,
        colors=DEFAULT_HYPOTHESIS_COLORS,
        output=output,
        figsize=(3.8, 2.7),
        dpi=240,
        transparent=transparent,
        show_axis=True,
        axis_color="#000000",
        background="#FFFFFF",
        bar_edge_color="#000000",
        bar_edge_width=DEFAULT_LINEWIDTH,
        bar_bottom_gap=0.0,
        axis_linewidth=DEFAULT_LINEWIDTH,
        scale_max=max(posterior),
    )
    outputs.append(output)

    print("Bradley-Terry P(candidate | hypothesis):")
    for idx, (pair, likelihood, weight) in enumerate(zip(score_pairs, likelihoods, posterior), start=1):
        print(f"  H{idx}: scores={pair}, likelihood={likelihood:.4f}, normalized={weight:.4f}")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate utility and Bradley-Terry posterior schematic bar charts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--score-pairs", type=parse_score_pairs, default=DEFAULT_SCORE_PAIRS, help="Candidate utility score pairs, e.g. '3,5;2,2;2,5'.")
    parser.add_argument("--chosen-idx", type=int, default=1, help="Chosen candidate index in each pair, zero-based.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Bradley-Terry softmax temperature.")
    parser.add_argument("--utility-scale-max", type=float, default=5.0, help="Fixed y-scale for utility charts.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures/bradley_terry_schematic"), help="Output directory.")
    parser.add_argument("--transparent", action="store_true", help="Export SVGs with transparent backgrounds.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.chosen_idx < 0:
        parser.error("--chosen-idx must be non-negative")
    if any(args.chosen_idx >= len(pair) for pair in args.score_pairs):
        parser.error("--chosen-idx is out of range for at least one score pair")
    if args.utility_scale_max < max(max(pair) for pair in args.score_pairs):
        parser.error("--utility-scale-max must be at least the largest utility score")

    outputs = draw_schematic_set(
        score_pairs=args.score_pairs,
        output_dir=args.output_dir,
        temperature=args.temperature,
        chosen_idx=args.chosen_idx,
        utility_scale_max=args.utility_scale_max,
        transparent=args.transparent,
    )
    print("Saved figures:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
