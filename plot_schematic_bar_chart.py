"""Generate a minimal rounded bar chart schematic for paper figures."""

from __future__ import annotations

import argparse
import os
import random
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence

# Keep matplotlib/fontconfig cache writes out of the user's home directory.
_CACHE_DIR = str(Path(tempfile.gettempdir()) / "preference_tracing_matplotlib")
os.environ.setdefault("MPLCONFIGDIR", _CACHE_DIR)
os.environ.setdefault("XDG_CACHE_HOME", _CACHE_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


DEFAULT_PALETTE = [
    "#4C78A8",  # blue
    "#F58518",  # orange
    "#54A24B",  # green
    "#E45756",  # red
    "#72B7B2",  # teal
    "#B279A2",  # mauve
    "#FF9DA6",  # pink
    "#9D755D",  # warm gray
    "#59A14F",  # leaf
    "#EDC948",  # gold
]

BEZIER_KAPPA = 0.6


def parse_float_list(raw: str | None, *, option_name: str) -> List[float] | None:
    if raw is None:
        return None
    values: List[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{option_name} must be comma-separated numbers: {raw}") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{option_name} cannot be empty")
    return values


def parse_color_list(raw: str | None) -> List[str] | None:
    if raw is None:
        return None
    colors = [item.strip() for item in raw.split(",") if item.strip()]
    if not colors:
        raise argparse.ArgumentTypeError("--colors cannot be empty")
    return colors


def cycle_to_length(values: Sequence[str], n: int) -> List[str]:
    return [values[i % len(values)] for i in range(n)]


def random_heights(n: int, seed: int, low: float, high: float) -> List[float]:
    if low <= 0:
        raise ValueError("--random-low must be positive")
    if high <= low:
        raise ValueError("--random-high must be greater than --random-low")
    rng = random.Random(seed)
    return [rng.uniform(low, high) for _ in range(n)]


def validate_heights(heights: Iterable[float]) -> List[float]:
    clean = list(heights)
    if not clean:
        raise ValueError("At least one bar height is required")
    if any(value < 0 for value in clean):
        raise ValueError("Bar heights must be non-negative")
    if max(clean) == 0:
        raise ValueError("At least one bar height must be greater than zero")
    return clean


def add_rounded_bar(
    ax: plt.Axes,
    *,
    x_center: float,
    y_base: float,
    height: float,
    width: float,
    color: str,
    rounding: float,
    edge_color: str,
    edge_width: float,
) -> None:
    x0 = x_center - width / 2
    x1 = x_center + width / 2
    y0 = y_base
    y1 = y_base + height
    r = min(rounding, width / 2, height / 2)
    k = BEZIER_KAPPA * r

    fill_path = MplPath(
        [
            (x0, y0),
            (x0, y1 - r),
            (x0, y1 - r + k),
            (x0 + r - k, y1),
            (x0 + r, y1),
            (x1 - r, y1),
            (x1 - r + k, y1),
            (x1, y1 - r + k),
            (x1, y1 - r),
            (x1, y0),
            (x0, y0),
            (x0, y0),
        ],
        [
            MplPath.MOVETO,
            MplPath.LINETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.LINETO,
            MplPath.CLOSEPOLY,
        ],
    )
    ax.add_patch(
        PathPatch(
            fill_path,
            linewidth=0,
            facecolor=color,
            alpha=0.96,
            zorder=3,
        )
    )

    outline_path = MplPath(
        [
            (x0, y0),
            (x0, y1 - r),
            (x0, y1 - r + k),
            (x0 + r - k, y1),
            (x0 + r, y1),
            (x1 - r, y1),
            (x1 - r + k, y1),
            (x1, y1 - r + k),
            (x1, y1 - r),
            (x1, y0),
        ],
        [
            MplPath.MOVETO,
            MplPath.LINETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.LINETO,
        ],
    )
    ax.add_patch(
        PathPatch(
            outline_path,
            fill=False,
            linewidth=edge_width,
            edgecolor=edge_color,
            capstyle="butt",
            joinstyle="round",
            zorder=4,
        )
    )


def add_soft_axes(ax: plt.Axes, *, x_end: float, y_end: float, color: str, linewidth: float) -> None:
    ax.plot([0, x_end], [0, 0], color=color, linewidth=linewidth, solid_capstyle="round", zorder=6)
    ax.plot([0, 0], [0, y_end], color=color, linewidth=linewidth, solid_capstyle="round", zorder=6)


def draw_chart(
    *,
    heights: Sequence[float],
    colors: Sequence[str],
    output: Path,
    figsize: tuple[float, float],
    dpi: int,
    transparent: bool,
    show_axis: bool,
    axis_color: str,
    background: str,
    bar_edge_color: str,
    bar_edge_width: float,
    bar_bottom_gap: float,
    axis_linewidth: float,
    scale_max: float | None,
) -> None:
    n = len(heights)
    max_height = scale_max if scale_max is not None else max(heights)
    if max_height <= 0:
        raise ValueError("scale_max must be positive")
    if max(heights) > max_height:
        raise ValueError("scale_max must be greater than or equal to the tallest bar")
    bar_width = 0.58
    x_positions = [i + 1 for i in range(n)]
    x_axis_end = n + 0.88
    y_axis_end = max_height * 1.16
    y_base = max_height * bar_bottom_gap
    rounding = min(bar_width * 0.34, max_height * 0.085)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("none" if transparent else background)
    ax.set_facecolor("none" if transparent else background)

    for x_center, height, color in zip(x_positions, heights, colors):
        add_rounded_bar(
            ax,
            x_center=x_center,
            y_base=y_base,
            height=height,
            width=bar_width,
            color=color,
            rounding=rounding,
            edge_color=bar_edge_color,
            edge_width=bar_edge_width,
        )

    if show_axis:
        add_soft_axes(ax, x_end=x_axis_end, y_end=y_axis_end, color=axis_color, linewidth=axis_linewidth)

    ax.set_xlim(-0.22, x_axis_end + 0.16)
    ax.set_ylim(-max_height * 0.035, y_axis_end + max_height * 0.045)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.04, transparent=transparent)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a clean rounded bar chart schematic with no labels or ticks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bars", type=int, default=5, help="Number of bars when --heights is not provided.")
    parser.add_argument("--heights", type=str, default=None, help="Comma-separated bar heights, e.g. 0.4,0.9,0.65.")
    parser.add_argument("--colors", type=str, default=None, help="Comma-separated bar colors. Defaults to a distinct palette.")
    parser.add_argument("--seed", type=int, default=7, help="Seed for random default heights.")
    parser.add_argument("--random-low", type=float, default=0.28, help="Lower bound for random heights.")
    parser.add_argument("--random-high", type=float, default=1.0, help="Upper bound for random heights.")
    parser.add_argument("--output", type=Path, default=Path("figures/schematic_bar_chart.svg"), help="Output path.")
    parser.add_argument("--width", type=float, default=4.2, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=2.8, help="Figure height in inches.")
    parser.add_argument("--dpi", type=int, default=240, help="DPI for raster outputs.")
    parser.add_argument("--scale-max", type=float, default=None, help="Fixed visual y-scale maximum. Defaults to tallest bar.")
    parser.add_argument("--axis-color", type=str, default="#000000", help="Color for schematic x/y axes.")
    parser.add_argument("--axis-linewidth", type=float, default=5.0, help="Line width for schematic x/y axes.")
    parser.add_argument("--bar-edge-color", type=str, default=None, help="Bar outline color. Defaults to --axis-color.")
    parser.add_argument("--bar-edge-width", type=float, default=None, help="Bar outline width. Defaults to --axis-linewidth.")
    parser.add_argument("--bar-bottom-gap", type=float, default=0.0, help="Gap between bar bottoms and the x-axis as a fraction of max height.")
    parser.add_argument("--background", type=str, default="#FFFFFF", help="Background color when not transparent.")
    parser.add_argument("--transparent", action="store_true", help="Export with transparent background.")
    parser.add_argument("--no-axis", action="store_true", help="Hide the schematic x/y axes.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        parsed_heights = parse_float_list(args.heights, option_name="--heights")
        if parsed_heights is None:
            if args.bars <= 0:
                raise ValueError("--bars must be positive")
            heights = random_heights(args.bars, args.seed, args.random_low, args.random_high)
        else:
            heights = validate_heights(parsed_heights)

        bar_edge_width = args.axis_linewidth if args.bar_edge_width is None else args.bar_edge_width
        bar_edge_color = args.axis_color if args.bar_edge_color is None else args.bar_edge_color

        if args.axis_linewidth < 0:
            raise ValueError("--axis-linewidth must be non-negative")
        if bar_edge_width < 0:
            raise ValueError("--bar-edge-width must be non-negative")
        if args.bar_bottom_gap < 0:
            raise ValueError("--bar-bottom-gap must be non-negative")
        if args.scale_max is not None and args.scale_max <= 0:
            raise ValueError("--scale-max must be positive")

        colors = parse_color_list(args.colors) or DEFAULT_PALETTE
    except (argparse.ArgumentTypeError, ValueError) as exc:
        parser.error(str(exc))

    colors = cycle_to_length(colors, len(heights))

    draw_chart(
        heights=heights,
        colors=colors,
        output=args.output,
        figsize=(args.width, args.height),
        dpi=args.dpi,
        transparent=args.transparent,
        show_axis=not args.no_axis,
        axis_color=args.axis_color,
        background=args.background,
        bar_edge_color=bar_edge_color,
        bar_edge_width=bar_edge_width,
        bar_bottom_gap=args.bar_bottom_gap,
        axis_linewidth=args.axis_linewidth,
        scale_max=args.scale_max,
    )
    print(f"Saved schematic bar chart: {args.output}")


if __name__ == "__main__":
    main()
