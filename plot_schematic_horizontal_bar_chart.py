"""Generate a minimal horizontal rounded bar chart schematic for paper figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

from plot_schematic_bar_chart import (
    BEZIER_KAPPA,
    cycle_to_length,
    parse_color_list,
    parse_float_list,
    random_heights,
    validate_heights,
)

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


DEFAULT_HORIZONTAL_PALETTE = [
    "#3d85c6",
    "#e69138",
    "#6aa84f",
    "#cc0000",
]


def add_horizontal_bar(
    ax: plt.Axes,
    *,
    x_base: float,
    y_center: float,
    value: float,
    height: float,
    color: str,
    rounding: float,
    edge_color: str,
    edge_width: float,
) -> None:
    x0 = x_base
    x1 = x_base + value
    y0 = y_center - height / 2
    y1 = y_center + height / 2
    r = min(rounding, value / 2, height / 2)
    k = BEZIER_KAPPA * r

    fill_path = MplPath(
        [
            (x0, y0),
            (x1 - r, y0),
            (x1 - r + k, y0),
            (x1, y0 + r - k),
            (x1, y0 + r),
            (x1, y1 - r),
            (x1, y1 - r + k),
            (x1 - r + k, y1),
            (x1 - r, y1),
            (x0, y1),
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
    ax.add_patch(PathPatch(fill_path, linewidth=0, facecolor=color, alpha=0.96, zorder=3))

    outline_path = MplPath(
        [
            (x0, y0),
            (x1 - r, y0),
            (x1 - r + k, y0),
            (x1, y0 + r - k),
            (x1, y0 + r),
            (x1, y1 - r),
            (x1, y1 - r + k),
            (x1 - r + k, y1),
            (x1 - r, y1),
            (x0, y1),
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
    ax.plot([0, x_end], [y_end, y_end], color=color, linewidth=linewidth, solid_capstyle="round", zorder=6)
    ax.plot([0, 0], [0, y_end], color=color, linewidth=linewidth, solid_capstyle="round", zorder=6)


def draw_horizontal_chart(
    *,
    values: Sequence[float],
    colors: Sequence[str],
    output: Path,
    figsize: tuple[float, float],
    dpi: int,
    transparent: bool,
    show_axis: bool,
    axis_color: str,
    axis_linewidth: float,
    background: str,
    bar_edge_color: str,
    bar_edge_width: float,
    scale_max: float | None,
    top_gap: float,
) -> None:
    max_value = scale_max if scale_max is not None else max(values)
    if max_value <= 0:
        raise ValueError("scale_max must be positive")
    if max(values) > max_value:
        raise ValueError("scale_max must be greater than or equal to the longest bar")

    n = len(values)
    bar_height = 0.48
    if top_gap < 0:
        raise ValueError("top_gap must be non-negative")
    y_positions = [n - top_gap - i for i in range(n)]
    x_axis_end = max_value * 1.14
    y_axis_end = n + 0.55
    rounding = min(bar_height * 0.45, max_value * 0.08)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("none" if transparent else background)
    ax.set_facecolor("none" if transparent else background)

    for y_center, value, color in zip(y_positions, values, colors):
        add_horizontal_bar(
            ax,
            x_base=0,
            y_center=y_center,
            value=value,
            height=bar_height,
            color=color,
            rounding=rounding,
            edge_color=bar_edge_color,
            edge_width=bar_edge_width,
        )

    if show_axis:
        add_soft_axes(ax, x_end=x_axis_end, y_end=y_axis_end, color=axis_color, linewidth=axis_linewidth)

    ax.set_xlim(-max_value * 0.035, x_axis_end + max_value * 0.04)
    ax.set_ylim(-0.35, y_axis_end + 0.18)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.04, transparent=transparent)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a clean horizontal rounded bar chart schematic with no labels or ticks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bars", type=int, default=4, help="Number of bars when --values is not provided. With the default 4 bars, the fourth is the mean of the first three.")
    parser.add_argument("--values", type=str, default=None, help="Comma-separated bar values, e.g. 0.8,0.45,1.0,0.62.")
    parser.add_argument("--colors", type=str, default=None, help="Comma-separated bar colors.")
    parser.add_argument("--seed", type=int, default=11, help="Seed for random default values.")
    parser.add_argument("--random-low", type=float, default=0.35, help="Lower bound for random values.")
    parser.add_argument("--random-high", type=float, default=1.0, help="Upper bound for random values.")
    parser.add_argument("--scale-max", type=float, default=None, help="Fixed visual x-scale maximum. Defaults to longest bar.")
    parser.add_argument("--output", type=Path, default=Path("figures/schematic_horizontal_bar_chart.svg"), help="Output path.")
    parser.add_argument("--width", type=float, default=4.0, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=5.0, help="Figure height in inches.")
    parser.add_argument("--dpi", type=int, default=240, help="DPI for raster outputs.")
    parser.add_argument("--axis-color", type=str, default="#000000", help="Color for schematic x/y axes.")
    parser.add_argument("--axis-linewidth", type=float, default=5.0, help="Line width for schematic x/y axes.")
    parser.add_argument("--bar-edge-color", type=str, default=None, help="Bar outline color. Defaults to --axis-color.")
    parser.add_argument("--bar-edge-width", type=float, default=None, help="Bar outline width. Defaults to --axis-linewidth.")
    parser.add_argument("--top-gap", type=float, default=0.4, help="Extra vertical gap between top bar and top axis, in bar-slot units.")
    parser.add_argument("--background", type=str, default="#FFFFFF", help="Background color when not transparent.")
    parser.add_argument("--transparent", action="store_true", help="Export with transparent background.")
    parser.add_argument("--no-axis", action="store_true", help="Hide the schematic top/left axes.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        parsed_values = parse_float_list(args.values, option_name="--values")
        if parsed_values is None:
            if args.bars <= 0:
                raise ValueError("--bars must be positive")
            if args.bars == 4:
                values = random_heights(3, args.seed, args.random_low, args.random_high)
                values.append(sum(values) / len(values))
            else:
                values = random_heights(args.bars, args.seed, args.random_low, args.random_high)
        else:
            values = validate_heights(parsed_values)

        bar_edge_width = args.axis_linewidth if args.bar_edge_width is None else args.bar_edge_width
        bar_edge_color = args.axis_color if args.bar_edge_color is None else args.bar_edge_color

        if args.axis_linewidth < 0:
            raise ValueError("--axis-linewidth must be non-negative")
        if bar_edge_width < 0:
            raise ValueError("--bar-edge-width must be non-negative")
        if args.scale_max is not None and args.scale_max <= 0:
            raise ValueError("--scale-max must be positive")
        if args.top_gap < 0:
            raise ValueError("--top-gap must be non-negative")

        colors = parse_color_list(args.colors) or DEFAULT_HORIZONTAL_PALETTE
    except (argparse.ArgumentTypeError, ValueError) as exc:
        parser.error(str(exc))

    colors = cycle_to_length(colors, len(values))
    draw_horizontal_chart(
        values=values,
        colors=colors,
        output=args.output,
        figsize=(args.width, args.height),
        dpi=args.dpi,
        transparent=args.transparent,
        show_axis=not args.no_axis,
        axis_color=args.axis_color,
        axis_linewidth=args.axis_linewidth,
        background=args.background,
        bar_edge_color=bar_edge_color,
        bar_edge_width=bar_edge_width,
        scale_max=args.scale_max,
        top_gap=args.top_gap,
    )
    print(f"Saved schematic horizontal bar chart: {args.output}")


if __name__ == "__main__":
    main()
