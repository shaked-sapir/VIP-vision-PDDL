"""Analyze whether later Conflict-Free Models (CFMs) improve over earlier ones.

For each consecutive pair (CFM_i, CFM_{i+1}) we compute the *difference*
    diff = metric(CFM_{i+1}) - metric(CFM_i)
and aggregate across all instances that have at least i+2 models.

NOTE: we use difference rather than ratio because solving_ratio is often
exactly 0 or 1, causing division-by-zero.  For the four predictive-power
metrics (continuous in [0,1]) ratio would also work — swap to
    ratio = metric(CFM_{i+1}) / metric(CFM_i)
if preferred.

Usage:
    python -m benchmark.evaluation.cfm_quality_analysis <experiment_root>

Outputs per-metric improvement/trend PNGs plus summary figures:
    all_trends_summary.png, precision_trends_summary.png, recall_trends_summary.png

Example:
    python -m benchmark.evaluation.cfm_quality_analysis \
        benchmark/running_results/blocksworld/TO=300__largest__cv5__singleFluentBranching
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


# ── Metrics to analyse ──────────────────────────────────────────────────────
METRICS = [
    ("pred_app_precision", "Predictive applicability precision"),
    ("pred_app_recall", "Predictive applicability recall"),
    ("pred_eff_precision", "Predicted effects precision"),
    ("pred_eff_recall", "Predicted effects recall"),
    ("solving_ratio", "Problem solving ratio"),
]

# Syntactic (non-predictive-power) metrics from all_solutions_metrics.json
SYNTACTIC_PRECISION_METRICS = [
    ("precision_precs_pos", "Preconditions precision (+)"),
    ("precision_precs_neg", "Preconditions precision (−)"),
    ("precision_eff_pos", "Effects precision (+)"),
    ("precision_eff_neg", "Effects precision (−)"),
    ("precision_overall", "Overall precision"),
]

SYNTACTIC_RECALL_METRICS = [
    ("recall_precs_pos", "Preconditions recall (+)"),
    ("recall_precs_neg", "Preconditions recall (−)"),
    ("recall_eff_pos", "Effects recall (+)"),
    ("recall_eff_neg", "Effects recall (−)"),
    ("recall_overall", "Overall recall"),
]



# ── Data loading ────────────────────────────────────────────────────────────

def load_domain_name(experiment_root: Path) -> str:
    """Load domain name from run_params.json, falling back to parent directory name."""
    run_params_path = experiment_root / "evaluation_results" / "run_params.json"
    if run_params_path.exists():
        with open(run_params_path) as f:
            run_params = json.load(f)
        for key in ("display_domain_name", "domain_key"):
            name = run_params.get(key)
            if name:
                return str(name)
    return experiment_root.parent.name


def _title_with_domain(domain_name: str, title: str) -> str:
    """Prefix a plot title with the uppercased domain name."""
    return f"{domain_name.upper()} — {title}"


def find_instance_dirs(testing_dir: Path) -> List[Path]:
    """Return all instance directories that contain all_solutions_metrics.json."""
    instances = []
    for metrics_file in testing_dir.rglob("all_solutions_metrics.json"):
        instances.append(metrics_file.parent)
    return sorted(instances)


def load_cfm_metrics(instance_dir: Path) -> List[Dict]:
    """Load all_solutions_metrics.json and return entries sorted by solution_index.

    Filters out solution_index == -1 (the "selected" model, not a numbered CFM).
    """
    metrics_path = instance_dir / "all_solutions_metrics.json"
    with open(metrics_path) as f:
        data = json.load(f)

    # Keep only numbered CFMs (index >= 0), sorted by index
    cfms = [d for d in data if d.get("solution_index", -1) >= 0]
    cfms.sort(key=lambda d: d["solution_index"])
    return cfms


# ── Analysis ────────────────────────────────────────────────────────────────

def compute_absolute_values(
    all_instances: List[List[Dict]],
    metric_key: str,
) -> Dict[int, List[float]]:
    """For each CFM index i, collect the raw metric value across all instances
    that have at least i+1 models.

    Returns: {i: [value_instance_1, value_instance_2, ...]}
    """
    values_by_index: Dict[int, List[float]] = defaultdict(list)

    for cfms in all_instances:
        for i, cfm in enumerate(cfms):
            val = cfm.get(metric_key)
            if val is not None:
                values_by_index[i].append(val)

    return dict(values_by_index)


def compute_consecutive_diffs(
    all_instances: List[List[Dict]],
    metric_key: str,
) -> Dict[int, List[float]]:
    """For each CFM index i, collect diff = metric(CFM_{i+1}) - metric(CFM_i)
    across all instances that have at least i+2 models.

    Returns: {i: [diff_instance_1, diff_instance_2, ...]}
    """
    diffs_by_index: Dict[int, List[float]] = defaultdict(list)

    for cfms in all_instances:
        if len(cfms) < 2:
            continue
        for i in range(len(cfms) - 1):
            val_curr = cfms[i].get(metric_key)
            val_next = cfms[i + 1].get(metric_key)
            if val_curr is None or val_next is None:
                continue
            diffs_by_index[i].append(val_next - val_curr)

    return dict(diffs_by_index)


# ── Plotting ────────────────────────────────────────────────────────────────

def _cfm_count_histogram(all_instances: List[List[Dict]]) -> Dict[int, int]:
    """Return {total_cfm_count: number_of_instances_with_that_count}."""
    hist: Dict[int, int] = defaultdict(int)
    for cfms in all_instances:
        hist[len(cfms)] += 1
    return dict(hist)


def _make_boxplot(ax, data, positions, color_main, color_fill, color_flier):
    """Draw a boxplot with means connected by a dashed line.

    Returns the boxplot artist dict.
    """
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.4,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor=color_main, markeredgecolor=color_main,
                       markersize=4.5),
        medianprops=dict(color=color_main, linewidth=1.5),
        boxprops=dict(facecolor=color_fill, edgecolor=color_main, linewidth=1),
        whiskerprops=dict(color=color_main, linewidth=1),
        capprops=dict(color=color_main, linewidth=1),
        flierprops=dict(marker="o", markerfacecolor=color_flier, markeredgecolor=color_main,
                        markersize=3.5),
        zorder=3,
    )
    means = [np.mean(d) if d else float("nan") for d in data]
    ax.plot(positions, means, color=color_main, linewidth=1.2, linestyle="--",
            alpha=0.6, zorder=4)
    return bp


def plot_cfm_quality(
    abs_by_index: Dict[int, List[float]],
    diffs_by_index: Dict[int, List[float]],
    cfm_histogram: Dict[int, int],
    metric_name: str,
    metric_label: str,
    output_path: Path,
    domain_name: str,
) -> None:
    """Three-panel stacked figure (shared x-axis, no twin axes):

    1. Top strip  – instance-count lines (≥ N CFMs, exactly N CFMs).
    2. Middle     – boxplots of absolute metric values per CFM index.
    3. Bottom     – boxplots of consecutive diffs (CFM_i − CFM_{i-1}).
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    all_indices = sorted(set(abs_by_index.keys()) | set(diffs_by_index.keys()))
    if not all_indices:
        print(f"  [SKIP] {metric_name}: no data found")
        return

    x_positions = np.arange(len(all_indices))
    x_labels = [str(i + 1) for i in all_indices]  # 1-indexed

    # ── Prepare data ──
    abs_data = [abs_by_index.get(i, []) for i in all_indices]
    abs_counts = [len(abs_by_index.get(i, [])) for i in all_indices]

    # Shift diffs: key j stores CFM_{j+1}−CFM_j → plot at index j+1.
    shifted_diffs: Dict[int, List[float]] = {j + 1: v for j, v in diffs_by_index.items()}
    diff_data = [shifted_diffs.get(i, []) for i in all_indices]

    # Exact-count histogram: instances whose total CFM count == i+1.
    exact_counts = [cfm_histogram.get(i + 1, 0) for i in all_indices]

    # ── Figure layout ──
    fig_width = max(8, len(all_indices) * 0.75 + 2)
    fig, (ax_count, ax_abs, ax_diff) = plt.subplots(
        3, 1,
        figsize=(fig_width, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2.5, 2.5], "hspace": 0.08},
    )

    # ============== TOP STRIP — instance counts ==============
    ax_count.plot(x_positions, abs_counts, color="#185FA5", linewidth=1.8,
                  marker="o", markersize=5, zorder=5, label="Instances with ≥ N CFMs")
    for x, c in zip(x_positions, abs_counts):
        ax_count.annotate(str(c), (x, c), textcoords="offset points",
                          xytext=(0, 7), ha="center", fontsize=6.5, color="#185FA5")

    ax_count.plot(x_positions, exact_counts, color="#2EA043", linewidth=1.8,
                  marker="s", markersize=5, zorder=5, label="Instances with exactly N CFMs")
    for x, c in zip(x_positions, exact_counts):
        if c > 0:
            ax_count.annotate(str(c), (x, c), textcoords="offset points",
                              xytext=(0, -11), ha="center", fontsize=6.5, color="#2EA043")

    ax_count.set_ylabel("# instances", fontsize=9)
    all_counts = abs_counts + exact_counts
    ax_count.set_ylim(0, max(all_counts) * 1.4 if all_counts else 1)
    ax_count.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax_count.set_title(
        _title_with_domain(domain_name, f"CFM quality progression — {metric_label}"),
        fontsize=12,
        pad=8,
    )

    # ============== MIDDLE — absolute metric values ==============
    _make_boxplot(ax_abs, abs_data, x_positions,
                  color_main="#1B6DB5", color_fill="#D6E9F8", color_flier="#7CBAE5")

    ax_abs.set_ylabel(metric_label, fontsize=10)

    legend_abs = [
        Patch(facecolor="#D6E9F8", edgecolor="#1B6DB5", label="Value distribution"),
        Line2D([0], [0], marker="D", color="#1B6DB5", linestyle="--",
               markerfacecolor="#1B6DB5", markersize=4.5, label="Mean value"),
    ]
    ax_abs.legend(handles=legend_abs, loc="lower left", fontsize=7, framealpha=0.9)

    # ============== BOTTOM — diffs ==============
    _make_boxplot(ax_diff, diff_data, x_positions,
                  color_main="#A32D2D", color_fill="#FCEBEB", color_flier="#F09595")

    ax_diff.axhline(y=0, color="#888780", linestyle=":", linewidth=0.8, zorder=1)
    ax_diff.set_ylabel(f"Diff (CFM_i − CFM_{{i−1}})", fontsize=10)

    legend_diff = [
        Patch(facecolor="#FCEBEB", edgecolor="#A32D2D", label="Diff distribution"),
        Line2D([0], [0], marker="D", color="#A32D2D", linestyle="--",
               markerfacecolor="#A32D2D", markersize=4.5, label="Mean diff"),
    ]
    ax_diff.legend(handles=legend_diff, loc="lower left", fontsize=7, framealpha=0.9)

    # ── Shared x-axis ──
    ax_diff.set_xlabel("CFM index", fontsize=11)
    ax_diff.set_xticks(x_positions)
    ax_diff.set_xticklabels(x_labels)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Trend plots (mean ± std with forward-fill padding) ─────────────────────

def load_fluent_patch_counts(instance_dir: Path) -> List[Dict]:
    """Load conflict_free_solutions_log.json and return entries sorted by index.

    Each entry has at least 'index' and 'fluent_patch_count'.
    Returns empty list if the file doesn't exist.
    """
    log_path = instance_dir / "conflict_free_solutions_log.json"
    if not log_path.exists():
        return []
    with open(log_path) as f:
        data = json.load(f)
    entries = [d for d in data if d.get("index", -1) >= 0]
    entries.sort(key=lambda d: d["index"])
    return entries


def compute_padded_trend(
    instance_dirs: List[Path],
    metric_key: str,
    source: str = "all_solutions_metrics",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Compute mean ± std of a metric across instances with forward-fill padding.

    For each instance, loads the per-solution metric values, then forward-fills
    (carries the last value) up to the global max solution_index. After padding,
    computes mean and std at each solution_index across all instances.

    Args:
        instance_dirs: Directories containing the metric files.
        metric_key: The metric to extract (e.g. 'solving_ratio', 'fluent_patch_count').
        source: Which file to load from:
            'all_solutions_metrics' → all_solutions_metrics.json (keyed by solution_index)
            'conflict_free_solutions_log' → conflict_free_solutions_log.json (keyed by index)

    Returns:
        (solution_ids, means, stds, n_instances) where solution_ids is 0..max_id.
    """
    all_series: List[List[float]] = []

    for d in instance_dirs:
        if source == "conflict_free_solutions_log":
            entries = load_fluent_patch_counts(d)
            if not entries:
                continue
            series = [(e["index"], e[metric_key]) for e in entries if metric_key in e]
        else:
            cfms = load_cfm_metrics(d)
            if not cfms:
                continue
            series = [(c["solution_index"], c[metric_key]) for c in cfms if metric_key in c]

        if series:
            all_series.append(series)

    if not all_series:
        return np.array([]), np.array([]), np.array([]), 0

    # Determine global max solution_index
    max_id = max(idx for series in all_series for idx, _ in series)

    # Forward-fill each instance to max_id
    padded = np.full((len(all_series), max_id + 1), np.nan)
    for i, series in enumerate(all_series):
        for idx, val in series:
            padded[i, idx] = val
        # Forward-fill: carry last known value
        last_val = np.nan
        for j in range(max_id + 1):
            if not np.isnan(padded[i, j]):
                last_val = padded[i, j]
            else:
                padded[i, j] = last_val

    # Drop any columns where all instances are still NaN (shouldn't happen after ffill)
    valid_mask = ~np.all(np.isnan(padded), axis=0)
    padded = padded[:, valid_mask]
    solution_ids = np.arange(max_id + 1)[valid_mask]

    means = np.nanmean(padded, axis=0)
    stds = np.nanstd(padded, axis=0)

    return solution_ids, means, stds, len(all_series)


def _draw_trend_on_ax(
    ax,
    solution_ids: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    n_instances: int,
    metric_label: str,
    title: str,
    expected_monotone: str | None = None,
    value_bounds: Tuple[float, float] | None = None,
    compact: bool = False,
    metric_key: str = "",
    legend: bool = True,
) -> None:
    """Draw a mean ± std trend on an existing axes.

    Args:
        ax: Matplotlib axes to draw on.
        solution_ids: X-axis values (solution indices).
        means: Mean values at each index.
        stds: Standard deviation at each index.
        n_instances: Number of instances used.
        metric_label: Y-axis label.
        title: Axes title.
        expected_monotone: If 'non_increasing', warn when violated.
        value_bounds: (lo, hi) — clamp the shading band to this range.
        compact: If True, use smaller fonts/markers for the summary figure.
    """
    marker_size = 3 if compact else 4
    line_width = 1.5 if compact else 2
    title_size = 9 if compact else 12
    label_size = 8 if compact else 11
    legend_size = 7 if compact else 9
    note_size = 6 if compact else 7

    ax.plot(solution_ids, means, color="#1B6DB5", linewidth=line_width,
            marker="o", markersize=marker_size, label="Mean", zorder=5)

    shade_lo = means - stds
    shade_hi = means + stds
    if value_bounds is not None:
        shade_lo = np.clip(shade_lo, value_bounds[0], value_bounds[1])
        shade_hi = np.clip(shade_hi, value_bounds[0], value_bounds[1])

    ax.fill_between(solution_ids, shade_lo, shade_hi,
                    color="#1B6DB5", alpha=0.2, label="± 1 std", zorder=3)

    # Check monotonicity if expected
    if expected_monotone == "non_increasing":
        violations = []
        for i in range(1, len(means)):
            if means[i] > means[i - 1] + 1e-9:
                violations.append(int(solution_ids[i]))
        if violations:
            ax.set_title(
                f"{title}\n⚠ Non-increasing violated at: {violations}",
                fontsize=title_size, color="#A32D2D",
            )
        else:
            ax.set_title(f"{title}\n✓ Monotonic non-increasing",
                         fontsize=title_size)
    else:
        ax.set_title(title, fontsize=title_size)

    ax.set_xlabel("Solution index (CFM)", fontsize=label_size)
    ax.set_ylabel(metric_label, fontsize=label_size)
    if legend:
        ax.legend(loc="best", fontsize=legend_size)
    ax.text(
        0.99, 0.01,
        f"n = {n_instances} instances (forward-fill padded)",
        transform=ax.transAxes, fontsize=note_size, ha="right", va="bottom",
        color="#888888",
    )

    _apply_axis_scaling(ax, metric_key)


# Metrics that are bounded in [0, 1]
_BOUNDED_METRICS = {
    "pred_app_precision", "pred_app_recall",
    "pred_eff_precision", "pred_eff_recall",
    "solving_ratio",
    *(key for key, _ in SYNTACTIC_PRECISION_METRICS),
    *(key for key, _ in SYNTACTIC_RECALL_METRICS),
}


# Bounded metrics that share one fixed [0, 1] y-scale so panels are directly
# comparable. solving_ratio is excluded (it swings far more than the others and
# would flatten them); fluent_patch_count is unbounded and keeps its own scale
# so its monotonic-decrease shape stays readable.
_SHARED_SCALE_METRICS = _BOUNDED_METRICS - {"solving_ratio"}

# Tiny headroom above 1.0 so the y=1.0 marker/line sits just below the top spine
# (border above it) without ever drawing a tick label greater than 1.0.
_TOP_HEADROOM = 0.02


def _apply_axis_scaling(ax, metric_key: str) -> None:
    """Force integer x-ticks and normalize the y-axis for bounded metrics.

    - X-axis: solution indices are integers, so only integer ticks are shown.
    - Y-axis (bounded metrics): capped at 1.0 with a hair of headroom and no
      tick label above 1.0. Metrics in ``_SHARED_SCALE_METRICS`` also get a
      fixed [0, 1] bottom so those panels share one scale; ``solving_ratio``
      keeps an auto bottom; unbounded metrics are left untouched.
    """
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if metric_key in _SHARED_SCALE_METRICS:
        ax.set_ylim(0.0, 1.0 + _TOP_HEADROOM)
    elif metric_key == "solving_ratio":
        ax.set_ylim(bottom=0.0, top=1.0 + _TOP_HEADROOM)

    if metric_key in _BOUNDED_METRICS:
        # Drop any tick label above 1.0 (keeps the 1.0 tick, border sits above it).
        ax.set_yticks([t for t in ax.get_yticks() if t <= 1.0 + 1e-9])


# Each entry: (metric_key, metric_label, source, expected_monotone)
TrendSpec = Tuple[str, str, str, str | None]


def plot_trend_with_shading(
    solution_ids: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    n_instances: int,
    metric_label: str,
    title: str,
    output_path: Path,
    expected_monotone: str | None = None,
    metric_key: str = "",
) -> None:
    """Plot mean ± std trend line with shaded region.

    Args:
        solution_ids: X-axis values (solution indices).
        means: Mean values at each index.
        stds: Standard deviation at each index.
        n_instances: Number of instances used (shown in subtitle).
        metric_label: Y-axis label.
        title: Plot title.
        output_path: Where to save the PNG.
        expected_monotone: If 'non_increasing', warn when violated.
        metric_key: Used to determine if shading should be clamped to [0, 1].
    """
    if len(solution_ids) == 0:
        print(f"  [SKIP] {title}: no data")
        return

    bounds = (0.0, 1.0) if metric_key in _BOUNDED_METRICS else None

    fig, ax = plt.subplots(figsize=(10, 5))
    _draw_trend_on_ax(
        ax, solution_ids, means, stds, n_instances,
        metric_label, title, expected_monotone, value_bounds=bounds,
        metric_key=metric_key,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_metrics_trends_summary(
    instance_dirs: List[Path],
    trend_specs: List[TrendSpec],
    suptitle: str,
    output_path: Path,
    domain_name: str,
) -> None:
    """Single figure with multiple padded mean ± std trend subplots."""
    trend_data: List[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray, int, str | None]] = []

    for metric_key, metric_label, source, expected_monotone in trend_specs:
        sol_ids, means, stds, n_inst = compute_padded_trend(
            instance_dirs, metric_key, source=source,
        )
        if len(sol_ids) > 0:
            trend_data.append(
                (metric_key, metric_label, sol_ids, means, stds, n_inst, expected_monotone)
            )

    if not trend_data:
        print(f"  [SKIP] No trend data for {output_path.name}")
        return

    n_plots = len(trend_data)
    n_cols = 2
    n_rows = (n_plots + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = np.atleast_2d(axes)

    for idx, (metric_key, metric_label, sol_ids, means, stds, n_inst, monotone) in enumerate(trend_data):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        bounds = (0.0, 1.0) if metric_key in _BOUNDED_METRICS else None
        _draw_trend_on_ax(
            ax, sol_ids, means, stds, n_inst,
            metric_label=metric_label,
            title=_title_with_domain(domain_name, metric_label),
            expected_monotone=monotone,
            value_bounds=bounds,
            compact=True,
            metric_key=metric_key,
            legend=False,
        )

    for idx in range(n_plots, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(_title_with_domain(domain_name, suptitle), fontsize=13, y=1.01)
    fig.tight_layout()

    # Single shared legend (Mean / ± 1 std) at the top-left of the whole figure,
    # since every panel uses identical styling.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper left",
                   bbox_to_anchor=(0.0, 1.0), fontsize=9, framealpha=0.9)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_all_trends_summary(
    instance_dirs: List[Path],
    metrics: List[Tuple[str, str]],
    n_instances_hint: int,
    output_path: Path,
    domain_name: str,
) -> None:
    """Single figure with predictive metrics + fluent_patch_count trends."""
    trend_specs: List[TrendSpec] = [
        (metric_key, metric_label, "all_solutions_metrics", None)
        for metric_key, metric_label in metrics
    ]
    trend_specs.append(
        ("fluent_patch_count", "Fluent patch count", "conflict_free_solutions_log", "non_increasing")
    )
    plot_metrics_trends_summary(
        instance_dirs,
        trend_specs,
        suptitle="CFM quality trends — all metrics (padded mean ± std)",
        output_path=output_path,
        domain_name=domain_name,
    )


# ── Main ────────────────────────────────────────────────────────────────────

def generate_cfm_quality_analysis(experiment_root: Path) -> Optional[Path]:
    """Generate the CFM-quality plots for a completed experiment directory.

    Reads <experiment_root>/testing/ and writes PNGs to
    <experiment_root>/evaluation_results/CFM_quality/.

    Args:
        experiment_root: Experiment dir (contains testing/ and evaluation_results/).

    Returns:
        The output dir when plots were written, or None when there is nothing
        to plot (no instance has >= 2 CFMs).

    Raises:
        FileNotFoundError: If no testing/ directory exists.
    """
    testing_dir = experiment_root / "testing"
    eval_dir = experiment_root / "evaluation_results"

    if not testing_dir.is_dir():
        raise FileNotFoundError(f"testing dir not found: {testing_dir}")

    output_dir = eval_dir / "CFM_quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    domain_name = load_domain_name(experiment_root)

    # ── Load data ──
    instance_dirs = find_instance_dirs(testing_dir)
    print(f"Found {len(instance_dirs)} instance directories")

    all_instances: List[List[Dict]] = []
    for d in instance_dirs:
        cfms = load_cfm_metrics(d)
        all_instances.append(cfms)

    multi_cfm = [inst for inst in all_instances if len(inst) >= 2]
    print(f"Instances with >= 2 CFMs: {len(multi_cfm)}")

    if not multi_cfm:
        print("No instances with multiple CFMs found. Nothing to plot.")
        return None

    # Histogram: how many instances have exactly N CFMs
    cfm_histogram = _cfm_count_histogram(all_instances)
    print(f"CFM count distribution: {dict(sorted(cfm_histogram.items()))}")

    # ── Per-metric analysis ──
    for metric_key, metric_label in METRICS:
        print(f"\nMetric: {metric_label}")
        abs_vals = compute_absolute_values(all_instances, metric_key)
        diffs = compute_consecutive_diffs(multi_cfm, metric_key)
        if not abs_vals and not diffs:
            print(f"  [SKIP] No data found for {metric_key}")
            continue

        output_path = output_dir / f"{metric_key}_cfm_improvement.png"
        plot_cfm_quality(
            abs_vals, diffs, cfm_histogram, metric_key, metric_label, output_path,
            domain_name,
        )

    # ── Trend plots (mean ± std with forward-fill padding) ──
    print("\n--- Trend plots (padded mean ± std) ---")

    # Per-metric trend plots (from all_solutions_metrics.json)
    for metric_key, metric_label in METRICS:
        sol_ids, means, stds, n_inst = compute_padded_trend(
            instance_dirs, metric_key, source="all_solutions_metrics",
        )
        plot_trend_with_shading(
            sol_ids, means, stds, n_inst,
            metric_label=metric_label,
            title=_title_with_domain(
                domain_name, f"{metric_label} vs. solution index (padded mean ± std)"
            ),
            output_path=output_dir / f"{metric_key}_trend.png",
            metric_key=metric_key,
        )

    # Fluent patch count trend (sanity — should be non-increasing)
    sol_ids, means, stds, n_inst = compute_padded_trend(
        instance_dirs, "fluent_patch_count", source="conflict_free_solutions_log",
    )
    plot_trend_with_shading(
        sol_ids, means, stds, n_inst,
        metric_label="Fluent patch count",
        title=_title_with_domain(
            domain_name, "Fluent patch count vs. solution index (padded mean ± std)"
        ),
        output_path=output_dir / "fluent_patch_count_trend.png",
        expected_monotone="non_increasing",
        metric_key="fluent_patch_count",
    )

    # Summary: all trends in one figure
    plot_all_trends_summary(
        instance_dirs, METRICS, len(instance_dirs),
        output_path=output_dir / "all_trends_summary.png",
        domain_name=domain_name,
    )

    plot_metrics_trends_summary(
        instance_dirs,
        [
            (metric_key, metric_label, "all_solutions_metrics", None)
            for metric_key, metric_label in SYNTACTIC_PRECISION_METRICS
        ],
        suptitle="Syntactic precision trends (padded mean ± std)",
        output_path=output_dir / "precision_trends_summary.png",
        domain_name=domain_name,
    )

    plot_metrics_trends_summary(
        instance_dirs,
        [
            (metric_key, metric_label, "all_solutions_metrics", None)
            for metric_key, metric_label in SYNTACTIC_RECALL_METRICS
        ],
        suptitle="Syntactic recall trends (padded mean ± std)",
        output_path=output_dir / "recall_trends_summary.png",
        domain_name=domain_name,
    )

    print(f"\nAll plots saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CFM quality progression across conflict-search instances."
    )
    parser.add_argument(
        "experiment_root",
        type=str,
        help="Path to experiment config dir (contains testing/ and evaluation_results/).",
    )
    args = parser.parse_args()

    try:
        generate_cfm_quality_analysis(Path(args.experiment_root))
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
