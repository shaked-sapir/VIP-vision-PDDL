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

Example:
    python -m benchmark.evaluation.cfm_quality_analysis \
        benchmark/data/new_experiments/blocksworld/TO=300__largest__cv5__singleFluentBranching
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ── Metrics to analyse ──────────────────────────────────────────────────────
METRICS = [
    ("pred_app_precision", "Predictive applicability precision"),
    ("pred_app_recall", "Predictive applicability recall"),
    ("pred_eff_precision", "Predicted effects precision"),
    ("pred_eff_recall", "Predicted effects recall"),
    ("solving_ratio", "Problem solving ratio"),
]



# ── Data loading ────────────────────────────────────────────────────────────

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
    ax_count.set_title(f"CFM quality progression — {metric_label}", fontsize=12, pad=8)

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


# ── Main ────────────────────────────────────────────────────────────────────

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

    experiment_root = Path(args.experiment_root)
    testing_dir = experiment_root / "testing"
    eval_dir = experiment_root / "evaluation_results"

    if not testing_dir.is_dir():
        print(f"ERROR: testing dir not found: {testing_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = eval_dir / "CFM_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

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
        sys.exit(0)

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
        )

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
