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


def plot_cfm_quality(
    diffs_by_index: Dict[int, List[float]],
    cfm_histogram: Dict[int, int],
    metric_name: str,
    metric_label: str,
    output_path: Path,
) -> None:
    """Create a dual-axis plot:
      - Left Y-axis (lines): two count trends —
          (a) instances with >= i+2 CFMs (sample size for the boxplot at that x)
          (b) instances with exactly N total CFMs (histogram)
      - Right Y-axis (boxplot): distribution of consecutive differences

    A horizontal dashed line at y=0 on the right axis marks no-improvement.
    """
    # Use the full range of indices present in diffs (no min_instances cutoff).
    indices = sorted(diffs_by_index.keys())
    if not indices:
        print(f"  [SKIP] {metric_name}: no consecutive pairs found")
        return

    # "At least" counts: how many instances contributed to each transition's boxplot.
    at_least_counts = [len(diffs_by_index[i]) for i in indices]
    diff_data = [diffs_by_index[i] for i in indices]
    x_positions = np.arange(len(indices))
    x_labels = [str(i + 1) for i in indices]  # 1-indexed for display

    # "Exact" counts: instances whose total CFM count == i+2
    # (i.e., their LAST transition is at index i).
    exact_counts = [cfm_histogram.get(i + 2, 0) for i in indices]

    fig, ax_count = plt.subplots(figsize=(max(8, len(indices) * 0.8 + 2), 5))
    ax_box = ax_count.twinx()

    # ── Lines (left axis): instance counts ──
    # "At least" trend
    ax_count.plot(x_positions, at_least_counts, color="#185FA5", linewidth=1.8,
                  marker="o", markersize=5, zorder=5, label="Instances with ≥ N CFMs")
    for x, c in zip(x_positions, at_least_counts):
        ax_count.annotate(str(c), (x, c), textcoords="offset points",
                          xytext=(0, 8), ha="center", fontsize=7, color="#185FA5")

    # "Exact" histogram trend
    ax_count.plot(x_positions, exact_counts, color="#2EA043", linewidth=1.8,
                  marker="s", markersize=5, zorder=5, label="Instances with exactly N CFMs")
    for x, c in zip(x_positions, exact_counts):
        if c > 0:
            ax_count.annotate(str(c), (x, c), textcoords="offset points",
                              xytext=(0, -12), ha="center", fontsize=7, color="#2EA043")

    ax_count.set_ylabel("Number of instances", fontsize=11)
    all_counts = at_least_counts + exact_counts
    ax_count.set_ylim(0, max(all_counts) * 1.3 if all_counts else 1)

    # ── Boxplot (right axis): difference distribution ──
    bp = ax_box.boxplot(
        diff_data,
        positions=x_positions,
        widths=0.35,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="#E24B4A", markeredgecolor="#E24B4A",
                       markersize=5),
        medianprops=dict(color="#A32D2D", linewidth=1.5),
        boxprops=dict(facecolor="#FCEBEB", edgecolor="#E24B4A", linewidth=1),
        whiskerprops=dict(color="#A32D2D", linewidth=1),
        capprops=dict(color="#A32D2D", linewidth=1),
        flierprops=dict(marker="o", markerfacecolor="#F09595", markeredgecolor="#E24B4A",
                        markersize=4),
        zorder=3,
    )

    # Connect means with a line
    means = [np.mean(d) for d in diff_data]
    ax_box.plot(x_positions, means, color="#E24B4A", linewidth=1.5, linestyle="--",
                alpha=0.7, zorder=4)

    ax_box.axhline(y=0, color="#888780", linestyle=":", linewidth=1, zorder=1)
    ax_box.set_ylabel(f"Diff: CFM(i+1) − CFM(i)  [{metric_label}]",
                      color="#A32D2D", fontsize=10)
    ax_box.tick_params(axis="y", labelcolor="#A32D2D")

    # ── Common ──
    ax_count.set_xlabel("CFM index i  (transition from CFM_i → CFM_{i+1})", fontsize=11)
    ax_count.set_xticks(x_positions)
    ax_count.set_xticklabels(x_labels)
    fig.suptitle(f"CFM quality progression — {metric_label}", fontsize=13, y=0.98)

    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color="#185FA5", marker="o", markersize=5,
               label="Instances with ≥ N CFMs (left)"),
        Line2D([0], [0], color="#2EA043", marker="s", markersize=5,
               label="Instances with exactly N CFMs (left)"),
        Patch(facecolor="#FCEBEB", edgecolor="#E24B4A",
              label="Diff distribution (right)"),
        Line2D([0], [0], marker="D", color="#E24B4A", linestyle="--",
               markerfacecolor="#E24B4A", markersize=5, label="Mean diff"),
    ]
    ax_count.legend(handles=legend_elements, loc="upper right", fontsize=7,
                    framealpha=0.9)

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
        diffs = compute_consecutive_diffs(multi_cfm, metric_key)
        if not diffs:
            print(f"  [SKIP] No consecutive pairs found for {metric_key}")
            continue

        output_path = output_dir / f"{metric_key}_cfm_improvement.png"
        plot_cfm_quality(
            diffs, cfm_histogram, metric_key, metric_label, output_path,
        )

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
