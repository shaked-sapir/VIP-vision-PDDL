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

MIN_INSTANCES_FOR_DISPLAY = 3  # skip CFM indices with fewer instances


# ── Data loading ────────────────────────────────────────────────────────────

def find_instance_dirs(testing_dir: Path) -> List[Path]:
    """Return all instance directories that contain all_solutions_metrics.json.

    Handles two layouts:
      - With inner CV:  testing/foldX_numtrajsY_gtrateZ/inner_N/
      - Without inner CV: testing/foldX_numtrajsY_gtrateZ/
    """
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

def plot_cfm_quality(
    diffs_by_index: Dict[int, List[float]],
    metric_name: str,
    metric_label: str,
    output_path: Path,
    min_instances: int = MIN_INSTANCES_FOR_DISPLAY,
) -> None:
    """Create a dual-axis plot:
      - Left Y-axis (bars):    number of instances at each CFM index
      - Right Y-axis (boxplot): distribution of differences

    A horizontal dashed line at y=0 on the right axis marks no-improvement.
    """
    # Filter to indices with enough instances
    indices = sorted(i for i, diffs in diffs_by_index.items()
                     if len(diffs) >= min_instances)
    if not indices:
        print(f"  [SKIP] {metric_name}: no CFM index has >= {min_instances} instances")
        return

    counts = [len(diffs_by_index[i]) for i in indices]
    diff_data = [diffs_by_index[i] for i in indices]
    x_positions = np.arange(len(indices))
    x_labels = [str(i + 1) for i in indices]  # 1-indexed for display

    fig, ax_bar = plt.subplots(figsize=(max(8, len(indices) * 0.8 + 2), 5))
    ax_box = ax_bar.twinx()

    # ── Bars (left axis): instance count ──
    bar_color = "#B5D4F4"
    bars = ax_bar.bar(x_positions, counts, width=0.6, color=bar_color,
                      edgecolor="#85B7EB", alpha=0.6, zorder=2, label="Instance count")
    # Label each bar
    for rect, c in zip(bars, counts):
        ax_bar.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.3,
                    str(c), ha="center", va="bottom", fontsize=8, color="#185FA5")

    ax_bar.set_ylabel("Number of instances", color="#185FA5", fontsize=11)
    ax_bar.tick_params(axis="y", labelcolor="#185FA5")
    ax_bar.set_ylim(0, max(counts) * 1.3)

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
    ax_bar.set_xlabel("CFM index i  (transition from CFM_i → CFM_{i+1})", fontsize=11)
    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels(x_labels)
    fig.suptitle(f"CFM quality progression — {metric_label}", fontsize=13, y=0.98)

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=bar_color, edgecolor="#85B7EB", alpha=0.6,
              label="Instance count (left axis)"),
        Patch(facecolor="#FCEBEB", edgecolor="#E24B4A",
              label="Diff distribution (right axis)"),
        Line2D([0], [0], marker="D", color="#E24B4A", linestyle="--",
               markerfacecolor="#E24B4A", markersize=5, label="Mean diff"),
    ]
    ax_bar.legend(handles=legend_elements, loc="upper right", fontsize=8,
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
    parser.add_argument(
        "--min-instances", type=int, default=MIN_INSTANCES_FOR_DISPLAY,
        help=f"Skip CFM indices with fewer instances (default: {MIN_INSTANCES_FOR_DISPLAY}).",
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

    # ── Per-metric analysis ──
    for metric_key, metric_label in METRICS:
        print(f"\nMetric: {metric_label}")
        diffs = compute_consecutive_diffs(multi_cfm, metric_key)
        if not diffs:
            print(f"  [SKIP] No consecutive pairs found for {metric_key}")
            continue

        output_path = output_dir / f"{metric_key}_cfm_improvement.png"
        plot_cfm_quality(
            diffs, metric_key, metric_label, output_path,
            min_instances=args.min_instances,
        )

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
