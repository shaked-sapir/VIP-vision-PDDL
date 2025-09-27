import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score

from .interactive_plot import plot_pr_interactive
from .static_plot import plot_pr_static_with_thresholds


def evaluate_micro_average_pr_curve(all_results, output_dir):
    """
    Evaluate micro-averaged PR curve over all predicate instances.
    - Save static PR curve with threshold annotations
    - Save interactive HTML plot

    Parameters:
        all_results: Dict[predicate_grounding] = {"<pred_name>": {"scores": [...], "labels": [...]}}
        output_dir: Directory to save output
    """
    all_scores = []
    all_labels = []

    for pred, data in all_results.items():
        all_scores.extend(data["scores"])
        all_labels.extend(data["labels"])

    precision_micro, recall_micro, thresholds_micro = precision_recall_curve(all_labels, all_scores)
    ap_micro = average_precision_score(all_labels, all_scores)

    title_micro = "Micro-Averaged Precision-Recall Curve"
    plot_pr_static_with_thresholds(
        precision_micro, recall_micro, thresholds_micro, ap_micro,
        title=title_micro,
        save_path=f"{output_dir}/pr_curve_micro.png"
    )

    plot_pr_interactive(
        precision_micro, recall_micro, thresholds_micro, ap_micro,
        title=title_micro,
        save_path=f"{output_dir}/pr_curve_micro_interactive.html"
    )
    print(f"✅ Saved micro PR: image + interactive")

    # Save PR data to Excel
    thresholds_complete = list(thresholds_micro) + [np.nan]  # Add threshold for first
    df_pr = pd.DataFrame({
        "precision": precision_micro,
        "recall": recall_micro,
        "threshold": thresholds_complete
    })
    df_pr.to_excel(f"{output_dir}/pr_data_micro.xlsx", index=False)


def evaluate_per_predicate_pr_curve(all_results, output_dir="pr_output"):
    """
    Evaluate per-predicate PR curve over all predicate instances.
    - Save static PR curve with threshold annotations
    - Save interactive HTML plot

    Parameters:
        all_results: Dict[predicate_grounding] = {"<pred_name>": {"scores": [...], "labels": [...]}}
        output_dir: Directory to save output
    """
    type_to_data = defaultdict(lambda: {"scores": [], "labels": []})
    summary = []

    for pred, data in all_results.items():
        pred_type = pred.split("(")[0]
        type_to_data[pred_type]["scores"].extend(data["scores"])
        type_to_data[pred_type]["labels"].extend(data["labels"])

    for pred_type, data in type_to_data.items():
        y_scores = data["scores"]
        y_true = data["labels"]

        if sum(y_true) == 0:
            print(f"⚠️ Skipping {pred_type} (no positives)")
            continue

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

        # Save static + interactive
        title = f"PR Curve for Predicate: {pred_type}"
        plot_pr_static_with_thresholds(
            precision, recall, thresholds, ap,
            title=title,
            save_path=f"{output_dir}/pr_curve_{pred_type}.png"
        )
        plot_pr_interactive(
            precision, recall, thresholds, ap,
            title=title,
            save_path=f"{output_dir}/pr_curve_{pred_type}_interactive.html"
        )

        # Save PR data to Excel
        thresholds_complete = list(thresholds) + [np.nan]  # Add threshold for first
        df_pr = pd.DataFrame({
            "precision": precision,
            "recall": recall,
            "threshold": thresholds_complete
        })
        df_pr.to_excel(f"{output_dir}/pr_data_{pred_type}.xlsx", index=False)

        print(f"📈 Saved PR for '{pred_type}': image + interactive")

        summary.append({
            "predicate_type": pred_type,
            "average_precision": ap,
            "positives": sum(y_true),
            "negatives": len(y_true) - sum(y_true)
        })

    df_summary = pd.DataFrame(summary).sort_values("average_precision", ascending=False)
    df_summary.to_excel(f"{output_dir}/ap_summary_per_predicate_type.xlsx", index=False)
    print(f"📋 Summary saved: ap_summary_per_predicate_type.xlsx")


def evaluate_all_pr_curves(all_results, output_dir="pr_output"):
    """
    Evaluate PR curves:
    1. Per predicate type
    2. Micro-averaged over all predicate instances
    For each:
    - Save static PR curve with threshold annotations
    - Save interactive HTML plot

    Parameters:
        all_results: Dict[predicate_grounding] = {"<pred_name>": {"scores": [...], "labels": [...]}}
        output_dir: Directory to save output
    """
    os.makedirs(output_dir, exist_ok=True)

    evaluate_micro_average_pr_curve(all_results, output_dir)
    evaluate_per_predicate_pr_curve(all_results, output_dir)


#TODO: this function has errors in its computations, fix them
def extract_confidence_thresholds_from_pr_values(precision, recall, thresholds, min_precision=0.99, min_recall=0.99):
    """
    Extract two thresholds:
    - High-confidence positive threshold: maximize precision ≥ min_precision
    - High-confidence negative threshold: maximize recall ≥ min_recall

    Returns:
        (t_low, t_high)
    """
    # thresholds array should be of len = len(precision) - 1, if not - this is because it has an extra NaN at the end
    if len(thresholds) != len(precision) - 1:
        thresholds = thresholds[:-1]
    thresholds = np.array(thresholds)

    # Find t_high: max threshold where precision ≥ min_precision
    valid_high_indices = np.where(precision >= min_precision)[0]
    print(valid_high_indices)

    t_high = thresholds[valid_high_indices[0]] if len(valid_high_indices) > 0 else 1.0  # fallback to max

    # Find t_low: min threshold where recall ≥ min_recall
    valid_low_indices = np.where(recall >= min_recall)[0]
    print(valid_low_indices)
    t_low = thresholds[valid_low_indices[-1]] if len(valid_low_indices) > 0 else 0.0  # fallback to min

    return t_low, t_high


def extract_confidence_thresholds_from_scores(y_true, y_scores, min_precision=0.99, min_recall=0.99):
    """
    Extract two thresholds:
    - High-confidence positive threshold: maximize precision ≥ min_precision
    - High-confidence negative threshold: maximize recall ≥ min_recall

    Returns:
        (t_low, t_high)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    return extract_confidence_thresholds_from_pr_values(precision, recall, thresholds, min_precision, min_recall)
