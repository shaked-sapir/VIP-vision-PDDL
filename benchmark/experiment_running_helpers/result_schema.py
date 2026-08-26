"""Canonical result-record schema shared by every algorithm.

Each per-cell, per-algorithm record has a set of **mode-agnostic base fields**
(identical for simulated and image runs) plus a nested ``algorithm_specific``
dict that each algorithm fills with its own extras (CDPS: conflict-search stats;
ROSAME: its own, empty for now).

Run-level context (``experiment_name``, ``p_mask``, ``p_noise``) is *not* stored
per record — it lives once in ``run_params.json`` and is joined at report time
(see ``collect_results``). That keeps this record schema mode-agnostic, so adding
the image version later needs no schema change.
"""

# Identity / grouping.
BASE_IDENTITY_FIELDS = ["domain", "algorithm", "fold", "num_trajectories", "gt_rate"]

# Shared data-context / cost fields (same for every algorithm in a cell).
BASE_CONTEXT_FIELDS = [
    "problems_count", "total_transitions", "total_gt_transitions", "learning_time_seconds",
]

# AMLGym evaluation metrics — produced identically for any learned model.
AMLGYM_METRIC_FIELDS = [
    "precision_precs_pos", "precision_precs_neg", "precision_eff_pos", "precision_eff_neg", "precision_overall",
    "recall_precs_pos", "recall_precs_neg", "recall_eff_pos", "recall_eff_neg", "recall_overall",
    "solving_ratio", "false_plans_ratio", "unsolvable_ratio", "planning_timed_out_ratio",
    "pred_app_precision", "pred_app_recall", "pred_eff_precision", "pred_eff_recall",
    "pred_undefined_reason",
]

BASE_FIELDS = BASE_IDENTITY_FIELDS + BASE_CONTEXT_FIELDS + AMLGYM_METRIC_FIELDS

# Run-level context joined from run_params.json (not stored per record).
RUN_CONTEXT_FIELDS = ["experiment_name", "p_mask", "p_noise"]

ALGORITHM_SPECIFIC_KEY = "algorithm_specific"


def flatten_record(record: dict) -> dict:
    """Flatten a record's nested ``algorithm_specific`` extras to top-level keys.

    Base fields and extras are disjoint by construction, so the merge is safe.
    """
    extras = record.get(ALGORITHM_SPECIFIC_KEY) or {}
    base = {k: v for k, v in record.items() if k != ALGORITHM_SPECIFIC_KEY}
    return {**base, **extras}
