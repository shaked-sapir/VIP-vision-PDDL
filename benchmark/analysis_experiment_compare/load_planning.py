from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_planning_rows(
    experiments: List[Dict[str, Any]],
    domain_filter: str = "blocksworld",
) -> pd.DataFrame:
    """Load solving/false/unsolvable ratios from evaluation csv files.

    Uses `results_*_combined_timeout*.csv` to keep both cleaned and unclean rows.
    """
    rows: List[Dict[str, Any]] = []

    for exp in experiments:
        eval_dir = Path(exp["evaluation_results_path"])
        combined_csvs = sorted(eval_dir.glob("results_*_combined_timeout*.csv"))

        for csv_path in combined_csvs:
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue

            if domain_filter and "domain" in df.columns:
                df = df[df["domain"] == domain_filter]

            if df.empty:
                continue

            required = {
                "algorithm",
                "_internal_phase",
                "num_trajectories",
                "gt_rate",
                "solving_ratio",
                "false_plans_ratio",
                "unsolvable_ratio",
            }
            if not required.issubset(set(df.columns)):
                continue

            for _, rec in df.iterrows():
                rows.append(
                    {
                        "experiment_id": exp["experiment_id"],
                        "timeout_seconds": exp["timeout_seconds"],
                        "model_constraint_weight": exp["model_constraint_weight"],
                        "date_tag": exp["date_tag"],
                        "algorithm": rec.get("algorithm"),
                        "phase": rec.get("_internal_phase"),
                        "fold": rec.get("fold"),
                        "num_trajectories": rec.get("num_trajectories"),
                        "gt_rate": rec.get("gt_rate"),
                        "solving_ratio": rec.get("solving_ratio"),
                        "false_plans_ratio": rec.get("false_plans_ratio"),
                        "unsolvable_ratio": rec.get("unsolvable_ratio"),
                        "source_file": str(csv_path),
                    }
                )

    return pd.DataFrame(rows)

