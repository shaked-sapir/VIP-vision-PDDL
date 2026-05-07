from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


FOLD_DIR_RE = re.compile(r"fold(?P<fold>\d+)_numtrajs(?P<num>\d+)_gtrate(?P<gt>\d+)")


def _extract_fold_tuple(path: Path) -> Optional[Tuple[int, int, int]]:
    for part in path.parts:
        match = FOLD_DIR_RE.fullmatch(part)
        if match:
            return (
                int(match.group("fold")),
                int(match.group("num")),
                int(match.group("gt")),
            )
    return None


def _safe_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _phase_rows_from_metrics(metrics: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    if "T_vs_GT" in metrics and isinstance(metrics["T_vs_GT"], dict):
        out.append(("unclean", metrics["T_vs_GT"]))
    if "T_prime_vs_GT" in metrics and isinstance(metrics["T_prime_vs_GT"], dict):
        out.append(("cleaned", metrics["T_prime_vs_GT"]))
    return out


def _select_metrics_files(exp_path: Path, model_policy: str) -> List[Path]:
    pattern = "testing/fold*_numtrajs*_gtrate*/conflict_free_models/**/metrics.json"
    all_metrics = sorted(exp_path.glob(pattern))
    if model_policy == "all":
        return all_metrics

    selected: Dict[str, Path] = {}
    for p in all_metrics:
        fold_info = _extract_fold_tuple(p)
        if fold_info is None:
            continue
        fold, num_traj, gt_rate = fold_info
        key = f"{fold}-{num_traj}-{gt_rate}"

        # Prefer final_model, fallback to first discovered conflict_free_model_0.
        if "final_model" in p.parts:
            selected[key] = p
        elif key not in selected:
            selected[key] = p

    return sorted(selected.values())


def load_fluent_rows(
    experiments: List[Dict[str, Any]],
    model_policy: str = "final_model",
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for exp in experiments:
        exp_path = Path(exp["experiment_path"])
        metrics_files = _select_metrics_files(exp_path, model_policy=model_policy)

        for metrics_path in metrics_files:
            fold_tuple = _extract_fold_tuple(metrics_path)
            if fold_tuple is None:
                continue
            fold, num_traj, gt_rate = fold_tuple

            data = _safe_json(metrics_path)
            if not data:
                continue

            for phase, bucket in _phase_rows_from_metrics(data):
                rows.append(
                    {
                        "experiment_id": exp["experiment_id"],
                        "timeout_seconds": exp["timeout_seconds"],
                        "model_constraint_weight": exp["model_constraint_weight"],
                        "date_tag": exp["date_tag"],
                        "algorithm": "PISAM",
                        "phase": phase,
                        "fold": fold,
                        "num_trajectories": num_traj,
                        "gt_rate": gt_rate,
                        "tp": bucket.get("tp"),
                        "fp": bucket.get("fp"),
                        "fn": bucket.get("fn"),
                        "precision": bucket.get("precision"),
                        "recall": bucket.get("recall"),
                        "metrics_source_path": str(metrics_path),
                        "model_selection_policy": model_policy,
                    }
                )

    return pd.DataFrame(rows)

