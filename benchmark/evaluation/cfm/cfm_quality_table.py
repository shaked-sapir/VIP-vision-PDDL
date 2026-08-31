"""Export CFM-quality raw data as a pivot-ready table.

Alongside the figures produced by :mod:`cfm_quality_analysis`, this writes a
tidy long table so the underlying per-CFM data can be pivoted / analysed
directly (Excel pivot tables, pandas, etc.).

Output is a single workbook ``evaluation_results/CFM_quality/cfm_quality_table.xlsx``
with two sheets:

* ``raw``      — one row per numbered CFM (solution_index >= 0) per instance,
                 with all metrics from ``all_solutions_metrics.json`` plus the
                 per-CFM search-log fields from ``conflict_free_solutions_log.json``.
                 Unpadded — real per-instance values only.
* ``selected`` — one row per instance's *selected* model (solution_index == -1),
                 keyed by fold + num_trajs (+ gt_rate), for later inspection.

Usage (standalone, mirrors cfm_quality_analysis / experiment_report):
    python -m benchmark.evaluation.cfm.cfm_quality_table <experiment_root>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from benchmark.evaluation.cfm.cfm_quality_analysis import (
    METRICS,
    SYNTACTIC_PRECISION_METRICS,
    SYNTACTIC_RECALL_METRICS,
    find_instance_dirs,
    load_domain_name,
    load_fluent_patch_counts,
)


# Known metric keys from all_solutions_metrics.json, in a stable display order.
# Any unrecognised keys found in the data are appended per-experiment so nothing
# is silently dropped.
_KNOWN_METRIC_KEYS: List[str] = [
    *(k for k, _ in METRICS),
    *(k for k, _ in SYNTACTIC_PRECISION_METRICS),
    *(k for k, _ in SYNTACTIC_RECALL_METRICS),
    "false_plans_ratio",
    "planning_timed_out_ratio",
    "unsolvable_ratio",
]

# Per-CFM search-log fields joined from conflict_free_solutions_log.json by index.
_LOG_FIELDS: List[str] = [
    "fluent_patch_count",
    "cost",
    "depth",
    "model_constraint_count",
    "nodes_expanded_so_far",
    "wall_time_so_far",
]

# Identifier columns for the two sheets.
_ID_COLUMNS: List[str] = [
    "domain", "experiment", "masking_p", "noising_p",
    "fold", "num_trajs", "gt_rate", "solution_index", "total_cfm_count",
]
_SELECTED_ID_COLUMNS: List[str] = [
    "domain", "experiment", "masking_p", "noising_p", "fold", "num_trajs", "gt_rate",
]

_INSTANCE_NAME_RE = re.compile(r"fold(\d+)_numtrajs(\d+)_gtrate(\d+)")
_PARAMS_NAME_RE = re.compile(r"mask=([\d.]+)__noise=([\d.]+)")


def _parse_instance_name(name: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Extract (fold, num_trajs, gt_rate) from an instance dir name."""
    match = _INSTANCE_NAME_RE.search(name)
    if not match:
        return None, None, None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _experiment_params(experiment_root: Path) -> Dict[str, Optional[float]]:
    """Resolve masking_p / noising_p from run_params.json, falling back to the dir name."""
    run_params_path = experiment_root / "evaluation_results" / "run_params.json"
    if run_params_path.exists():
        with open(run_params_path) as f:
            params = json.load(f)
        masking = params.get("masking_p", params.get("masking_percentage"))
        noising = params.get("noising_p", params.get("noising_percentage"))
        if masking is not None or noising is not None:
            return {"masking_p": masking, "noising_p": noising}

    match = _PARAMS_NAME_RE.search(experiment_root.name)
    if match:
        return {"masking_p": float(match.group(1)), "noising_p": float(match.group(2))}
    return {"masking_p": None, "noising_p": None}


def _load_solution_entries(instance_dir: Path) -> List[Dict]:
    """Load raw all_solutions_metrics.json entries (both numbered and selected).

    Empty when absent: the file is CDPS's numbered-CFM set, which a sweep with
    no CDPS arm never writes.
    """
    path = instance_dir / "all_solutions_metrics.json"
    if not path.is_file():
        return []
    with open(path) as f:
        return json.load(f)


def _metric_values(entry: Dict) -> Dict:
    """Return an entry's metric fields (everything except solution_index)."""
    return {k: v for k, v in entry.items() if k != "solution_index"}


def build_cfm_quality_records(
    experiment_root: Path,
) -> Tuple[List[dict], List[dict], List[str]]:
    """Build tidy records for the raw (numbered CFM) and selected-model tables.

    Args:
        experiment_root: Experiment dir (contains testing/ and evaluation_results/).

    Returns:
        (raw_rows, selected_rows, metric_columns) where metric_columns is the
        ordered list of metric keys present (known first, extras appended).

    Raises:
        FileNotFoundError: If no testing/ directory exists.
    """
    testing_dir = experiment_root / "testing"
    if not testing_dir.is_dir():
        raise FileNotFoundError(f"testing dir not found: {testing_dir}")

    domain = load_domain_name(experiment_root)
    params = _experiment_params(experiment_root)
    experiment = experiment_root.name

    raw_rows: List[dict] = []
    selected_rows: List[dict] = []
    seen_metric_keys: Set[str] = set()

    for instance_dir in find_instance_dirs(testing_dir):
        fold, num_trajs, gt_rate = _parse_instance_name(instance_dir.name)
        entries = _load_solution_entries(instance_dir)
        numbered = sorted(
            (e for e in entries if e.get("solution_index", -1) >= 0),
            key=lambda e: e["solution_index"],
        )
        selected = [e for e in entries if e.get("solution_index", -1) == -1]

        # index -> full search-log entry (for the search-log fields).
        log_by_index = {e["index"]: e for e in load_fluent_patch_counts(instance_dir)}

        base = {
            "domain": domain, "experiment": experiment,
            "masking_p": params["masking_p"], "noising_p": params["noising_p"],
            "fold": fold, "num_trajs": num_trajs, "gt_rate": gt_rate,
        }

        for entry in numbered:
            metrics = _metric_values(entry)
            seen_metric_keys.update(metrics)
            solution_index = entry["solution_index"]
            log_entry = log_by_index.get(solution_index, {})
            row = {
                **base,
                "solution_index": solution_index,
                "total_cfm_count": len(numbered),
                **metrics,
                **{field: log_entry.get(field) for field in _LOG_FIELDS},
            }
            raw_rows.append(row)

        for entry in selected:
            metrics = _metric_values(entry)
            seen_metric_keys.update(metrics)
            selected_rows.append({**base, **metrics})

    extra = sorted(seen_metric_keys - set(_KNOWN_METRIC_KEYS))
    metric_columns = _KNOWN_METRIC_KEYS + extra
    return raw_rows, selected_rows, metric_columns


def generate_cfm_quality_table(experiment_root: Path) -> Optional[Path]:
    """Write the CFM-quality raw table workbook for a completed experiment.

    Reads <experiment_root>/testing/ and writes
    <experiment_root>/evaluation_results/CFM_quality/cfm_quality_table.xlsx.

    Args:
        experiment_root: Experiment dir (contains testing/ and evaluation_results/).

    Returns:
        The output path when a table was written, or None when there is no CFM
        solution data to tabulate.

    Raises:
        FileNotFoundError: If no testing/ directory exists.
    """
    raw_rows, selected_rows, metric_columns = build_cfm_quality_records(experiment_root)

    if not raw_rows and not selected_rows:
        print("No CFM solution metrics found. Nothing to tabulate.")
        return None

    output_dir = experiment_root / "evaluation_results" / "CFM_quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cfm_quality_table.xlsx"

    raw_columns = _ID_COLUMNS + metric_columns + _LOG_FIELDS
    selected_columns = _SELECTED_ID_COLUMNS + metric_columns

    raw_df = pd.DataFrame(raw_rows).reindex(columns=raw_columns)
    selected_df = pd.DataFrame(selected_rows).reindex(columns=selected_columns)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="raw", index=False)
        selected_df.to_excel(writer, sheet_name="selected", index=False)

    print(
        f"  Saved: {output_path}  "
        f"(raw: {len(raw_df)} rows, selected: {len(selected_df)} rows)"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export CFM-quality raw data as a pivot-ready XLSX table."
    )
    parser.add_argument(
        "experiment_root",
        type=str,
        help="Path to experiment config dir (contains testing/ and evaluation_results/).",
    )
    args = parser.parse_args()

    try:
        generate_cfm_quality_table(Path(args.experiment_root))
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
