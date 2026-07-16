"""Assemble a per-algorithm results table from an experiment's per-cell JSONs.

Now that the results CSVs are gone, this is the single entry point for any
report/analysis over an experiment's summary rows. It reads every
``testing/fold*_numtrajs*_gtrate*/fold_result.json`` (the resume markers, which
store each algorithm's base record + nested ``algorithm_specific``), flattens the
extras to columns, and joins run-level context (``experiment_name, p_mask,
p_noise``) from ``run_params.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from benchmark.experiment_running_helpers.resume import FOLD_RESULT_FILENAME
from benchmark.experiment_running_helpers.result_schema import (
    RUN_CONTEXT_FIELDS,
    flatten_record,
)


def _load_run_params(experiment_dir: Path) -> dict:
    for p in (experiment_dir / "evaluation_results" / "run_params.json",
              experiment_dir / "run_params.json"):
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return {}


def collect_results(experiment_dir: Path) -> pd.DataFrame:
    """Return one flat row per ``(algorithm, fold, num_trajectories, gt_rate)``.

    Columns = base fields + flattened ``algorithm_specific`` extras + joined run
    context (``experiment_name, p_mask, p_noise``). Empty DataFrame if the
    experiment has no completed folds.
    """
    experiment_dir = Path(experiment_dir)
    testing = experiment_dir / "testing"
    run_params = _load_run_params(experiment_dir)
    context = {field: run_params.get(field) for field in RUN_CONTEXT_FIELDS}

    rows = []
    if testing.is_dir():
        for cell in sorted(testing.glob("fold*_numtrajs*_gtrate*")):
            marker = cell / FOLD_RESULT_FILENAME
            if not marker.exists():
                continue
            try:
                records = json.loads(marker.read_text())
            except Exception:
                continue
            for rec in records:
                row = flatten_record(rec)
                for k, v in context.items():
                    row.setdefault(k, v)
                rows.append(row)

    return pd.DataFrame(rows)
