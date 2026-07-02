"""
Correlation analysis: join evaluation metrics with fluent-change (T vs GT / T' vs GT) metrics.

Per-fold step:
    build_correlation_table() reads all_solutions_metrics.json and each
    conflict_free_model_{idx}/metrics.json, joins them by solution_index,
    computes delta columns (T' minus T), and writes both JSON and CSV.

Cross-fold aggregation:
    aggregate_correlation_tables() stacks per-fold tables and computes
    Spearman / Pearson correlations between fluent-change deltas and
    evaluation quality metrics.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Eval metric columns to correlate against ────────────────────────────
EVAL_METRIC_COLS = [
    "solving_ratio",
    "pred_app_precision",
    "pred_app_recall",
    "pred_eff_precision",
    "pred_eff_recall",
    "precision_overall",
    "recall_overall",
]

# ── Fluent-change columns (from metrics.json) ──────────────────────────
FLUENT_COLS = ["tp", "fp", "fn", "precision", "recall"]


def _load_fluent_metrics(model_dir: Path) -> Optional[dict]:
    """Load metrics.json from a conflict-free model directory."""
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read {metrics_path}: {e}")
        return None


def build_correlation_table(
    fold_work_dir: Path,
    output_dir: Optional[Path] = None,
) -> List[dict]:
    """
    Join evaluation metrics with fluent-change metrics for every conflict-free model
    in a single fold.  Writes ``correlation_analysis.json`` and
    ``correlation_analysis.csv`` to *output_dir* (defaults to *fold_work_dir*).

    Returns the joined table (list of row dicts), one row per solution.
    """
    if output_dir is None:
        output_dir = fold_work_dir

    # ── 1. Load all_solutions_metrics.json ──────────────────────────────
    eval_metrics_path = fold_work_dir / "all_solutions_metrics.json"
    if not eval_metrics_path.exists():
        logger.warning(f"all_solutions_metrics.json not found in {fold_work_dir}")
        return []

    with open(eval_metrics_path) as f:
        eval_metrics_rows: List[dict] = json.load(f)

    if not eval_metrics_rows:
        return []

    # Index by solution_index for easy lookup
    eval_metrics_by_idx: Dict[int, dict] = {r["solution_index"]: r for r in eval_metrics_rows}

    # ── 2. Discover model directories ───────────────────────────────────
    cf_dir = fold_work_dir / "conflict_free_models"
    if not cf_dir.exists():
        return []

    model_dirs = sorted(
        [d for d in cf_dir.iterdir() if d.is_dir() and d.name.startswith("conflict_free_model_")],
        key=lambda d: int(d.name.split("_")[-1]),
    )
    # Include final_model fallback (solution_index = -1)
    final_model_dir = cf_dir / "final_model"
    if not model_dirs and final_model_dir.exists():
        model_dirs = [final_model_dir]

    # ── 3. Join ─────────────────────────────────────────────────────────
    rows: List[dict] = []
    for model_dir in model_dirs:
        if model_dir.name == "final_model":
            sol_idx = -1
        else:
            sol_idx = int(model_dir.name.split("_")[-1])

        eval_metrics = eval_metrics_by_idx.get(sol_idx)
        fluent = _load_fluent_metrics(model_dir)

        row: dict = {"solution_index": sol_idx}

        # Fluent-change columns (T vs GT is constant across solutions but
        # we include it in every row so the table is self-contained)
        if fluent is not None:
            t = fluent.get("T_vs_GT", {})
            tp = fluent.get("T_prime_vs_GT", {})
            for col in FLUENT_COLS:
                row[f"T_vs_GT_{col}"] = t.get(col)
                row[f"T_prime_vs_GT_{col}"] = tp.get(col)
                # Delta = T' - T  (positive = T' is higher)
                t_val = t.get(col)
                tp_val = tp.get(col)
                if t_val is not None and tp_val is not None:
                    row[f"delta_{col}"] = tp_val - t_val
                else:
                    row[f"delta_{col}"] = None
        else:
            for col in FLUENT_COLS:
                row[f"T_vs_GT_{col}"] = None
                row[f"T_prime_vs_GT_{col}"] = None
                row[f"delta_{col}"] = None

        # Eval metric columns
        if eval_metrics is not None:
            for col in EVAL_METRIC_COLS:
                row[col] = eval_metrics.get(col)
        else:
            for col in EVAL_METRIC_COLS:
                row[col] = None

        rows.append(row)

    if not rows:
        return []

    # ── 4. Write outputs ────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / "correlation_analysis.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    # CSV
    csv_path = output_dir / "correlation_analysis.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  [CORR] Wrote {len(rows)} rows → {json_path.name}, {csv_path.name}")
    return rows


# ── Cross-fold aggregation ──────────────────────────────────────────────

def aggregate_correlation_tables(
    fold_dirs: List[Path],
    output_dir: Path,
) -> dict:
    """
    Stack per-fold correlation tables and compute Spearman & Pearson
    correlations between fluent-change deltas and AMLGym metrics.

    Args:
        fold_dirs: Paths to fold working directories that contain
            ``correlation_analysis.json``.
        output_dir: Where to write the aggregate outputs.

    Returns:
        Dict with keys ``correlations`` (nested dict of metric pairs →
        {spearman_r, spearman_p, pearson_r, pearson_p}), ``stacked_rows``
        (the full stacked table), and ``n`` (number of data points).
    """
    try:
        from scipy import stats as sp_stats
        _HAS_SCIPY = True
    except ImportError:
        _HAS_SCIPY = False
        print("  [CORR-AGG] WARNING: scipy not installed — correlation coefficients will be skipped.")

    # ── 1. Stack all per-fold rows ──────────────────────────────────────
    stacked: List[dict] = []
    for fd in fold_dirs:
        json_path = fd / "correlation_analysis.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            fold_rows = json.load(f)
        # Tag with fold identifier
        fold_label = fd.name  # e.g. fold0_numtrajs8_gtrate25
        for r in fold_rows:
            r["fold"] = fold_label
        stacked.extend(fold_rows)

    if not stacked:
        print("  [CORR-AGG] No per-fold correlation tables found.")
        return {"correlations": {}, "stacked_rows": [], "n": 0}

    # ── 2. Compute correlations ─────────────────────────────────────────
    delta_cols = [f"delta_{c}" for c in FLUENT_COLS]
    correlations: Dict[str, Dict[str, dict]] = {}

    for delta_col in delta_cols:
        correlations[delta_col] = {}
        for aml_col in EVAL_METRIC_COLS:
            # Gather paired non-None values
            pairs = [
                (r[delta_col], r[aml_col])
                for r in stacked
                if r.get(delta_col) is not None and r.get(aml_col) is not None
            ]
            if len(pairs) < 3 or not _HAS_SCIPY:
                correlations[delta_col][aml_col] = {
                    "n": len(pairs),
                    "spearman_r": None, "spearman_p": None,
                    "pearson_r": None, "pearson_p": None,
                }
                continue

            xs, ys = zip(*pairs)
            try:
                sp_r, sp_p = sp_stats.spearmanr(xs, ys)
            except Exception:
                sp_r, sp_p = None, None
            try:
                pe_r, pe_p = sp_stats.pearsonr(xs, ys)
            except Exception:
                pe_r, pe_p = None, None

            correlations[delta_col][aml_col] = {
                "n": len(pairs),
                "spearman_r": _safe_float(sp_r),
                "spearman_p": _safe_float(sp_p),
                "pearson_r": _safe_float(pe_r),
                "pearson_p": _safe_float(pe_p),
            }

    # ── 3. Write outputs ────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stacked table (JSON + CSV)
    stacked_json = output_dir / "correlation_stacked.json"
    with open(stacked_json, "w") as f:
        json.dump(stacked, f, indent=2)

    if stacked:
        stacked_csv = output_dir / "correlation_stacked.csv"
        fieldnames = list(stacked[0].keys())
        with open(stacked_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stacked)

    # Correlation summary
    corr_json = output_dir / "correlation_summary.json"
    with open(corr_json, "w") as f:
        json.dump({"n_total": len(stacked), "correlations": correlations}, f, indent=2)

    # Flat CSV for easy viewing: one row per (delta_col, aml_col) pair
    flat_rows = []
    for delta_col, aml_dict in correlations.items():
        for aml_col, vals in aml_dict.items():
            flat_rows.append({
                "fluent_delta": delta_col,
                "eval_metric": aml_col,
                **vals,
            })
    corr_csv = output_dir / "correlation_summary.csv"
    if flat_rows:
        with open(corr_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)

    print(
        f"  [CORR-AGG] Aggregated {len(stacked)} rows across {len(fold_dirs)} folds. "
        f"Correlations written to {corr_json.name}"
    )
    return {"correlations": correlations, "stacked_rows": stacked, "n": len(stacked)}


def _safe_float(v) -> Optional[float]:
    """Convert numpy/scipy scalar to plain float, or None if NaN."""
    if v is None:
        return None
    try:
        fv = float(v)
        import math
        return None if math.isnan(fv) else fv
    except (TypeError, ValueError):
        return None
