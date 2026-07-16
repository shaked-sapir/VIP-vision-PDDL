"""
Generate a fully-detailed Excel report from a completed experiment directory.

Usage:
    python -m benchmark.evaluation.experiment_report <experiment_path>

The report is written to <experiment_path>/evaluation_results/fully-detailed-report.xlsx
with sheets:
    1. run_params                    — key-value table of run_params.json
    2. cs_traversal_metrics          — per-fold learning + node-expansion stats
    3. solving_folds_summary         — folds where at least one model has solving_ratio > 0
    4. raw_data                      — every CFM across all folds/numtrajs with search + metric + correlation info
    5. pivot                         — grouped by numtrajs → fold → solution_index
    6. correlation                   — Spearman/Pearson correlations between fluent_patch_count and pred_* metrics
    7. summary                       — per-numtrajs aggregates
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


# =============================================================================
# DATA LOADING
# =============================================================================

def _parse_fold_dir_name(name: str) -> Optional[Dict[str, Any]]:
    """Extract fold, numtrajs, gtrate from a directory name like 'fold0_numtrajs3_gtrate0'."""
    m = re.match(r"fold(\d+)_numtrajs(\d+)_gtrate(\d+)", name)
    if not m:
        return None
    return {
        "fold": int(m.group(1)),
        "numtrajs": int(m.group(2)),
        "gt_rate": int(m.group(3)),
    }


def _load_json(path: Path) -> Any:
    """Load a JSON file, returning None if missing."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def collect_raw_rows(testing_dir: Path) -> List[Dict[str, Any]]:
    """Walk testing/ and join per-fold JSONs into flat rows (one per CFM)."""
    rows: List[Dict[str, Any]] = []

    fold_dirs = sorted(
        [d for d in testing_dir.iterdir() if d.is_dir() and d.name.startswith("fold")],
        key=lambda d: d.name,
    )

    for fold_dir in fold_dirs:
        info = _parse_fold_dir_name(fold_dir.name)
        if info is None:
            continue

        # Load per-fold data
        solutions_log: List[dict] = _load_json(fold_dir / "conflict_free_solutions_log.json") or []
        all_metrics: List[dict] = _load_json(fold_dir / "all_solutions_metrics.json") or []
        learning_metrics: dict = _load_json(fold_dir / "learning_metrics.json") or {}
        correlation_data: List[dict] = _load_json(fold_dir / "correlation_analysis.json") or []

        # Index by solution_index for fast lookup
        metrics_by_idx = {m["solution_index"]: m for m in all_metrics}
        corr_by_idx = {c["solution_index"]: c for c in correlation_data}

        num_solutions = len(solutions_log)
        terminated_by = learning_metrics.get("terminated_by", "")
        total_nodes = learning_metrics.get("nodes_expanded", 0)

        for sol in solutions_log:
            idx = sol["index"]
            row: Dict[str, Any] = {}

            # Fold identification
            row["fold"] = info["fold"]
            row["numtrajs"] = info["numtrajs"]

            # Search info from solutions log
            row["solution_index"] = idx
            row["fluent_patch_count"] = sol.get("fluent_patch_count", 0)
            row["model_constraint_count"] = sol.get("model_constraint_count", 0)
            row["cost"] = sol.get("cost", 0)
            row["depth"] = sol.get("depth", 0)
            row["nodes_expanded_so_far"] = sol.get("nodes_expanded_so_far", 0)
            row["wall_time_so_far"] = sol.get("wall_time_so_far", 0)

            # Learning-level info
            row["terminated_by"] = terminated_by
            row["total_nodes_expanded"] = total_nodes
            row["num_solutions_in_fold"] = num_solutions

            # Syntactic + solving + predictive metrics
            met = metrics_by_idx.get(idx, {})
            row["precision_overall"] = met.get("precision_overall")
            row["recall_overall"] = met.get("recall_overall")
            row["solving_ratio"] = met.get("solving_ratio")
            row["false_plans_ratio"] = met.get("false_plans_ratio")
            row["unsolvable_ratio"] = met.get("unsolvable_ratio")
            row["pred_app_precision"] = met.get("pred_app_precision")
            row["pred_app_recall"] = met.get("pred_app_recall")
            row["pred_eff_precision"] = met.get("pred_eff_precision")
            row["pred_eff_recall"] = met.get("pred_eff_recall")

            # Correlation / trajectory comparison
            corr = corr_by_idx.get(idx, {})
            row["T_vs_GT_tp"] = corr.get("T_vs_GT_tp")
            row["T_vs_GT_fp"] = corr.get("T_vs_GT_fp")
            row["T_vs_GT_fn"] = corr.get("T_vs_GT_fn")
            row["T_prime_vs_GT_tp"] = corr.get("T_prime_vs_GT_tp")
            row["T_prime_vs_GT_fp"] = corr.get("T_prime_vs_GT_fp")
            row["T_prime_vs_GT_fn"] = corr.get("T_prime_vs_GT_fn")
            row["delta_tp"] = corr.get("delta_tp")
            row["delta_fp"] = corr.get("delta_fp")
            row["delta_fn"] = corr.get("delta_fn")

            rows.append(row)

    return rows


_LEARNING_METRICS_KEYS = [
    "learning_time_seconds", "max_depth", "nodes_expanded",
    "terminated_by", "conflict_free_model_count", "actual_timeout_seconds",
]


def collect_traversal_metrics(testing_dir: Path) -> List[Dict[str, Any]]:
    """Collect per-fold learning metrics + node expansion summary."""
    rows: List[Dict[str, Any]] = []

    fold_dirs = sorted(
        [d for d in testing_dir.iterdir() if d.is_dir() and d.name.startswith("fold")],
        key=lambda d: d.name,
    )

    for fold_dir in fold_dirs:
        info = _parse_fold_dir_name(fold_dir.name)
        if info is None:
            continue

        row: Dict[str, Any] = {"fold": info["fold"], "numtrajs": info["numtrajs"]}

        # learning_metrics.json
        lm = _load_json(fold_dir / "learning_metrics.json") or {}
        for key in _LEARNING_METRICS_KEYS:
            row[key] = lm.get(key)

        # node_expansion_times.json → summary
        net = _load_json(fold_dir / "node_expansion_times.json") or {}
        summary = net.get("summary", {})
        for key, val in summary.items():
            row[key] = val

        rows.append(row)

    return rows


def collect_solving_folds(testing_dir: Path) -> List[Dict[str, Any]]:
    """Collect folds where at least one model has solving_ratio > 0.

    Each row contains: fold, numtrajs, solving_ratio, solution_index, model_type.
    """
    rows: List[Dict[str, Any]] = []
    fold_dirs = sorted(
        [d for d in testing_dir.iterdir() if d.is_dir() and d.name.startswith("fold")],
        key=lambda d: d.name,
    )
    for fold_dir in fold_dirs:
        info = _parse_fold_dir_name(fold_dir.name)
        if info is None:
            continue
        entries = _load_json(fold_dir / "all_solutions_metrics.json")
        if not entries or not isinstance(entries, list):
            continue
        candidates = [e for e in entries if e.get("solving_ratio", 0) > 0]
        if not candidates:
            continue
        best = max(candidates, key=lambda e: e["solving_ratio"])
        sol_idx = best.get("solution_index", -1)
        rows.append({
            "fold": fold_dir.name,
            "solving_ratio": best["solving_ratio"],
            "solution_index": sol_idx,
            "model_type": "CFM" if sol_idx >= 0 else "fallback",
        })
    return rows


# =============================================================================
# PIVOT TABLE
# =============================================================================

def build_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Build a pivot table grouped by numtrajs → fold → solution_index."""
    if df.empty:
        return pd.DataFrame()

    pivot_rows: List[Dict[str, Any]] = []

    for numtrajs in sorted(df["numtrajs"].unique()):
        nt_df = df[df["numtrajs"] == numtrajs]

        for fold in sorted(nt_df["fold"].unique()):
            fold_df = nt_df[nt_df["fold"] == fold]

            for sol_idx in sorted(fold_df["solution_index"].unique()):
                sol_row = fold_df[fold_df["solution_index"] == sol_idx].iloc[0]
                pivot_rows.append({
                    "numtrajs": numtrajs,
                    "fold": fold,
                    "solution_index": sol_idx,
                    "fluent_patch_count": sol_row["fluent_patch_count"],
                    "model_constraint_count": sol_row["model_constraint_count"],
                    "solving_ratio": sol_row["solving_ratio"],
                })

        # Totals per numtrajs
        pivot_rows.append({
            "numtrajs": f"{numtrajs} Total",
            "fold": "",
            "solution_index": "",
            "fluent_patch_count": nt_df["fluent_patch_count"].sum(),
            "model_constraint_count": nt_df["model_constraint_count"].sum(),
            "solving_ratio": nt_df["solving_ratio"].mean() if "solving_ratio" in nt_df else None,
        })

    # Grand total
    pivot_rows.append({
        "numtrajs": "Grand Total",
        "fold": "",
        "solution_index": "",
        "fluent_patch_count": df["fluent_patch_count"].sum(),
        "model_constraint_count": df["model_constraint_count"].sum(),
        "solving_ratio": df["solving_ratio"].mean(),
    })

    return pd.DataFrame(pivot_rows)


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

PRED_METRICS = ["pred_app_precision", "pred_app_recall", "pred_eff_precision", "pred_eff_recall"]


def _safe_spearman(x, y) -> Tuple[float, float]:
    """Spearman correlation, returns (rho, p) or (NaN, NaN) on failure."""
    if scipy_stats is None:
        return (float("nan"), float("nan"))
    try:
        if x.nunique() <= 1 or y.nunique() <= 1:
            return (float("nan"), float("nan"))
        res = scipy_stats.spearmanr(x, y)
        return (res.statistic, res.pvalue)
    except Exception:
        return (float("nan"), float("nan"))


def _safe_pearson(x, y) -> Tuple[float, float]:
    """Pearson correlation, returns (r, p) or (NaN, NaN) on failure."""
    if scipy_stats is None:
        return (float("nan"), float("nan"))
    try:
        if x.nunique() <= 1 or y.nunique() <= 1:
            return (float("nan"), float("nan"))
        res = scipy_stats.pearsonr(x, y)
        return (res.statistic, res.pvalue)
    except Exception:
        return (float("nan"), float("nan"))


def _sig_marker(p: float) -> str:
    """Return significance marker for a p-value."""
    if pd.isna(p):
        return "N/A"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def build_metric_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for each pred_* metric."""
    rows = []
    for m in PRED_METRICS:
        col = df[m].dropna()
        note = ""
        if col.nunique() <= 1:
            note = "CONSTANT — no variance, excluded from analysis"
        rows.append({
            "Metric": m,
            "min": col.min() if len(col) else None,
            "max": col.max() if len(col) else None,
            "mean": col.mean() if len(col) else None,
            "std": col.std() if len(col) else None,
            "unique values": col.nunique(),
            "note": note,
        })
    return pd.DataFrame(rows)


def build_global_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Global correlations: fluent_patch_count vs each pred_* metric."""
    rows = []
    fpc = df["fluent_patch_count"]
    for m in PRED_METRICS:
        col = df[m].dropna()
        valid = df[["fluent_patch_count", m]].dropna()
        if valid.empty or col.nunique() <= 1:
            rows.append({
                "Metric": m,
                "Spearman ρ": "N/A", "p-value (Spearman)": "N/A", "sig (Spearman)": "N/A",
                "Pearson r": "N/A", "p-value (Pearson)": "N/A", "sig (Pearson)": "constant" if col.nunique() <= 1 else "N/A",
            })
            continue
        sp_rho, sp_p = _safe_spearman(valid["fluent_patch_count"], valid[m])
        pe_r, pe_p = _safe_pearson(valid["fluent_patch_count"], valid[m])
        rows.append({
            "Metric": m,
            "Spearman ρ": sp_rho, "p-value (Spearman)": sp_p, "sig (Spearman)": _sig_marker(sp_p),
            "Pearson r": pe_r, "p-value (Pearson)": pe_p, "sig (Pearson)": _sig_marker(pe_p),
        })
    return pd.DataFrame(rows)


def build_per_numtrajs_correlations(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-numtrajs Spearman correlation between fluent_patch_count and a metric."""
    rows = []
    for nt in sorted(df["numtrajs"].unique()):
        sub = df[df["numtrajs"] == nt][["fluent_patch_count", metric]].dropna()
        n = len(sub)
        if n < 3 or sub[metric].nunique() <= 1:
            rows.append({"numtrajs": nt, "n": n, "Spearman ρ": "N/A", "p-value": "N/A", "sig": "N/A", "interpretation": "insufficient data"})
            continue
        rho, p = _safe_spearman(sub["fluent_patch_count"], sub[metric])
        sig = _sig_marker(p)
        if sig == "n.s.":
            interp = "no significant relationship at this trajectory count"
        elif rho < 0:
            interp = f"fewer patches → better {metric.replace('pred_', '')}"
        else:
            interp = f"fewer patches → WORSE {metric.replace('pred_', '')}"
        rows.append({"numtrajs": nt, "n": n, "Spearman ρ": rho, "p-value": p, "sig": sig, "interpretation": interp})
    return pd.DataFrame(rows)


def build_mann_whitney(df: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U test: pred_* metrics for solving vs non-solving models."""
    if scipy_stats is None:
        return pd.DataFrame()

    solves = df[df["solving_ratio"] > 0]
    no_solves = df[df["solving_ratio"] == 0]

    if solves.empty or no_solves.empty:
        return pd.DataFrame()

    rows = []
    for m in PRED_METRICS:
        s_vals = solves[m].dropna()
        n_vals = no_solves[m].dropna()
        if s_vals.empty or n_vals.empty or s_vals.nunique() <= 1:
            rows.append({"pred_* metric": m, "mean (solves)": "N/A", "mean (no-solve)": "N/A",
                         "Mann-Whitney U": "N/A", "p-value (greater)": "N/A", "sig": "N/A"})
            continue
        try:
            u_stat, p_val = scipy_stats.mannwhitneyu(s_vals, n_vals, alternative="greater")
        except Exception:
            u_stat, p_val = float("nan"), float("nan")
        rows.append({
            "pred_* metric": m,
            "mean (solves)": s_vals.mean(),
            "mean (no-solve)": n_vals.mean(),
            "Mann-Whitney U": u_stat,
            "p-value (greater)": p_val,
            "sig": _sig_marker(p_val),
        })
    return pd.DataFrame(rows)


def build_pred_vs_solving_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlations between pred_* metrics and solving outcomes."""
    solving_outcomes = ["solving_ratio", "unsolvable_ratio", "false_plans_ratio"]
    rows = []
    for pred_m in PRED_METRICS:
        for solve_m in solving_outcomes:
            valid = df[[pred_m, solve_m]].dropna()
            if valid.empty or valid[pred_m].nunique() <= 1:
                rows.append({"Predictor (pred_*)": pred_m, "Outcome (solving)": solve_m,
                             "Spearman ρ": "N/A", "p-value": "N/A", "sig": "N/A"})
                continue
            rho, p = _safe_spearman(valid[pred_m], valid[solve_m])
            rows.append({
                "Predictor (pred_*)": pred_m,
                "Outcome (solving)": solve_m,
                "Spearman ρ": rho, "p-value": p, "sig": _sig_marker(p),
            })
        # Blank separator row
        rows.append({"Predictor (pred_*)": "", "Outcome (solving)": "", "Spearman ρ": "", "p-value": "", "sig": ""})
    return pd.DataFrame(rows)


# =============================================================================
# SUMMARY STATS
# =============================================================================

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-numtrajs aggregate statistics."""
    rows = []
    for nt in sorted(df["numtrajs"].unique()):
        sub = df[df["numtrajs"] == nt]
        rows.append({
            "numtrajs": nt,
            "n_models": len(sub),
            "mean_fpc": sub["fluent_patch_count"].mean(),
            "std_fpc": sub["fluent_patch_count"].std(),
            "mean_mcc": sub["model_constraint_count"].mean(),
            "mean_solving": sub["solving_ratio"].mean() if "solving_ratio" in sub else None,
            "n_solving>0": (sub["solving_ratio"] > 0).sum() if "solving_ratio" in sub else 0,
            "mean_pred_app_prec": sub["pred_app_precision"].mean(),
            "mean_pred_eff_prec": sub["pred_eff_precision"].mean(),
            "mean_pred_eff_rec": sub["pred_eff_recall"].mean(),
        })
    return pd.DataFrame(rows)


# =============================================================================
# EXCEL WRITER
# =============================================================================

RAW_COLUMNS = [
    "fold", "numtrajs", "solution_index",
    "fluent_patch_count", "model_constraint_count", "cost", "depth",
    "nodes_expanded_so_far", "wall_time_so_far",
    "terminated_by", "total_nodes_expanded", "num_solutions_in_fold",
    "precision_overall", "recall_overall",
    "solving_ratio", "false_plans_ratio", "unsolvable_ratio",
    "pred_app_precision", "pred_app_recall", "pred_eff_precision", "pred_eff_recall",
    "T_vs_GT_tp", "T_vs_GT_fp", "T_vs_GT_fn",
    "T_prime_vs_GT_tp", "T_prime_vs_GT_fp", "T_prime_vs_GT_fn",
    "delta_tp", "delta_fp", "delta_fn",
]


def _write_df_to_sheet(
    writer: pd.ExcelWriter,
    df: pd.DataFrame,
    sheet_name: str,
    title: Optional[str] = None,
    start_row: int = 0,
) -> int:
    """Write a DataFrame to a sheet with optional title row. Returns next available row."""
    workbook = writer.book
    header_fmt = workbook.add_format({"bold": True, "border": 1, "bg_color": "#D9E1F2", "text_wrap": True})
    cell_fmt = workbook.add_format({"border": 1})
    title_fmt = workbook.add_format({"bold": True, "font_size": 13})

    sheet = writer.sheets.get(sheet_name)
    if sheet is None:
        df.head(0).to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row)
        sheet = writer.sheets[sheet_name]
        # Clear what pandas wrote — we'll write manually
        for r in range(start_row, start_row + 1):
            for c in range(len(df.columns)):
                sheet.write_blank(r, c, None)

    row = start_row

    if title:
        sheet.write(row, 0, title, title_fmt)
        row += 1

    # Headers
    for c, col_name in enumerate(df.columns):
        sheet.write(row, c, col_name, header_fmt)
    row += 1

    # Data
    for _, data_row in df.iterrows():
        for c, col_name in enumerate(df.columns):
            val = data_row[col_name]
            if pd.isna(val) if not isinstance(val, str) else False:
                sheet.write_blank(row, c, None, cell_fmt)
            elif isinstance(val, float):
                sheet.write_number(row, c, val, cell_fmt)
            elif isinstance(val, int):
                sheet.write_number(row, c, val, cell_fmt)
            else:
                sheet.write(row, c, str(val), cell_fmt)
        row += 1

    # Auto-width columns (approximate)
    for c, col_name in enumerate(df.columns):
        max_len = max(
            len(str(col_name)),
            df[col_name].astype(str).str.len().max() if len(df) else 0,
        )
        sheet.set_column(c, c, min(max_len + 2, 30))

    return row


def _write_per_algorithm_sheets(writer: pd.ExcelWriter, df_algos: pd.DataFrame) -> None:
    """One sheet per algorithm: mean-over-folds table + the per-fold rows.

    Columns that are all-empty for an algorithm (e.g. CDPS-only extras on ROSAME)
    are dropped, so each sheet shows only that algorithm's base + own extras.
    """
    if df_algos is None or df_algos.empty:
        return

    id_cols = ["domain", "algorithm", "fold", "num_trajectories", "gt_rate", "experiment_name"]
    group = ["num_trajectories", "gt_rate"]

    for algo in sorted(df_algos["algorithm"].dropna().unique()):
        sub = df_algos[df_algos["algorithm"] == algo].dropna(axis=1, how="all").reset_index(drop=True)
        sheet = f"algo_{algo}"[:31]

        numeric = sub.select_dtypes("number").columns.tolist()
        agg_metrics = [c for c in numeric if c not in group and c != "fold"]
        if set(group).issubset(sub.columns) and agg_metrics:
            agg = sub.groupby(group)[agg_metrics].mean().reset_index()
            next_row = _write_df_to_sheet(writer, agg, sheet, title=f"{algo} — mean over folds")
        else:
            next_row = 0

        _write_df_to_sheet(writer, sub, sheet, title=f"{algo} — per-fold rows",
                           start_row=next_row + 2)


def write_report(
    experiment_path: Path,
    run_params: dict,
    df_traversal: pd.DataFrame,
    df_solving_folds: pd.DataFrame,
    df_raw: pd.DataFrame,
    df_pivot: pd.DataFrame,
    df_summary: pd.DataFrame,
    df_algos: pd.DataFrame = None,
) -> Path:
    """Write the fully-detailed report xlsx."""
    output_dir = experiment_path / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fully-detailed-report.xlsx"

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:

        # --- Sheet 1: run_params ---
        params_df = pd.DataFrame([
            {"Parameter": str(k), "Value": json.dumps(v) if isinstance(v, (list, dict)) else str(v)}
            for k, v in run_params.items()
        ])
        _write_df_to_sheet(writer, params_df, "run_params", title="Run Parameters")

        # --- Per-algorithm sheets (base + each algorithm's own extras) ---
        _write_per_algorithm_sheets(writer, df_algos)

        # --- Sheet 2: cs_traversal_metrics ---
        if not df_traversal.empty:
            _write_df_to_sheet(
                writer, df_traversal,
                "cs_traversal_metrics",
                title="Conflict Search Traversal Metrics (per fold)",
            )

        # --- Sheet 3: solving_folds_summary ---
        if not df_solving_folds.empty:
            _write_df_to_sheet(
                writer, df_solving_folds,
                "solving_folds_summary",
                title="Folds with solving_ratio > 0",
            )

        # --- Sheet 4: raw_data (CDPS CFM solutions; skipped for baseline-only runs) ---
        if not df_raw.empty:
            cols = [c for c in RAW_COLUMNS if c in df_raw.columns]
            _write_df_to_sheet(writer, df_raw[cols], "raw_data", title="All CFM Solutions — Raw Data")

        # --- Sheet 3: pivot ---
        if not df_pivot.empty:
            _write_df_to_sheet(writer, df_pivot, "pivot", title="Pivot: numtrajs × fold × solution_index")

        # --- Sheet 4: correlation ---
        if scipy_stats is not None and not df_raw.empty:
            sheet_name = "correlation"
            # Create the sheet first
            pd.DataFrame().to_excel(writer, sheet_name=sheet_name)

            row = 0

            # 4a: Metric descriptives
            desc_df = build_metric_descriptives(df_raw)
            row = _write_df_to_sheet(writer, desc_df, sheet_name, title="Metric Descriptives", start_row=row)
            row += 2

            # 4b: Global correlations
            global_corr = build_global_correlations(df_raw)
            row = _write_df_to_sheet(writer, global_corr, sheet_name, title="Global: fluent_patch_count vs pred_* metrics", start_row=row)
            row += 2

            # 4c: Per-numtrajs correlations for key metrics
            for metric in ["pred_app_precision", "pred_eff_precision", "pred_eff_recall"]:
                per_nt = build_per_numtrajs_correlations(df_raw, metric)
                row = _write_df_to_sheet(writer, per_nt, sheet_name, title=f"Per-numtrajs: fluent_patch_count vs {metric}", start_row=row)
                row += 2

            # 4d: Mann-Whitney
            mw = build_mann_whitney(df_raw)
            if not mw.empty:
                row = _write_df_to_sheet(writer, mw, sheet_name, title="Mann-Whitney U: pred_* for solving vs non-solving models", start_row=row)
                row += 2

            # 4e: pred_* vs solving outcomes
            pred_solve = build_pred_vs_solving_correlations(df_raw)
            if not pred_solve.empty:
                row = _write_df_to_sheet(writer, pred_solve, sheet_name, title="Spearman: pred_* → solving outcomes", start_row=row)

        # --- Sheet 5: summary ---
        if not df_summary.empty:
            _write_df_to_sheet(writer, df_summary, "summary", title="Per-numtrajs Aggregates")

    return output_path


# =============================================================================
# MAIN
# =============================================================================

def generate_experiment_report(experiment_path: Path) -> Path:
    """Main entry point: generate the fully-detailed report for an experiment."""
    testing_dir = experiment_path / "testing"
    if not testing_dir.exists():
        raise FileNotFoundError(f"No testing/ directory found in {experiment_path}")

    # Load run params
    run_params_path = experiment_path / "evaluation_results" / "run_params.json"
    if run_params_path.exists():
        run_params = _load_json(run_params_path)
    else:
        # Try experiment root
        alt = experiment_path / "run_params.json"
        run_params = _load_json(alt) or {"note": "run_params.json not found"}

    # Collect traversal metrics
    traversal_rows = collect_traversal_metrics(testing_dir)
    df_traversal = pd.DataFrame(traversal_rows)

    # Collect solving folds
    solving_rows = collect_solving_folds(testing_dir)
    df_solving_folds = pd.DataFrame(solving_rows)

    # Collect raw CFM data (CDPS only; empty for baseline-only runs).
    rows = collect_raw_rows(testing_dir)
    df_raw = pd.DataFrame(rows)
    if not df_raw.empty:
        df_raw = df_raw.sort_values(["numtrajs", "fold", "solution_index"]).reset_index(drop=True)
        df_pivot = build_pivot(df_raw)
        df_summary = build_summary(df_raw)
    else:
        df_pivot = pd.DataFrame()
        df_summary = pd.DataFrame()

    # Per-algorithm summary rows (base + algorithm_specific extras), all algorithms.
    from benchmark.experiment_running_helpers.collect_results import collect_results
    df_algos = collect_results(experiment_path)

    if df_raw.empty and df_algos.empty:
        raise ValueError(f"No results found in {testing_dir}")

    # Write
    output_path = write_report(experiment_path, run_params, df_traversal,
                               df_solving_folds, df_raw, df_pivot, df_summary, df_algos)
    print(f"Report written to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate fully-detailed experiment report (xlsx)")
    parser.add_argument("experiment_path", type=str, help="Path to experiment directory")
    args = parser.parse_args()

    experiment_path = Path(args.experiment_path)
    if not experiment_path.exists():
        print(f"Error: {experiment_path} does not exist", file=sys.stderr)
        sys.exit(1)

    generate_experiment_report(experiment_path)


if __name__ == "__main__":
    main()
