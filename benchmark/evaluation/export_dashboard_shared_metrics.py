"""Export cross-domain shared-metrics workbook from dashboard-config experiments.

Stacks every algorithm row from each experiment listed in
``dashboard_config.yaml`` (simulation grid cells + image experiments) into:

    benchmark/evaluation/raw_data/all_domains_shared_metrics.xlsx

Only schema base fields are kept (``BASE_FIELDS`` + run context), so CDPS,
ROSAME, and every ``CDPS_MILP_*`` variant sit side by side. Per-domain
``<domain>_shared_metrics.xlsx`` workbooks are written too.

Usage:
    python -m benchmark.evaluation.export_dashboard_shared_metrics
    python -m benchmark.evaluation.export_dashboard_shared_metrics --modes simulation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from benchmark.evaluation.cfm.combine_dashboard_reports import (
    _discover_experiments,
    _load_dashboard_config,
)
from benchmark.experiment_running_helpers.collect_results import collect_results
from benchmark.experiment_running_helpers.result_schema import (
    BASE_FIELDS,
    RUN_CONTEXT_FIELDS,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = (
    Path(__file__).resolve().parent / "cfm" / "dashboard_config.yaml"
)
_DEFAULT_OUT = Path(__file__).resolve().parent / "raw_data"
_COMBINED_NAME = "all_domains_shared_metrics.xlsx"

_LEAD_COLS = [
    "mode", "domain", "experiment_name", "p_mask", "p_noise", "algorithm",
]


def _shared_metrics_frame(df_algos: pd.DataFrame) -> pd.DataFrame:
    """Keep only shared/base columns present in ``df_algos``, stable order."""
    cols = [c for c in BASE_FIELDS + RUN_CONTEXT_FIELDS if c in df_algos.columns]
    out = df_algos[cols].copy()
    sort_keys = [k for k in ("num_trajectories", "gt_rate", "fold", "algorithm")
                 if k in out.columns]
    if sort_keys:
        out = out.sort_values(sort_keys).reset_index(drop=True)
    return out


def _enrich_shared(
    df: pd.DataFrame,
    *,
    domain: str,
    mode: str,
    experiment_name: str,
    p_mask: Optional[float],
    p_noise: Optional[float],
) -> pd.DataFrame:
    """Attach mode / cell metadata; force domain from the dashboard entry."""
    if df.empty:
        return df
    out = df.copy()
    out["domain"] = domain
    out["mode"] = mode
    out["experiment_name"] = experiment_name
    out["p_mask"] = p_mask
    out["p_noise"] = p_noise
    lead = [c for c in _LEAD_COLS if c in out.columns]
    rest = [c for c in out.columns if c not in lead]
    return out[lead + rest]


def _write_combined(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="shared_metrics", index=False)


def _write_domain_workbook(df: pd.DataFrame, path: Path) -> None:
    """Mean-over-folds + per-fold sheets for one domain."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            pd.DataFrame({"note": ["no shared-metric rows"]}).to_excel(
                writer, sheet_name="per_fold", index=False,
            )
        return

    group = [c for c in (
        "mode", "domain", "experiment_name", "p_mask", "p_noise",
        "algorithm", "num_trajectories", "gt_rate",
    ) if c in df.columns]
    numeric = df.select_dtypes("number").columns.tolist()
    skip = {"fold", "num_trajectories", "gt_rate", "p_mask", "p_noise"}
    agg_metrics = [c for c in numeric if c not in skip]

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        if group and agg_metrics:
            mean_df = (
                df.groupby(group, dropna=False)[agg_metrics]
                .mean(numeric_only=True)
                .reset_index()
            )
            mean_df.to_excel(writer, sheet_name="mean_over_folds", index=False)
        df.to_excel(writer, sheet_name="per_fold", index=False)


def export_dashboard_shared_metrics(
    config_path: Path = _DEFAULT_CONFIG,
    out_dir: Path = _DEFAULT_OUT,
    modes: Optional[List[str]] = None,
) -> Path:
    """Build combined (+ per-domain) shared-metrics workbooks from fold_result.json."""
    cfg = _load_dashboard_config(config_path)
    sim_experiments, image_experiments = _discover_experiments(cfg, _PROJECT_ROOT)
    want = set(modes or ["simulation", "image"])

    all_frames: List[pd.DataFrame] = []
    by_domain: dict[str, List[pd.DataFrame]] = {}

    specs: List[tuple[str, dict]] = []
    if "simulation" in want:
        specs.extend(("simulation", s) for s in sim_experiments)
    if "image" in want:
        specs.extend(("image", s) for s in image_experiments)

    for mode, spec in specs:
        exp_path: Path = spec["path"]
        domain = spec["domain"]
        if not (exp_path / "testing").is_dir():
            print(f"  SKIP (no testing/): {exp_path}")
            continue
        try:
            df = _shared_metrics_frame(collect_results(exp_path))
            df = _enrich_shared(
                df,
                domain=domain,
                mode=mode,
                experiment_name=exp_path.name,
                p_mask=spec.get("p_mask"),
                p_noise=spec.get("p_noise"),
            )
        except Exception as exc:  # noqa: BLE001 — keep exporting other cells
            print(f"  FAIL {domain}/{exp_path.name}: {exc}")
            continue
        if df.empty:
            print(f"  SKIP (empty): {domain}/{exp_path.name}")
            continue
        algs = sorted(df["algorithm"].dropna().astype(str).unique())
        print(f"  OK {mode}/{domain}/{exp_path.name}: {len(df)} rows, algs={algs}")
        all_frames.append(df)
        by_domain.setdefault(domain, []).append(df)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for domain, frames in by_domain.items():
        frames = [f for f in frames if not f.empty]
        if not frames:
            continue
        df_domain = pd.concat(frames, ignore_index=True, sort=False)
        path = out_dir / domain / f"{domain}_shared_metrics.xlsx"
        _write_domain_workbook(df_domain, path)
        print(f"  wrote {path.relative_to(_PROJECT_ROOT)} ({len(df_domain)} rows)")

    combined_path = out_dir / _COMBINED_NAME
    all_frames = [f for f in all_frames if not f.empty]
    if not all_frames:
        print("No rows collected; nothing written.", file=sys.stderr)
        return combined_path

    combined = pd.concat(all_frames, ignore_index=True, sort=False)
    sort_keys = [c for c in (
        "mode", "domain", "experiment_name", "p_mask", "p_noise",
        "algorithm", "num_trajectories", "gt_rate", "fold",
    ) if c in combined.columns]
    if sort_keys:
        combined = combined.sort_values(sort_keys).reset_index(drop=True)
    _write_combined(combined, combined_path)
    algs = sorted(combined["algorithm"].dropna().astype(str).unique())
    print(f"\nWrote {combined_path} — {len(combined)} rows, algorithms={algs}")
    return combined_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    ap.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    ap.add_argument(
        "--modes", nargs="+", choices=["simulation", "image"], default=None,
        help="Restrict to these modes (default: both).",
    )
    args = ap.parse_args()
    if not args.config.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    export_dashboard_shared_metrics(
        config_path=args.config,
        out_dir=args.out_dir,
        modes=args.modes,
    )


if __name__ == "__main__":
    main()
