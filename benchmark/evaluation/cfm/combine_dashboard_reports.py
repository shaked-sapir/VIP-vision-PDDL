"""Combine fully-detailed experiment reports for dashboard-config experiments.

Reads ``dashboard_config.yaml`` (image + simulation experiment paths), ensures
each experiment has ``fully-detailed-report.xlsx``, then writes a single workbook
with two pivot-ready sheets:

    imaged     — raw CFM rows from all image experiments
    simulated  — raw CFM rows from all simulation grid cells (+ p_mask, p_noise)

Usage:
    python -m benchmark.evaluation.cfm.combine_dashboard_reports
    python -m benchmark.evaluation.cfm.combine_dashboard_reports --skip-regenerate
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from benchmark.evaluation.experiment_report import (
    RAW_COLUMNS,
    _parse_fold_dir_name,
    collect_raw_rows,
    generate_experiment_report,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = Path(__file__).resolve().parent / "dashboard_config.yaml"
_CELL_PATTERN = re.compile(r"mask=([0-9.]+)__noise=([0-9.]+)")

_META_COLUMNS = ["domain", "experiment_name", "p_mask", "p_noise", "gt_rate"]
_OUTPUT_NAME = "combined_experiment_reports.xlsx"


def _parse_cell(dir_name: str) -> Optional[Tuple[str, str]]:
    match = _CELL_PATTERN.search(dir_name)
    return (match.group(1), match.group(2)) if match else None


def _find_grid_cells(domain_dir: Path, prefix: str) -> List[Path]:
    cells = [
        c for c in domain_dir.iterdir()
        if c.is_dir() and c.name.startswith(prefix) and _parse_cell(c.name)
    ]
    return sorted(cells, key=lambda p: tuple(float(x) for x in _parse_cell(p.name)))


def _load_dashboard_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)["results_dashboard"]


def _gt_rate_map(testing_dir: Path) -> Dict[Tuple[int, int], int]:
    mapping: Dict[Tuple[int, int], int] = {}
    for fold_dir in testing_dir.iterdir():
        if not fold_dir.is_dir():
            continue
        info = _parse_fold_dir_name(fold_dir.name)
        if info is None:
            continue
        mapping[(info["fold"], info["numtrajs"])] = info["gt_rate"]
    return mapping


def _collect_enriched_rows(
    experiment_path: Path,
    domain: str,
    p_mask: Optional[float] = None,
    p_noise: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Collect raw_data rows with experiment-level metadata columns prepended."""
    testing_dir = experiment_path / "testing"
    rows = collect_raw_rows(testing_dir)
    gt_map = _gt_rate_map(testing_dir)

    enriched: List[Dict[str, Any]] = []
    for row in rows:
        out: Dict[str, Any] = {
            "domain": domain,
            "experiment_name": experiment_path.name,
            "p_mask": p_mask,
            "p_noise": p_noise,
            "gt_rate": gt_map.get((row["fold"], row["numtrajs"])),
        }
        for col in RAW_COLUMNS:
            out[col] = row.get(col)
        enriched.append(out)
    return enriched


def _discover_experiments(cfg: dict, project_root: Path) -> Tuple[List[dict], List[dict]]:
    results_root = project_root / cfg["results_root"]
    sim_experiments: List[dict] = []
    image_experiments: List[dict] = []

    prefixes = cfg.get("simulation", {}).get("prefix", {})
    for domain in cfg.get("domains", []):
        prefix = prefixes.get(domain)
        if prefix:
            domain_dir = results_root / domain
            for cell in _find_grid_cells(domain_dir, prefix):
                m, n = _parse_cell(cell.name)
                sim_experiments.append({
                    "path": cell,
                    "domain": domain,
                    "p_mask": float(m),
                    "p_noise": float(n),
                })

    for domain, rel in cfg.get("image", {}).get("experiment_dir", {}).items():
        image_experiments.append({
            "path": project_root / rel.strip(),
            "domain": domain,
        })

    return sim_experiments, image_experiments


def _ensure_report(exp_path: Path, skip_regenerate: bool) -> None:
    report = exp_path / "evaluation_results" / "fully-detailed-report.xlsx"
    if skip_regenerate and report.is_file():
        return
    generate_experiment_report(exp_path)


def _build_sheet(experiments: List[dict], skip_regenerate: bool) -> pd.DataFrame:
    all_rows: List[Dict[str, Any]] = []
    column_order = _META_COLUMNS + [c for c in RAW_COLUMNS if c not in _META_COLUMNS]

    for spec in experiments:
        exp_path: Path = spec["path"]
        if not (exp_path / "testing").is_dir():
            print(f"  SKIP (no testing/): {exp_path}")
            continue
        try:
            _ensure_report(exp_path, skip_regenerate)
            rows = _collect_enriched_rows(
                exp_path,
                domain=spec["domain"],
                p_mask=spec.get("p_mask"),
                p_noise=spec.get("p_noise"),
            )
            all_rows.extend(rows)
            print(f"  OK {spec['domain']}/{exp_path.name}: {len(rows)} CFM rows")
        except Exception as exc:
            print(f"  FAIL {spec['domain']}/{exp_path.name}: {exc}")

    if not all_rows:
        return pd.DataFrame(columns=column_order)

    df = pd.DataFrame(all_rows)
    for col in column_order:
        if col not in df.columns:
            df[col] = None
    return df[column_order].sort_values(
        ["domain", "experiment_name", "numtrajs", "fold", "gt_rate", "solution_index"],
        na_position="last",
    ).reset_index(drop=True)


def combine_dashboard_reports(
    config_path: Path = _DEFAULT_CONFIG,
    output_path: Optional[Path] = None,
    skip_regenerate: bool = False,
) -> Path:
    """Generate per-experiment reports and write the combined workbook."""
    cfg = _load_dashboard_config(config_path)
    sim_experiments, image_experiments = _discover_experiments(cfg, _PROJECT_ROOT)

    if output_path is None:
        output_path = _PROJECT_ROOT / cfg["results_root"] / _OUTPUT_NAME

    print(f"Image experiments: {len(image_experiments)}")
    df_image = _build_sheet(image_experiments, skip_regenerate)

    print(f"\nSimulation experiments: {len(sim_experiments)}")
    df_sim = _build_sheet(sim_experiments, skip_regenerate)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_image.to_excel(writer, sheet_name="imaged", index=False)
        df_sim.to_excel(writer, sheet_name="simulated", index=False)

    print(f"\nCombined report written to: {output_path}")
    print(f"  imaged:    {len(df_image)} rows")
    print(f"  simulated: {len(df_sim)} rows")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine dashboard experiment reports into one xlsx")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG, help="Path to dashboard_config.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output xlsx path (default: <results_root>/{_OUTPUT_NAME})",
    )
    parser.add_argument(
        "--skip-regenerate",
        action="store_true",
        help="Reuse existing fully-detailed-report.xlsx files",
    )
    args = parser.parse_args()

    if not args.config.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    combine_dashboard_reports(
        config_path=args.config,
        output_path=args.output,
        skip_regenerate=args.skip_regenerate,
    )


if __name__ == "__main__":
    main()
