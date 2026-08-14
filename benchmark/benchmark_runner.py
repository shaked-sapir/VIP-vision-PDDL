"""Config-driven batch benchmark runner.

Reads a YAML run config and runs one benchmark experiment per cell, delegating
each cell to ``experiment_runner.main(...)`` (the single-experiment runner).

The default config is the single editable file ``benchmark/run_config.yaml``.
Edit it in place with the desired ``run_name`` and params, then run the batch —
no need to create a new file per run. ``--config`` can override the default.

Two sources are supported via the top-level ``source`` field:
    - ``image``     — real image-pipeline data (ImageDataSource); one cell per domain.
    - ``simulated`` — synthetic in-memory noise (SimulatedDataSource); cells are
      the cartesian product of domains × grid.masking_ps × grid.noising_ps.
      GT trajectories are auto-discovered from ``<data_dir>/gt_trajectories/``.
      A pre-flight check validates every simulated cell's GT trajectories before
      anything runs; if any are missing it aborts up front, printing the exact
      generation command for each offending data_dir.

Results are written by experiment_runner.main to:
    benchmark/running_results/<domain>/<experiment_name>/
A timestamped run manifest (params + per-domain output paths) is archived to:
    benchmark/finished_run_configs/<run_name>/<timestamp>.json
Runs where no cell succeeded (e.g. missing GT trajectories) are archived under:
    benchmark/finished_run_configs/failed_runs/<run_name>/<timestamp>.json

Usage:
    python -m benchmark.benchmark_runner [--config path/to/run_config.yaml] [--dry-run]

Example run config (simulated):
    run_name: sim_run_2026_07
    source: simulated
    shared:
      n_folds: 5
      num_trajectories: [3, 5, 8]
      gt_rates: [0]
      algorithms: [cdps, rosame]
    simulation:
      masking_strategy: percentage
      noising_strategy: percentage
      seed: 42
      grid:
        masking_ps: [0.2, 0.4]
        noising_ps: [0.1, 0.15]
    domains:
      - {domain_key: blocksworld, data_dir: benchmark/data/blocksworld/<experiment>}
      - {domain_key: hanoi,       data_dir: benchmark/data/hanoi/<experiment>}
"""

import argparse
import json
import re
import sys
import traceback
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

benchmark_path = Path(__file__).resolve().parent
project_root = benchmark_path.parent
sys.path.insert(0, str(project_root))

from benchmark import experiment_runner
from benchmark.algorithms import resolve_algorithms
from benchmark.evaluation.cfm.cfm_quality_analysis import generate_cfm_quality_analysis
from benchmark.evaluation.cfm.cfm_quality_table import generate_cfm_quality_table
from benchmark.evaluation.experiment_report import generate_experiment_report
from benchmark.experiment_running_helpers.data_source import ImageDataSource, SimulatedDataSource
from src.observation_degradation.masking import MaskingType
from src.observation_degradation.noising import NoisingType
from src.plan_denoising.milp_denoiser.config import expand_cdps_milp_ablations
from src.utils.time import create_experiment_timestamp

# Default editable config; override with --config.
DEFAULT_CONFIG_PATH = benchmark_path / "run_config.yaml"
# Where per-run manifests are archived (timestamped, never overwritten).
MANIFEST_ROOT = benchmark_path / "finished_run_configs"
# Runs that never got off the ground (no cell succeeded) go under this subdir.
FAILED_MANIFEST_ROOT = MANIFEST_ROOT / "failed_runs"
# Filesystem-safe timestamp (no ':') for manifest filenames.
_MANIFEST_TIME_FORMAT = "%d-%m-%YT%H-%M-%S"


# ---------------------------------------------------------------------------
# Cell model
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    """Outcome of running a single run cell."""
    domain_key: str
    experiment_name: str
    result_dir: str
    masking_p: Optional[float]
    noising_p: Optional[float]
    status: str            # "ok" | "failed" | "skipped"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Config loading & path resolution
# ---------------------------------------------------------------------------

def _load_config(config_path: Path) -> dict:
    """Parse the YAML run config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Run config must be a mapping, got {type(config)}")
    return config


def _resolve_data_dir(data_dir: str) -> Path:
    """Resolve a (possibly relative) data_dir against the project root."""
    path = Path(data_dir).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _gt_generation_command(data_dir: Path) -> str:
    """The CLI command that generates GT trajectories for a data dir."""
    return f"python -m benchmark.generate_gt_trajectories --experiment-dir {data_dir}"


def _discover_gt_trajectories(data_dir: Path) -> List[Path]:
    """Collect GT .trajectory files from <data_dir>/gt_trajectories/problemN/.

    Mirrors the nested layout written by generate_gt_trajectories.py. When the
    GT trajectories are missing, the raised error carries the exact command that
    generates them for this data_dir.
    """
    gt_root = data_dir / "gt_trajectories"
    hint = f"Generate them with:\n    {_gt_generation_command(data_dir)}"
    if not gt_root.is_dir():
        raise FileNotFoundError(
            f"No gt_trajectories/ dir found under {data_dir}. Expected: {gt_root}\n{hint}"
        )
    paths = sorted(gt_root.glob("*/*.trajectory"), key=_natural_sort_key)
    if not paths:
        raise FileNotFoundError(f"No .trajectory files found under {gt_root}\n{hint}")
    return paths


def _natural_sort_key(path: Path):
    """Order problem1, problem2, ..., problem10 numerically by parent dir name."""
    name = path.parent.name
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', name)]


# ---------------------------------------------------------------------------
# Cell expansion
# ---------------------------------------------------------------------------

def _filter_domains(domains: List[dict], selected: Optional[List[str]]) -> List[dict]:
    """Keep only the domains whose domain_key is in `selected` (order-preserving).

    When `selected` is None/empty, all domains are returned. Unknown keys raise
    a ValueError listing the available domain keys (fail fast on typos).
    """
    if not selected:
        return domains

    available = {d["domain_key"] for d in domains}
    unknown = [key for key in selected if key not in available]
    if unknown:
        raise ValueError(
            f"Unknown domain(s) {unknown} not in config. "
            f"Available: {sorted(available)}"
        )
    selected_set = set(selected)
    return [d for d in domains if d["domain_key"] in selected_set]


def _expand_cells(config: dict, selected_domains: Optional[List[str]] = None) -> List[dict]:
    """Build the list of (domain × config) cells from the run config.

    Each cell is a dict with keys: domain_key, data_dir, masking_p, noising_p.
    For image runs masking_p / noising_p are None. When `selected_domains` is
    given, only those domains are expanded (subset run without editing the file).
    """
    source = config.get("source")
    if source not in ("image", "simulated"):
        raise ValueError(f"source must be 'image' or 'simulated', got {source!r}")

    domains = config.get("domains") or []
    if not domains:
        raise ValueError("Run config has no 'domains'.")
    domains = _filter_domains(domains, selected_domains)

    if source == "image":
        return [
            {"domain_key": d["domain_key"], "data_dir": d["data_dir"],
             "masking_p": None, "noising_p": None}
            for d in domains
        ]

    sim = config.get("simulation") or {}
    grid = sim.get("grid") or {}
    masking_ps = grid.get("masking_ps")
    noising_ps = grid.get("noising_ps")
    if not masking_ps or not noising_ps:
        raise ValueError("simulation.grid must define non-empty masking_ps and noising_ps.")

    cells = []
    for d, mp, np_ in product(domains, masking_ps, noising_ps):
        cells.append({
            "domain_key": d["domain_key"], "data_dir": d["data_dir"],
            "masking_p": mp, "noising_p": np_,
        })
    return cells


def _cell_experiment_name(run_name: str, cell: dict) -> str:
    """Compose a distinct experiment_name for a cell.

    Image cells (no grid) → run_name (domain is already in the result path).
    Simulated cells → run_name__mask=<mp>__noise=<np>.
    """
    if cell["masking_p"] is None:
        return run_name
    return f"{run_name}__mask={cell['masking_p']}__noise={cell['noising_p']}"


def _cell_experiment_paths(run_name: str, cell: dict) -> Tuple[str, str]:
    """Return (experiment_name, result_dir) for a cell.

    Shared by the run loop and the pre-flight abort path so both compose the
    result directory identically.
    """
    experiment_name = _cell_experiment_name(run_name, cell)
    domain_display_name = _safe_display_name(cell["domain_key"])
    result_dir = str(benchmark_path / "running_results" / domain_display_name / experiment_name)
    return experiment_name, result_dir


# ---------------------------------------------------------------------------
# Data source construction
# ---------------------------------------------------------------------------

def _build_data_source(source: str, sim_cfg: dict, data_dir: Path, cell: dict):
    """Construct the DataSource for a cell."""
    if source == "image":
        return ImageDataSource()

    gt_paths = _discover_gt_trajectories(data_dir)
    return SimulatedDataSource(
        gt_trajectory_paths=gt_paths,
        masking_strategy=MaskingType(sim_cfg.get("masking_strategy", "percentage")),
        masking_p=cell["masking_p"],
        noising_strategy=NoisingType(sim_cfg.get("noising_strategy", "percentage")),
        noising_p=cell["noising_p"],
        seed=sim_cfg.get("seed", 42),
    )


# ---------------------------------------------------------------------------
# main() kwargs assembly
# ---------------------------------------------------------------------------

# Shared-config keys that map onto experiment_runner.main() parameters as-is.
_PASSTHROUGH_KEYS = {
    "n_folds", "frame_axiom_mode",
    "learning_timeout_seconds", "planning_timeout_seconds",
    "fluent_patch_cost", "fluent_patch_weight",
    "model_patch_cost", "model_constraint_weight",
    "max_search_nodes", "search_mode", "node_choosing_strategy",
    "conflict_group_strategy", "fluent_branch_mode",
    "normalize", "force_normalize", "events_tracing", "resume",
}


def _build_main_kwargs(shared: dict) -> Dict[str, Any]:
    """Translate the shared config block into experiment_runner.main() kwargs.

    Only keys present in the config are forwarded; anything omitted falls back
    to main()'s own defaults. Unknown keys raise, so a typo (e.g. ``n_fold``) or
    a stale key (e.g. the removed ``mode``) fails loudly instead of being
    silently dropped.
    """
    known_keys = _PASSTHROUGH_KEYS | {
        "num_trajectories", "gt_rates", "algorithms", "baselines", "cdps_milp",
    }
    unknown = set(shared) - known_keys
    if unknown:
        raise ValueError(
            f"Unknown shared config key(s): {sorted(unknown)}. "
            f"Known keys: {sorted(known_keys)}."
        )

    kwargs: Dict[str, Any] = {}

    for key in _PASSTHROUGH_KEYS:
        if key in shared:
            kwargs[key] = shared[key]

    # List params use different names in main().
    if "num_trajectories" in shared:
        kwargs["num_trajectories_list"] = shared["num_trajectories"]
    if "gt_rates" in shared:
        kwargs["gt_rate_percentages"] = shared["gt_rates"]

    # Algorithms: resolve names → (run_cdps, baseline runners). Accepts either
    # the new `algorithms` key or a legacy `baselines` list (implies cdps too).
    if "algorithms" in shared:
        algorithm_names = shared["algorithms"]
    elif "baselines" in shared:
        algorithm_names = ["cdps"] + [b for b in shared["baselines"] if b != "none"]
    else:
        algorithm_names = ["cdps", "rosame"]
    (kwargs["run_cdps"], kwargs["run_cdps_anchored"], kwargs["run_cdps_milp"],
     kwargs["run_cdps_milp_loop"], kwargs["baselines"]) = resolve_algorithms(
        algorithm_names
    )

    # The block both MILP arms share; each arm pins its own variant from the
    # selected key. Expanded here because an `ablations:` sub-block turns it into
    # several arms, and the run banner has to name them before anything runs.
    # Validated (and rejected on a typo) even when both arms are off, so a bad
    # `cdps_milp:` never sits unnoticed in a config that will later enable one.
    kwargs["cdps_milp_configs"] = expand_cdps_milp_ablations(shared.get("cdps_milp"))

    return kwargs


# ---------------------------------------------------------------------------
# Manifest & summary
# ---------------------------------------------------------------------------

def _index_results_by_domain(results: List[CellResult]) -> Dict[str, List[str]]:
    """Map each domain_key to the list of its cells' output result_dirs."""
    by_domain: Dict[str, List[str]] = {}
    for r in results:
        by_domain.setdefault(r.domain_key, []).append(r.result_dir)
    return by_domain


def _write_manifest(
    run_name: str, source: str, shared: dict, sim_cfg: dict,
    results: List[CellResult], selected_domains: Optional[List[str]] = None,
    failed: bool = False,
) -> Path:
    """Archive a timestamped run manifest under finished_run_configs/.

    The manifest records the run params (for reproducibility), the domain subset
    (if any), every cell's outcome, and a per-domain index of output paths.
    Never overwrites — each run gets its own timestamped file. Runs where no
    cell succeeded are archived under finished_run_configs/failed_runs/ instead.
    """
    timestamp = create_experiment_timestamp(_MANIFEST_TIME_FORMAT)
    root = FAILED_MANIFEST_ROOT if failed else MANIFEST_ROOT
    manifest_dir = root / run_name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{timestamp}.json"

    manifest = {
        "run_name": run_name,
        "source": source,
        "generated_at": timestamp,
        "run_status": "failed" if failed else "ok",
        "selected_domains": selected_domains or "all",
        "shared": shared,
        "simulation": sim_cfg if source == "simulated" else None,
        "num_cells": len(results),
        "by_domain": _index_results_by_domain(results),
        "cells": [asdict(r) for r in results],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def _print_summary(results: List[CellResult]) -> None:
    """Print a pass/fail summary of all cells."""
    ok = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status != "ok"]

    print("\n" + "=" * 80)
    print(f"RUN SUMMARY: {len(ok)}/{len(results)} cells succeeded")
    print("=" * 80)
    for r in results:
        marker = "✓" if r.status == "ok" else "✗"
        cfg = "" if r.masking_p is None else f" (mask={r.masking_p}, noise={r.noising_p})"
        line = f"  {marker} {r.domain_key}{cfg} → {r.experiment_name}"
        if r.error:
            line += f"  [{r.status}: {r.error}]"
        print(line)
    print()


def _finalize_run(
    run_name: str, source: str, shared: dict, simulation_config: dict,
    results: List[CellResult], selected_domains: Optional[List[str]],
) -> List[CellResult]:
    """Archive the manifest, print the summary, and return the results.

    A run where no cell succeeded is archived under failed_runs/ (see
    _write_manifest); the printed label reflects that.
    """
    run_failed = not any(r.status == "ok" for r in results)
    manifest_path = _write_manifest(
        run_name, source, shared, simulation_config, results, selected_domains,
        failed=run_failed,
    )
    _print_summary(results)
    label = "Failed-run manifest" if run_failed else "Manifest"
    print(f"{label} written to: {manifest_path}")
    return results


# ---------------------------------------------------------------------------
# Run driver
# ---------------------------------------------------------------------------

def _missing_gt_data_dirs(experiment_cells: List[dict]) -> Dict[str, str]:
    """Return {data_dir: reason} for cells whose GT trajectories are missing.

    Each unique data_dir is checked once (grid expansion repeats it). An empty
    result means every simulated cell has its GT trajectories ready to go.
    """
    missing: Dict[str, str] = {}
    for data_dir_raw in dict.fromkeys(cell["data_dir"] for cell in experiment_cells):
        resolved = _resolve_data_dir(data_dir_raw)
        if not resolved.exists():
            missing[data_dir_raw] = f"data_dir not found: {resolved}"
            continue
        try:
            _discover_gt_trajectories(resolved)
        except FileNotFoundError as exc:
            missing[data_dir_raw] = str(exc)
    return missing


def _abort_on_missing_gt(
    run_name: str, source: str, shared: dict, simulation_config: dict,
    experiment_cells: List[dict], missing_gt: Dict[str, str],
    selected_domains: Optional[List[str]],
) -> List[CellResult]:
    """Report missing GT trajectories and finalize a failed run without running.

    Prints the fix command for every offending data_dir, records each cell as
    failed, and writes a failed-run manifest. No experiment is executed.
    """
    print("\n" + "=" * 80)
    print("PRE-FLIGHT FAILED — missing GT trajectories; nothing will run.")
    print("=" * 80)
    for data_dir_raw, reason in missing_gt.items():
        print(f"\n- {data_dir_raw}:\n    {reason}")

    results = []
    for cell in experiment_cells:
        experiment_name, result_dir = _cell_experiment_paths(run_name, cell)
        results.append(CellResult(
            domain_key=cell["domain_key"], experiment_name=experiment_name,
            result_dir=result_dir, masking_p=cell["masking_p"], noising_p=cell["noising_p"],
            status="failed",
            error=missing_gt.get(cell["data_dir"], "not run: missing GT for another domain"),
        ))
    return _finalize_run(run_name, source, shared, simulation_config, results, selected_domains)


def _generate_cell_reports(result_dir: Path) -> None:
    """Build the xlsx report + CFM quality plots for a completed cell.

    Best-effort: each report is generated independently and a failure is logged
    but never propagated — a broken report must not fail an experiment that
    already succeeded.
    """
    for label, generate in (
        ("experiment report", generate_experiment_report),
        ("CFM quality analysis", generate_cfm_quality_analysis),
        ("CFM quality table", generate_cfm_quality_table),
    ):
        try:
            print(f"\nGenerating {label} for {result_dir} ...")
            generate(result_dir)
        except Exception as exc:  # noqa: BLE001 — report failure must not fail the cell
            print(f"  ! Skipped {label}: {exc}")


def execute_run(
    run_config: dict, dry_run: bool = False, selected_domains: Optional[List[str]] = None,
) -> List[CellResult]:
    """Run a benchmark configuration as a batch of experiments — one per cell.

    A run config is expanded into "cells", where each cell is a single
    experiment (a domain paired with one parameter combination). Every cell is
    executed independently via ``experiment_runner.main``; a failed cell is
    recorded and the run continues. Afterwards a timestamped manifest and a
    pass/fail summary are written.

    Args:
        run_config: Parsed run config (see module docstring for the schema).
        dry_run: If True, print the configurations that would run and return
            without executing anything.
        selected_domains: Optional subset of domain_keys to run. None → all
            domains defined in the config.

    Returns:
        One CellResult per executed cell (empty on a dry run).
    """
    run_name = run_config.get("run_name")
    if not run_name:
        raise ValueError("Run config must define 'run_name'.")
    source = run_config.get("source")
    shared = run_config.get("shared") or {}
    simulation_config = run_config.get("simulation") or {}

    experiment_cells = _expand_cells(run_config, selected_domains)
    domains_label = "all domains" if not selected_domains else f"domains={selected_domains}"
    print(f"Run '{run_name}' (source={source}, {domains_label}): {len(experiment_cells)} cell(s)")

    if dry_run:
        for cell in experiment_cells:
            experiment_name = _cell_experiment_name(run_name, cell)
            print(f"  - {cell['domain_key']}: {experiment_name}  (data_dir={cell['data_dir']})")
        return []

    # Pre-flight: simulated runs need GT trajectories for every cell. Abort the
    # whole run up front (before any experiment) if any are missing, listing the
    # generation command for each offending data_dir at once.
    if source == "simulated":
        missing_gt = _missing_gt_data_dirs(experiment_cells)
        if missing_gt:
            return _abort_on_missing_gt(
                run_name, source, shared, simulation_config,
                experiment_cells, missing_gt, selected_domains,
            )

    shared_experiment_kwargs = _build_main_kwargs(shared)
    results: List[CellResult] = []

    for cell_index, cell in enumerate(experiment_cells, start=1):
        domain_key = cell["domain_key"]
        experiment_name, result_dir = _cell_experiment_paths(run_name, cell)

        print("\n" + "=" * 80)
        print(f"[cell {cell_index}/{len(experiment_cells)}] domain={domain_key} experiment={experiment_name}")
        print("=" * 80)

        try:
            data_dir = _resolve_data_dir(cell["data_dir"])
            if not data_dir.exists():
                raise FileNotFoundError(f"data_dir not found: {data_dir}")

            data_source = _build_data_source(source, simulation_config, data_dir, cell)

            experiment_runner.main(
                domain_key=domain_key,
                data_dir=data_dir,
                data_source=data_source,
                experiment_name=experiment_name,
                **shared_experiment_kwargs,
            )
            results.append(CellResult(
                domain_key=domain_key, experiment_name=experiment_name, result_dir=result_dir,
                masking_p=cell["masking_p"], noising_p=cell["noising_p"], status="ok",
            ))
            _generate_cell_reports(Path(result_dir))
        except Exception as exc:  # noqa: BLE001 — run must survive a failed cell
            traceback.print_exc()
            results.append(CellResult(
                domain_key=domain_key, experiment_name=experiment_name, result_dir=result_dir,
                masking_p=cell["masking_p"], noising_p=cell["noising_p"],
                status="failed", error=str(exc),
            ))

    return _finalize_run(run_name, source, shared, simulation_config, results, selected_domains)


def _safe_display_name(domain_key: str) -> str:
    """Resolve the domain's display name (result-dir component); fall back to key."""
    try:
        return experiment_runner._resolve_domain_config(domain_key)["display_name"]
    except Exception:  # noqa: BLE001 — manifest path is best-effort before the run
        return domain_key


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Config-driven batch benchmark runner.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help=f"Path to the YAML run config (default: {DEFAULT_CONFIG_PATH}).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print all configurations (domain × params) that would run, "
                             "without executing them.")
    parser.add_argument("--domains", nargs="*", default=None, metavar="DOMAIN_KEY",
                        help="Subset of domain_keys to run (space-separated). "
                             "Omit to run all domains in the config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = _load_config(args.config)
    results = execute_run(config, dry_run=args.dry_run, selected_domains=args.domains)

    if results and any(r.status != "ok" for r in results):
        sys.exit(1)
