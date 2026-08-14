"""Read-only cProfile harness for CDPS on individual experiment fold cells.

This replays the Conflict-Directed Patch Search (CDPS) for a single, already-run
fold *without writing anything back into the fold*. It reads the frozen
``original_observations/`` and ``domain_reference.pddl`` from the cell, rebuilds
the exact ``CDPSConfig`` from the experiment's ``run_params.json``, and runs
``ConflictDrivenPatchSearch.run(...)`` inside ``cProfile``.

Read-only guarantee: the search is constructed with ``conflict_free_models_dir=None``
and ``save_t_prime_fn=None`` and no ``on_node_expanded`` callback, so every disk
write inside CDPS (all guarded by ``conflict_free_models_dir``) is disabled. The
only file produced is the ``.prof`` dump, written to a separate collection dir.

Usage:
    # single cell
    python -m benchmark.diagnosis.profile_cdps_fold --cell <path/to/foldN_numtrajsK_gtrateG>

    # predefined npuzzle matrix (noise=0.2, fold0, numtrajs 3/5/7, masks 0.0/0.01/0.1)
    python -m benchmark.diagnosis.profile_cdps_fold --driver

Then inspect with: snakeviz <prof_file>

Note (flagged decision): the CDPS default logger streams at DEBUG level. To keep
the profile focused on real search work (and the console usable across long
runs), this harness passes a WARNING-level logger. This changes logging only, not
search behavior.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root is importable when run as a plain script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Observation

from src.plan_denoising.conflict_search import ConflictDrivenPatchSearch
from src.plan_denoising.conflict_search_config import CDPSConfig
from src.utils.masking import load_masked_observation

# Default output collection dir for the npuzzle noise=0.2 study.
_DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "cdps_profiles_npuzzle_noise0.2"
_NPUZZLE_RESULTS_ROOT = _PROJECT_ROOT / "benchmark" / "running_results" / "npuzzle"


def _quiet_logger() -> logging.Logger:
    """Return a WARNING-level logger so profiling reflects search work, not logging."""
    logger = logging.getLogger("cdps_profile")
    logger.setLevel(logging.WARNING)
    return logger


def _experiment_dir_for_cell(cell_dir: Path) -> Path:
    """Return the experiment root for a cell at ``<exp>/testing/<cell>``."""
    return cell_dir.parent.parent


def load_cdps_config_from_run_params(run_params_path: Path) -> Tuple[CDPSConfig, int]:
    """Rebuild the CDPSConfig and learning timeout from an experiment's run_params.json.

    Args:
        run_params_path: Path to ``evaluation_results/run_params.json``.

    Returns:
        Tuple of (CDPSConfig, learning_timeout_seconds).
    """
    params = json.loads(run_params_path.read_text())
    config = CDPSConfig(
        fluent_patch_cost=params["fluent_patch_cost"],
        fluent_patch_weight=params["fluent_patch_weight"],
        model_patch_cost=params["model_patch_cost"],
        model_constraint_weight=params["model_constraint_weight"],
        max_search_nodes=params.get("max_search_nodes"),
        search_mode=params["search_mode"],
        node_choosing_strategy=params["node_choosing_strategy"],
        conflict_group_strategy=params["conflict_group_strategy"],
        fluent_branch_mode=params["fluent_branch_mode"],
    )
    timeout_seconds = int(params["learning_timeout_seconds"])
    return config, timeout_seconds


def load_fold_observations(cell_dir: Path, domain: Domain) -> List[Observation]:
    """Load all masked observations from a cell's frozen original_observations/ (read-only).

    Args:
        cell_dir: The fold cell directory.
        domain: Partially parsed PDDL domain for predicate/action parsing.

    Returns:
        List of fully grounded, masked Observations (one per problem).
    """
    obs_dir = cell_dir / "original_observations"
    observations: List[Observation] = []
    for traj_path in sorted(obs_dir.glob("*.trajectory")):
        masking_info_path = traj_path.with_suffix(".masking_info")
        if not masking_info_path.exists():
            continue
        observations.append(load_masked_observation(traj_path, masking_info_path, domain))
    if not observations:
        raise FileNotFoundError(f"No (trajectory, masking_info) pairs found in {obs_dir}")
    return observations


def profile_cell(
    cell_dir: Path,
    out_path: Path,
    timeout_override: Optional[int] = None,
) -> Path:
    """Profile a single CDPS fold run, writing only a .prof file (no fold writes).

    Args:
        cell_dir: The fold cell directory (contains original_observations/ + domain_reference.pddl).
        out_path: Where to write the cProfile stats dump.
        timeout_override: Optional timeout override; defaults to the experiment's learning timeout.

    Returns:
        The path to the written .prof file.
    """
    exp_dir = _experiment_dir_for_cell(cell_dir)
    run_params_path = exp_dir / "evaluation_results" / "run_params.json"
    config, timeout_seconds = load_cdps_config_from_run_params(run_params_path)
    if timeout_override is not None:
        timeout_seconds = timeout_override

    domain_ref = cell_dir / "domain_reference.pddl"
    domain = DomainParser(domain_ref, partial_parsing=True).parse_domain()
    observations = load_fold_observations(cell_dir, domain)

    # Read-only search: no conflict_free_models_dir, no save_t_prime_fn -> zero disk writes.
    search = ConflictDrivenPatchSearch(
        partial_domain_template=deepcopy(domain),
        negative_preconditions_policy=config.negative_precondition_policy,
        seed=config.seed,
        logger=_quiet_logger(),
        search_mode=config.search_mode,
        conflict_free_models_dir=None,
        save_t_prime_fn=None,
        fluent_patch_cost=config.fluent_patch_cost,
        fluent_patch_weight=config.fluent_patch_weight,
        model_patch_cost=config.model_patch_cost,
        model_constraint_weight=config.model_constraint_weight,
        node_choosing_strategy=config.node_choosing_strategy,
        conflict_group_strategy=config.conflict_group_strategy,
        fluent_branch_mode=config.fluent_branch_mode,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[profile] {cell_dir.name} (mask/noise exp={exp_dir.name}) "
        f"| {len(observations)} obs | timeout={timeout_seconds}s -> {out_path.name}"
    )
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        search.run(
            observations=observations,
            max_nodes=config.max_search_nodes,
            timeout_seconds=timeout_seconds,
            gt_source_indices_by_obs=None,  # gtrate=0 cells carry no GT source states
            on_node_expanded=None,
        )
    finally:
        profiler.disable()
        profiler.dump_stats(str(out_path))
    print(f"[profile] wrote {out_path}")
    return out_path


def _cfm_found(cell_dir: Path) -> bool:
    """True if the cell's conflict_free_solutions_log.json is present and non-empty."""
    log_path = cell_dir / "conflict_free_solutions_log.json"
    if not log_path.exists():
        return False
    try:
        return len(json.loads(log_path.read_text())) > 0
    except (json.JSONDecodeError, OSError):
        return False


def run_driver(out_dir: Path, timeout_override: Optional[int] = None) -> List[Path]:
    """Profile the predefined npuzzle matrix: noise=0.2, fold0, numtrajs 3/5/7, masks 0.0/0.01/0.1.

    Args:
        out_dir: Directory to collect the .prof files.
        timeout_override: Optional per-cell timeout override (seconds).

    Returns:
        List of written .prof paths.
    """
    masks = ["0.0", "0.01", "0.1"]
    noise = "0.2"
    num_trajs = [3, 5, 7]
    fold = 0

    written: List[Path] = []
    for mask in masks:
        exp_dir = _NPUZZLE_RESULTS_ROOT / f"simulation-checkup-after-gtfixed__mask={mask}__noise={noise}"
        for nt in num_trajs:
            cell_dir = exp_dir / "testing" / f"fold{fold}_numtrajs{nt}_gtrate0"
            if not cell_dir.exists():
                print(f"[driver] SKIP missing cell: {cell_dir}")
                continue
            status = "CFM" if _cfm_found(cell_dir) else "none"
            out_path = out_dir / f"mask{mask}_nt{nt}_fold{fold}_{status}.prof"
            written.append(profile_cell(cell_dir, out_path, timeout_override))
    return written


def _print_snakeviz_commands(prof_paths: List[Path]) -> None:
    """Print ready-to-run snakeviz commands for the generated .prof files."""
    if not prof_paths:
        return
    print("\n=== snakeviz commands ===")
    for p in prof_paths:
        print(f"snakeviz {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only cProfile harness for CDPS fold cells.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cell", type=Path, help="Path to a single fold cell directory to profile.")
    group.add_argument(
        "--driver",
        action="store_true",
        help="Profile the predefined npuzzle noise=0.2 matrix (fold0, numtrajs 3/5/7, masks 0.0/0.01/0.1).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help="Collection dir for .prof files (driver mode) or parent for a single cell's .prof.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Explicit .prof path for --cell mode.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Override the CDPS timeout in seconds (default: experiment's learning timeout).",
    )
    args = parser.parse_args()

    if args.driver:
        written = run_driver(args.out_dir, timeout_override=args.timeout)
    else:
        cell_dir = args.cell.resolve()
        if args.out is not None:
            out_path = args.out
        else:
            status = "CFM" if _cfm_found(cell_dir) else "none"
            out_path = args.out_dir / f"{cell_dir.name}_{status}.prof"
        written = [profile_cell(cell_dir, out_path, timeout_override=args.timeout)]

    _print_snakeviz_commands(written)


if __name__ == "__main__":
    main()
