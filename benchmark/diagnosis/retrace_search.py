"""
retrace_search.py — Re-run a fold's conflict-driven patch search with full tracing.

Produces a search_trace.json file that can be visualized with visualize_trace.py.

This script is intentionally decoupled from trace_spurious_effects.py: it takes only
the fold directory and domain — no action/predicate needed. Filtering and diagnosis
happen downstream (in the HTML viewer or the diagnosis script).

Observations are loaded from the fold's ``original_observations/`` directory, which
contains the noised/masked observations that the original experiment actually fed to
the conflict search. If ``original_observations/`` is absent, supply ``--traj-base``
to fall back to loading from the base trajectory files (ground-truth, pre-noise).

Usage:
    python benchmark/diagnosis/retrace_search.py \\
        --fold-dir  benchmark/data/blocksworld/.../testing/fold0_numtrajs3_gtrate0 \\
        --domain    benchmark/data/blocksworld/domain.pddl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Project root on sys.path so src/ and benchmark/ imports work.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Observation

from src.utils.masking import load_masked_observation

# Re-use fold metadata helpers from the diagnosis script.
from benchmark.diagnosis.trace_spurious_effects import (
    load_fold_info,
    resolve_trajectory_paths,
    load_observations_in_order,
)


def load_original_observations(
    fold_dir: Path,
    fold_info: Dict,
    domain: Domain,
) -> List[Tuple[str, Observation]]:
    """Load observations from the fold's original_observations/ directory.

    These are the noised/masked observations that the experiment pipeline actually
    fed to the conflict search — as opposed to the base trajectory files which are
    pre-noise ground truth.

    Args:
        fold_dir:   Path to the fold directory.
        fold_info:  Parsed fold_info.json dict (provides processing order).
        domain:     Parsed partial domain.

    Returns:
        Ordered list of (problem_name, observation).
    """
    obs_dir = fold_dir / "original_observations"
    observations: List[Tuple[str, Observation]] = []
    for entry in fold_info["trajectories"]:
        problem = entry["problem"]
        traj_path = obs_dir / f"original_observation_{problem}.trajectory"
        masking_path = obs_dir / f"original_observation_{problem}.masking_info"
        if not traj_path.exists():
            raise FileNotFoundError(f"Original observation not found: {traj_path}")
        if not masking_path.exists():
            raise FileNotFoundError(f"Original masking info not found: {masking_path}")
        obs = load_masked_observation(traj_path, masking_path, domain)
        n_components = len(obs.components)
        print(f"  Loaded obs[{len(observations)}] problem={problem} | components={n_components}")
        observations.append((problem, obs))
    return observations


def retrace_conflict_search(
    ordered_observations: List[Tuple[str, Observation]],
    fold_dir: Path,
    domain: Domain,
) -> Path:
    """Re-run ConflictDrivenPatchSearch with per-node tracing and save the trace JSON.

    Reads search parameters from learning_metrics.json to reproduce the original run.
    Falls back to defaults if the metrics file is absent.

    Returns:
        Path to the saved search_trace.json.
    """
    from src.plan_denoising.conflict_search import ConflictDrivenPatchSearch
    from src.plan_denoising.frontier import (
        SearchMode, NodeChoosingStrategy, ConflictGroupStrategy, FluentBranchMode,
    )
    from src.pi_sam.noisy_pisam.typings import NodeExpansionEvent

    # Load original search parameters from saved metrics if available.
    metrics_path = fold_dir / "learning_metrics.json"
    params: Dict = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            params = json.load(f)

    _MODE_MAP = {
        "anytime_dfs": SearchMode.ANYTIME_DFS,
        "ucs": SearchMode.UCS,
    }
    _NODE_MAP = {
        "model_patch_first": NodeChoosingStrategy.MODEL_PATCH_FIRST,
        "fluent_patch_first": NodeChoosingStrategy.FLUENT_PATCH_FIRST,
        "fluent_patch_first_then_model": NodeChoosingStrategy.FLUENT_PATCH_FIRST_THEN_MODEL,
        "randomized": NodeChoosingStrategy.RANDOMIZED,
    }
    _GROUP_MAP = {
        "first": ConflictGroupStrategy.FIRST,
        "largest": ConflictGroupStrategy.LARGEST,
        "largest_model_patchable": ConflictGroupStrategy.LARGEST_MODEL_PATCHABLE,
        "most_observations": ConflictGroupStrategy.MOST_OBSERVATIONS,
        "smallest": ConflictGroupStrategy.SMALLEST,
    }
    _FLUENT_MAP = {
        "group": FluentBranchMode.GROUP,
        "single": FluentBranchMode.SINGLE,
    }

    search_mode = _MODE_MAP.get(
        str(params.get("search_mode", "")).lower(), SearchMode.ANYTIME_DFS
    )
    node_choosing = _NODE_MAP.get(
        str(params.get("node_choosing_strategy", "")).lower(),
        NodeChoosingStrategy.MODEL_PATCH_FIRST,
    )
    conflict_group = _GROUP_MAP.get(
        str(params.get("conflict_group_strategy", "")).lower(),
        ConflictGroupStrategy.FIRST,
    )
    fluent_branch = _FLUENT_MAP.get(
        str(params.get("fluent_branch_mode", "")).lower(),
        FluentBranchMode.GROUP,
    )
    fluent_patch_cost = float(params.get("fluent_patch_cost") or 1.0)
    fluent_patch_weight = float(params.get("fluent_patch_weight") or 1.0)
    model_patch_cost = float(params.get("model_patch_cost") or 1.0)
    model_constraint_weight = float(params.get("model_constraint_weight") or 0.0)
    timeout_seconds = int(params.get("actual_timeout_seconds") or 60)
    max_nodes = params.get("max_search_nodes")

    print(f"  Search params: mode={search_mode.value}, node_order={node_choosing.value}, "
          f"group={conflict_group.value}, fluent_branch={fluent_branch.value}")
    print(f"  Costs: fluent={fluent_patch_cost}(w={fluent_patch_weight}), "
          f"model={model_patch_cost}(w={model_constraint_weight})")
    print(f"  Limits: timeout={timeout_seconds}s, max_nodes={max_nodes}")

    # --- Callback: collect all node expansion events ---
    trace_log: List[NodeExpansionEvent] = []

    def _on_node(event: NodeExpansionEvent) -> None:
        trace_log.append(event)

    searcher = ConflictDrivenPatchSearch(
        partial_domain_template=domain,
        search_mode=search_mode,
        fluent_patch_cost=fluent_patch_cost,
        fluent_patch_weight=fluent_patch_weight,
        model_patch_cost=model_patch_cost,
        model_constraint_weight=model_constraint_weight,
        node_choosing_strategy=node_choosing,
        conflict_group_strategy=conflict_group,
        fluent_branch_mode=fluent_branch,
    )

    obs_list = [obs for _, obs in ordered_observations]
    search_result = searcher.run(
        observations=obs_list,
        max_nodes=max_nodes,
        timeout_seconds=timeout_seconds,
        on_node_expanded=_on_node,
    )
    learned_model = search_result.learned_domain
    final_constraints = search_result.model_constraints
    final_fluent_patches = search_result.fluent_patches
    report = search_result.report

    # --- Serialize trace ---
    from benchmark.diagnosis.trace_serialization import write_trace_json

    search_params_used = {
        "search_mode": search_mode.value,
        "node_choosing_strategy": node_choosing.value,
        "conflict_group_strategy": conflict_group.value,
        "fluent_branch_mode": fluent_branch.value,
        "fluent_patch_cost": fluent_patch_cost,
        "model_patch_cost": model_patch_cost,
        "timeout_seconds": timeout_seconds,
    }

    trace_path = fold_dir / "search_trace.json"
    write_trace_json(
        trace_log, trace_path, search_params_used,
        fold_dir=fold_dir, ordered_observations=ordered_observations,
    )

    cfm_nodes = [e for e in trace_log if e.is_conflict_free]
    print(f"\n  Trace saved to: {trace_path}")
    print(f"  Nodes expanded: {len(trace_log)}")
    print(f"  Conflict-free models found: {len(cfm_nodes)}")
    if cfm_nodes:
        print(f"  Best CFM cost: {min(e.cost for e in cfm_nodes):.2f}")

    return trace_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run a fold's conflict search with tracing and save search_trace.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--fold-dir", required=True, type=Path,
        help="Path to the fold directory containing fold_info.json.",
    )
    parser.add_argument(
        "--domain", required=True, type=Path,
        help="Path to the domain PDDL file (partial_parsing=True will be used).",
    )
    parser.add_argument(
        "--traj-base", required=False, type=Path, default=None,
        help=(
            "Fallback: base directory containing one subdirectory per problem. "
            "Only needed if the fold has no original_observations/ directory. "
            "Warning: these are pre-noise ground-truth trajectories and may not "
            "reproduce the original search."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fold_dir: Path = args.fold_dir.resolve()
    domain_path: Path = args.domain.resolve()
    traj_base: Path = args.traj_base.resolve() if args.traj_base else None

    print(f"\nRetrace conflict search:")
    print(f"  Fold dir:  {fold_dir}")
    print(f"  Domain:    {domain_path}")

    # Check if trace already exists
    existing_trace = fold_dir / "search_trace.json"
    if existing_trace.exists():
        print(f"\n  search_trace.json already exists at: {existing_trace}")
        answer = input("  Re-run retrace and overwrite? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print(f"\n  Existing trace: {existing_trace}")
            print(f"  Visualize with:")
            print(f"    python benchmark/diagnosis/visualize_trace.py {existing_trace}")
            return

    # Stage 0 — Load fold metadata
    print("\n[Stage 0] Loading fold_info.json …")
    fold_info = load_fold_info(fold_dir)
    problems = [e["problem"] for e in fold_info["trajectories"]]
    print(f"  Processing order ({len(problems)} trajectories): {problems}")

    # Stage 1 — Load observations (prefer original_observations/ over traj_base)
    obs_dir = fold_dir / "original_observations"
    if obs_dir.is_dir():
        print(f"\n[Stage 1] Loading from original_observations/ (noised/masked) …")
        domain = DomainParser(domain_path, partial_parsing=True).parse_domain()
        ordered_observations = load_original_observations(fold_dir, fold_info, domain)
    elif traj_base is not None:
        print(f"\n[Stage 1] No original_observations/ found — falling back to --traj-base …")
        print(f"  WARNING: traj-base contains pre-noise trajectories; results may differ from original run.")
        domain = DomainParser(domain_path, partial_parsing=True).parse_domain()
        ordered_paths = resolve_trajectory_paths(fold_info, traj_base)
        ordered_observations = load_observations_in_order(ordered_paths, domain)
    else:
        print(f"\nERROR: No original_observations/ directory in fold and no --traj-base provided.")
        sys.exit(1)

    # Run search with tracing
    print("\n[Retrace] Running conflict search with tracing …")
    domain_for_search = DomainParser(domain_path, partial_parsing=True).parse_domain()
    trace_path = retrace_conflict_search(ordered_observations, fold_dir, domain_for_search)

    print(f"\nDone. Visualize with:")
    print(f"  python benchmark/diagnosis/visualize_trace.py {trace_path}")


if __name__ == "__main__":
    main()
