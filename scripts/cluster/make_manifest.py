"""Generate SLURM array-job manifests from benchmark/run_config.yaml.

Reads the same run config used by ``benchmark_runner`` (single source of
truth) and emits, into scripts/cluster/:

  - ``manifest.csv``        — one row per (domain x p_mask x p_noise x num_trajs)
                              array task ("cell-level"; each task runs all folds
                              of that instance in parallel in-process), OR
  - ``manifest_folds.csv``  — with ``--per-fold``, one row per
                              (... x fold) array task ("fold-level"; requires
                              the ``--folds`` flag on experiment_runner).
  - ``run_flags.sh``        — shared experiment_runner CLI flags derived from
                              the config's ``shared``/``simulation`` blocks,
                              sourced by the sbatch templates so every task
                              runs with exactly the run config's parameters.

Experiment names follow benchmark_runner's convention
(``{run_name}__mask={p}__noise={q}``), so collect_results / experiment_report
and the rest of the evaluation tooling work on the outputs unchanged.

Usage (from the project root):
    python scripts/cluster/make_manifest.py                # cell-level manifest
    python scripts/cluster/make_manifest.py --per-fold     # fold-level manifest
    python scripts/cluster/make_manifest.py --domains blocksworld hanoi
"""

import argparse
import csv
import shlex
from itertools import product
from pathlib import Path

import yaml

project_root = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = project_root / "benchmark" / "run_config.yaml"
DEFAULT_OUT_DIR = project_root / "scripts" / "cluster"

# Maps run_config ``shared`` keys -> experiment_runner CLI flags.
# (num_trajectories is intentionally absent: it comes from the manifest row.)
_SHARED_FLAG_MAP = {
    "n_folds": "--n-folds",
    "gt_rates": "--gt-rates",
    "learning_timeout_seconds": "--learning-timeout-seconds",
    "planning_timeout_seconds": "--planning-timeout-seconds",
    "search_mode": "--search-mode",
    "node_choosing_strategy": "--node-choosing-strategy",
    "model_constraint_weight": "--model-constraint-weight",
    "conflict_group_strategy": "--conflict-group-strategy",
    "fluent_branch_mode": "--fluent-branch-mode",
}


def _experiment_name(run_name: str, p_mask, p_noise) -> str:
    """Mirror benchmark_runner._cell_experiment_name for simulated cells."""
    return f"{run_name}__mask={p_mask}__noise={p_noise}"


def _build_common_flags(config: dict) -> str:
    """Translate shared + simulation config blocks into CLI flags."""
    shared = config.get("shared", {})
    sim = config.get("simulation", {})

    parts = []
    for key, flag in _SHARED_FLAG_MAP.items():
        if key not in shared:
            continue  # fall back to experiment_runner's own default
        value = shared[key]
        if isinstance(value, list):
            value = ",".join(str(v) for v in value)
        parts.extend([flag, shlex.quote(str(value))])

    algorithms = shared.get("algorithms")
    if algorithms:
        parts.append("--algorithms")
        parts.extend(shlex.quote(str(a)) for a in algorithms)

    if shared.get("events_tracing"):
        parts.append("--events-tracing")

    # Always resume on the cluster: fold_result.json markers make re-submission
    # of failed array indices idempotent.
    parts.append("--resume")

    parts.extend(["--simulated-masking-strategy",
                  shlex.quote(str(sim.get("masking_strategy", "percentage")))])
    parts.extend(["--simulated-noising-strategy",
                  shlex.quote(str(sim.get("noising_strategy", "percentage")))])
    parts.extend(["--simulated-seed", shlex.quote(str(sim.get("seed", 42)))])

    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"Run config YAML (default: {DEFAULT_CONFIG})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--per-fold", action="store_true", default=False,
                        help="One row per fold (fold-level jobs) instead of one "
                             "row per cell x num_trajs")
    parser.add_argument("--domains", nargs="*", default=None, metavar="DOMAIN_KEY",
                        help="Restrict to these domain keys (default: all in config)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if config.get("source") != "simulated":
        raise SystemExit(
            f"run config source is {config.get('source')!r}; this generator "
            f"currently supports source: simulated only."
        )

    run_name = config["run_name"]
    shared = config.get("shared", {})
    sim = config.get("simulation", {})
    grid = sim.get("grid", {})

    masking_ps = grid.get("masking_ps") or []
    noising_ps = grid.get("noising_ps") or []
    if not masking_ps or not noising_ps:
        raise SystemExit("simulation.grid must define non-empty masking_ps and noising_ps.")

    num_trajectories = shared.get("num_trajectories", [3, 4, 5, 6, 7, 8])
    n_folds = int(shared.get("n_folds", 5))

    domains = config.get("domains", [])
    if args.domains is not None:
        known = {d["domain_key"] for d in domains}
        unknown = set(args.domains) - known
        if unknown:
            raise SystemExit(f"Unknown domain key(s) {sorted(unknown)}; config has {sorted(known)}")
        domains = [d for d in domains if d["domain_key"] in args.domains]
    if not domains:
        raise SystemExit("No domains selected.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Manifest ─────────────────────────────────────────────────────────
    header = ["domain_key", "data_dir", "p_mask", "p_noise", "num_trajs", "experiment_name"]
    if args.per_fold:
        header.append("fold")
        manifest_path = args.out_dir / "manifest_folds.csv"
    else:
        manifest_path = args.out_dir / "manifest.csv"

    rows = []
    for d, mp, np_, nt in product(domains, masking_ps, noising_ps, num_trajectories):
        base = [d["domain_key"], d["data_dir"], mp, np_, nt,
                _experiment_name(run_name, mp, np_)]
        if args.per_fold:
            rows.extend(base + [fold] for fold in range(n_folds))
        else:
            rows.append(base)

    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # ── Shared CLI flags ─────────────────────────────────────────────────
    flags_path = args.out_dir / "run_flags.sh"
    with open(flags_path, "w") as f:
        f.write(
            "# Generated by scripts/cluster/make_manifest.py from "
            f"{args.config.relative_to(project_root)} — do not edit by hand;\n"
            "# regenerate after changing the run config. Sourced by the sbatch templates.\n"
            f'COMMON_FLAGS="{_build_common_flags(config)}"\n'
        )

    # ── Summary ──────────────────────────────────────────────────────────
    n = len(rows)
    template = "run_fold.sbatch" if args.per_fold else "run_cell.sbatch"
    print(f"Wrote {manifest_path.relative_to(project_root)} ({n} tasks) "
          f"and {flags_path.relative_to(project_root)}")
    print(f"\nSubmit from the project root on the cluster with:")
    print(f"  mkdir -p logs")
    print(f"  sbatch --array=0-{n - 1}%15 scripts/cluster/{template} "
          f"{manifest_path.relative_to(project_root)}")
    print(f"\n(or use scripts/cluster/submit.sh, which computes the range for you)")


if __name__ == "__main__":
    main()
