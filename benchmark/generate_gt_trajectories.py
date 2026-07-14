"""GT-trajectory backfill + validation CLI.

The normal data-generation run now exports ground-truth trajectories in-line
(see ``data_generator._run_generation_inference`` and the predefined/external
loop, which call ``gt_builder.export_gt_from_problem_dir``). This script is the
standalone counterpart for existing experiment dirs:

- ``--validate``   : check every ``gt_trajectories/<problem>/<problem>.trajectory``
                     against structural + per-domain physical invariants.
- ``--reexport``   : (re)build ``gt_trajectories/`` from each problem's
                     ``training/trajectories/<problem>/<problem>_trajectory.json``.
                     Only valid when that JSON is genuine GT (predefined/external
                     datasets, where it is never rebuilt from the classifier).

Important: for **generate-mode** datasets whose GT JSON was already overwritten by
the classifier (the historical bug), re-export cannot recover ground truth — the
true intermediate states were destroyed. Recover those by re-running
``data_generator`` generation with the fixed pipeline (deterministic, seeded),
which now re-steps the gym env and writes correct GT alongside the noisy data.

Examples:
    python -m benchmark.generate_gt_trajectories --experiment-dir <path> --validate --domain-key hanoi
    python -m benchmark.generate_gt_trajectories --experiment-dir <path> --reexport
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmark.experiment_running_helpers.gt_builder import (
    export_gt_from_problem_dir,
    validate_gt_dir,
)


def _resolve_trajectories_dir(experiment_dir: Path) -> Path:
    """Locate the per-problem trajectories dir inside an experiment dir."""
    trajectories_dir = experiment_dir / "training" / "trajectories"
    if not trajectories_dir.is_dir():
        raise FileNotFoundError(
            f"No training/trajectories dir found under {experiment_dir}. "
            f"Expected: {trajectories_dir}"
        )
    return trajectories_dir


def reexport_experiment_gt(experiment_dir: Path) -> Path:
    """Rebuild gt_trajectories/ from each problem's eval-schema GT JSON."""
    trajectories_dir = _resolve_trajectories_dir(experiment_dir)
    gt_root = experiment_dir / "gt_trajectories"

    problem_dirs = sorted(
        d for d in trajectories_dir.iterdir()
        if d.is_dir() and not d.name.startswith((".", "_"))
    )
    for problem_dir in problem_dirs:
        try:
            export_gt_from_problem_dir(
                problem_dir, gt_root, problem_dir.name,
                handler=None, needs_schema_translation=False,
            )
            print(f"  ✓ exported GT for {problem_dir.name}")
        except Exception as e:
            print(f"  ✗ {problem_dir.name}: {e}")
    return gt_root


def validate_experiment_gt(experiment_dir: Path, domain_key: str = None) -> int:
    """Validate gt_trajectories/ and print a report. Returns violation count."""
    gt_root = experiment_dir / "gt_trajectories"
    if not gt_root.is_dir():
        raise FileNotFoundError(f"No gt_trajectories/ under {experiment_dir}")

    report = validate_gt_dir(gt_root, domain_key)
    if not report:
        print(f"✓ All GT trajectories under {gt_root} are clean "
              f"(domain_key={domain_key}).")
        return 0

    total = 0
    print(f"✗ GT invariant violations under {gt_root} (domain_key={domain_key}):")
    for problem, viols in report.items():
        total += len(viols)
        print(f"  {problem}: {len(viols)} violation(s)")
        for v in viols[:8]:
            print(f"      {v}")
    print(f"\nTotal: {total} violation(s) across {len(report)} problem(s).")
    return total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate or re-export GT trajectories for an experiment dir."
    )
    parser.add_argument("--experiment-dir", type=str, required=True,
                        help="Experiment dir containing training/trajectories/ "
                             "and/or gt_trajectories/.")
    parser.add_argument("--validate", action="store_true",
                        help="Validate gt_trajectories/ against invariants.")
    parser.add_argument("--reexport", action="store_true",
                        help="Rebuild gt_trajectories/ from eval-schema GT JSONs.")
    parser.add_argument("--domain-key", type=str, default=None,
                        help="Domain key for physical invariants (e.g. hanoi, npuzzle).")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).expanduser().resolve()

    if not args.validate and not args.reexport:
        parser.error("choose at least one of --validate / --reexport")

    if args.reexport:
        gt_root = reexport_experiment_gt(experiment_dir)
        print(f"\nGT trajectories written to: {gt_root}")

    if args.validate:
        violations = validate_experiment_gt(experiment_dir, args.domain_key)
        sys.exit(1 if violations else 0)
