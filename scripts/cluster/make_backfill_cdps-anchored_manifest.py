"""Generate a SLURM array-job manifest for backfilling CDPS_ANCHORED rows.

Unlike ``make_manifest.py`` (which expands run_config.yaml into *new* runs),
this generator scans the **already-finished** experiment directories under
``benchmark/running_results/`` and emits one array task per
(experiment_dir x num_trajs) group — i.e. the 5 folds of one cell group, which
``backfill_cdps.py`` then runs in parallel in-process via ``--workers``.

Why this granularity: per-cell tasks would need 1350 array indices, but this
cluster's ``MaxArraySize`` is 1001. Grouping the 5 folds of a num_trajs value
into one task gives 270 indices and mirrors ``run_cell.sbatch``'s proven
resource shape (5 CPU-bound fold processes, 12 logical CPUs, 24G).

Every other backfill parameter (data_dir, learn/planning timeouts, CDPS search
shape, frame-axiom mode) is read by ``backfill_cdps.py`` from each experiment's
own ``evaluation_results/run_params.json``, so the anchored run reproduces the
original CDPS configuration exactly and nothing needs restating here.

Naming: this generator is specific to the ``cdps_anchored`` backfill. Future
backfills (other algorithms) get their own ``make_backfill_<algo>_manifest.py``
plus a matching sbatch template, rather than flags on this one.

Usage (from the project root):
    python scripts/cluster/make_backfill_cdps-anchored_manifest.py \
        --run-name simulation-cluster-run-cpu256-10min
    python scripts/cluster/make_backfill_cdps-anchored_manifest.py \
        --run-name my-run --domains blocksworld hanoi
"""

import argparse
import csv
import re
from pathlib import Path
from typing import List, Optional

project_root = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS_ROOT = project_root / "benchmark" / "running_results"
DEFAULT_OUT_DIR = project_root / "scripts" / "cluster"

CELL_PATTERN = re.compile(r"^fold(\d+)_numtrajs(\d+)_gtrate(\d+)$")


def _num_trajs_in(testing_dir: Path) -> List[int]:
    """Return the sorted distinct num_trajs values present as cells under testing/."""
    values = set()
    for cell in testing_dir.iterdir():
        if not cell.is_dir():
            continue
        match = CELL_PATTERN.match(cell.name)
        if match is not None:
            values.add(int(match.group(2)))
    return sorted(values)


def _experiment_dirs(results_root: Path, run_name: str,
                     domains: Optional[List[str]]) -> List[Path]:
    """Collect experiment dirs named ``{run_name}__*`` under the selected domains."""
    if not results_root.is_dir():
        raise SystemExit(f"Results root does not exist: {results_root}")

    found: List[Path] = []
    for domain_dir in sorted(results_root.iterdir()):
        if not domain_dir.is_dir():
            continue
        if domains is not None and domain_dir.name not in domains:
            continue
        for exp_dir in sorted(domain_dir.glob(f"{run_name}__*")):
            if (exp_dir / "testing").is_dir():
                found.append(exp_dir)
            else:
                print(f"  Warning: no testing/ under {exp_dir} — skipped")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", required=True,
                        help="Run name prefix of the finished experiments "
                             "(matches dirs named '{run_name}__mask=..__noise=..')")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
                        help=f"Results root (default: {DEFAULT_RESULTS_ROOT})")
    parser.add_argument("--out", type=Path,
                        default=DEFAULT_OUT_DIR / "backfill_cdps-anchored_manifest.csv",
                        help="Output manifest path (default: "
                             "scripts/cluster/backfill_cdps-anchored_manifest.csv)")
    parser.add_argument("--domains", nargs="*", default=None, metavar="DOMAIN",
                        help="Restrict to these domain dirs (default: all found)")
    args = parser.parse_args()

    exp_dirs = _experiment_dirs(args.results_root, args.run_name, args.domains)
    if not exp_dirs:
        raise SystemExit(
            f"No experiment dirs matching '{args.run_name}__*' under {args.results_root}"
        )

    rows = []
    for exp_dir in exp_dirs:
        num_trajs_values = _num_trajs_in(exp_dir / "testing")
        if not num_trajs_values:
            print(f"  Warning: no cells under {exp_dir}/testing — skipped")
            continue
        domain = exp_dir.parent.name
        for num_trajs in num_trajs_values:
            # Paths are stored relative to the project root: the sbatch template
            # cd's to $SLURM_SUBMIT_DIR, so the manifest stays machine-portable.
            rows.append([domain, exp_dir.relative_to(project_root).as_posix(), num_trajs])

    if not rows:
        raise SystemExit("No cells found — nothing to backfill.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # lineterminator="\n": csv's default excel dialect emits CRLF, and the sbatch
    # template parses rows with `sed` + `IFS=, read`, which leaves the trailing \r
    # glued to the last field — it then leaks into the --cells filter and matches
    # nothing (silently backfilling zero cells).
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["domain", "experiment_dir", "num_trajs"])
        writer.writerows(rows)

    n = len(rows)
    print(f"Wrote {args.out} ({n} tasks over {len(exp_dirs)} experiment dirs)")
    print("\nSubmit from the project root on the cluster with:")
    print("  mkdir -p logs")
    print(f"  sbatch --array=0-{n - 1}%60 "
          f"scripts/cluster/run_backfill_cdps-anchored.sbatch "
          f"{args.out.relative_to(project_root).as_posix()}")


if __name__ == "__main__":
    main()
