"""Build anytime profiles for a cell's folds.

    python -m benchmark.evaluation.anytime.run_anytime \
        benchmark/running_results/blocksworld/sim_run__mask=0.1__noise=0.2

Reads whatever checkpoints each fold happens to hold, scores them against that
fold's frozen observations, and writes ``anytime_scores.json`` plus
``anytime_curve.png`` into the fold. Nothing here re-runs a learner, so it is
safe to point at a finished run and safe to re-run after changing how the score
is computed.

Deliberately per-fold. Averaging profiles across folds requires deciding what to
do when folds have different horizons and different numbers of traces, and every
answer to that smuggles in an assumption; the per-fold figures answer the
question the plan actually asks without needing one.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from benchmark.evaluation.anytime.checkpoints import ARM_SUBDIRS, read_fold_checkpoints
from benchmark.evaluation.anytime.curves import plot_fold
from benchmark.evaluation.anytime.score import score_fold, write_scores

CURVE_NAME = "anytime_curve.png"


def find_folds(target: Path) -> List[Path]:
    """The fold directories under ``target``, which may itself be one."""
    target = Path(target)
    if (target / "original_observations").is_dir():
        return [target]
    testing = target / "testing"
    root = testing if testing.is_dir() else target
    return sorted(p for p in root.glob("fold*") if p.is_dir())


def process_fold(
    fold: Path, arms: Optional[List[str]], plot: bool = True
) -> Optional[Path]:
    """Score one fold and draw its figure; returns the figure path, if drawn."""
    checkpoints = read_fold_checkpoints(fold, arms)
    if not checkpoints:
        print(f"  {fold.name}: no checkpoints from any arm, skipping")
        return None

    scored = score_fold(fold, checkpoints)
    write_scores(fold, scored)
    counts = ", ".join(f"{arm}={len(v)}" for arm, v in sorted(scored.items()))
    print(f"  {fold.name}: {counts}")

    if not plot:
        return None
    return plot_fold(scored, fold / CURVE_NAME, title=f"{fold.parent.parent.name} / {fold.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="A cell directory, or a single fold")
    parser.add_argument(
        "--arms", nargs="+", choices=sorted(ARM_SUBDIRS),
        help="Restrict to these arms. Default: every arm that left artifacts.",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Write anytime_scores.json only. Useful when iterating on the score.",
    )
    args = parser.parse_args()

    folds = find_folds(args.target)
    if not folds:
        raise SystemExit(f"no folds found under {args.target}")

    print(f"Anytime profiles for {len(folds)} fold(s) under {args.target}")
    for fold in folds:
        process_fold(fold, args.arms, plot=not args.no_plot)


if __name__ == "__main__":
    main()
