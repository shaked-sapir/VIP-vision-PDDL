"""Relabel ``pisam_milp_loop`` final observations written before the subset-naming fix.

The loop returns the winning round's subset, but the saver used to name the files
by position in that list, so a fold ``[p7, p5, p3]`` with subset ``[0, 2]`` wrote
``p3``'s observation under ``final_observation_p5``. The correct label for
position ``j`` is the fold trajectory at ``subset[j]``; the subset comes from
``milp_loop_rounds.json`` and the trajectory order from the fold's ``fold_info.json``.

    python -m benchmark.rename_loop_final_observations [--dry-run] [results_root]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PREFIX = "final_observation_"
SUFFIXES = (".trajectory", ".masking_info")


def _fold_problems(fold_dir: Path) -> List[str]:
    """Problem names in the fold's trajectory order."""
    info = json.loads((fold_dir / "fold_info.json").read_text())
    return [entry["problem"] for entry in info["trajectories"]]


def _winning_subset(loop_dir: Path) -> Optional[List[int]]:
    """The learner input of the winning round, or ``None`` when it cannot be trusted."""
    record = json.loads((loop_dir / "milp_loop_rounds.json").read_text())
    best = record.get("best_round")
    if best is None:
        return None
    rounds = {r.get("round", r.get("round_index")): r for r in record["rounds"]}
    winner = rounds[best]
    subset = list(winner["subset"])
    if winner.get("learner_input_size") != len(subset):
        return None  # accumulated input: the files are not the subset
    return subset


def plan_renames(loop_dir: Path) -> Tuple[str, List[Tuple[Path, Path]]]:
    """Classify one loop directory and list the ``(old, new)`` renames it needs."""
    final_dir = loop_dir / "final_observations"
    present = sorted(p.stem[len(PREFIX):] for p in final_dir.glob(f"{PREFIX}*.trajectory"))
    if not present:
        return "empty", []
    if not (loop_dir / "milp_loop_rounds.json").exists():
        return "no_rounds_record", []
    subset = _winning_subset(loop_dir)
    if subset is None:
        return "unrecoverable_subset", []
    problems = _fold_problems(loop_dir.parent)
    expected = sorted(problems[i] for i in subset)
    if present == expected:
        return "already_correct", []
    if len(present) != len(subset) or present != sorted(problems[: len(subset)]):
        return "unexpected_files", []
    renames = []
    for position, index in enumerate(subset):
        old, new = problems[position], problems[index]
        if old == new:
            continue
        for suffix in SUFFIXES:
            source = final_dir / f"{PREFIX}{old}{suffix}"
            if source.exists():
                renames.append((source, final_dir / f"{PREFIX}{new}{suffix}"))
    return "rename", renames


def apply_renames(renames: List[Tuple[Path, Path]]) -> None:
    """Rename through temporaries, so a target that is also a source is not clobbered."""
    staged = []
    for source, target in renames:
        temp = source.with_name(source.name + ".relabel-tmp")
        source.rename(temp)
        staged.append((temp, target))
    for temp, target in staged:
        temp.rename(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results_root", nargs="?", default="benchmark/running_results", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="report only, rename nothing")
    args = parser.parse_args()

    outcomes: Dict[str, int] = collections.Counter()
    flagged: Dict[str, List[Path]] = collections.defaultdict(list)
    for loop_dir in sorted(args.results_root.glob("*/*/testing/*/pisam_milp_loop*")):
        outcome, renames = plan_renames(loop_dir)
        outcomes[outcome] += 1
        if outcome in ("no_rounds_record", "unrecoverable_subset", "unexpected_files"):
            flagged[outcome].append(loop_dir)
        if outcome == "rename" and not args.dry_run:
            apply_renames(renames)
    for outcome, count in sorted(outcomes.items()):
        print(f"{outcome}: {count}")
    for outcome, dirs in flagged.items():
        print(f"\n{outcome} ({len(dirs)}):")
        for d in dirs[:20]:
            print(f"  {d}")


if __name__ == "__main__":
    main()
