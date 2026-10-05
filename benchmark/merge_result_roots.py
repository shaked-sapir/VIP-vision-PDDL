"""Merge rows from mirrored result roots into the main results tree.

A mirrored root (``benchmark/running_results_<suffix>/``) holds, for every cell
it ran, ``<domain>/<experiment>/testing/<cell>/fold_result.json`` with the rows
of its own arms only. This script copies the rows a config selects into the
same cell of the target tree, replacing a row with the same label, and brings
the label-named artifacts along:

- ``learned_domain_<label>.pddl`` and ``baseline_models/<label>/``;
- ``rosame_training/<arm>.json``, stored under the row's label so it never
  overwrites the target's own file of that arm;
- the JSON logs at the top of a ``pisam_milp_*`` arm's work directory.

Each target cell gets a ``merge_provenance.json`` naming the root every merged
label came from. The main tree's rows are never removed.

Usage::

    python -m benchmark.merge_result_roots --dry-run
    python -m benchmark.merge_result_roots
    python -m benchmark.merge_result_roots --config benchmark/paper_result_sources.yaml --roots benchmark/running_results_pisam_raw
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmark.algorithms import (  # noqa: E402
    PISAM_MILP_LOOP,
    PISAM_MILP_LOOP_ALGORITHM_NAME,
    PISAM_MILP_SINGLE_ROUND,
    PISAM_MILP_SINGLE_ROUND_ALGORITHM_NAME,
)
from benchmark.backfill_common import is_cell_dir, merge_row  # noqa: E402
from benchmark.experiment_running_helpers.resume import FOLD_RESULT_FILENAME  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "paper_result_sources.yaml"
PROVENANCE_FILENAME = "merge_provenance.json"
ROSAME_TRAINING_DIRNAME = "rosame_training"
BASELINE_MODELS_DIRNAME = "baseline_models"

_MILP_WORK_KEYS = {
    PISAM_MILP_LOOP_ALGORITHM_NAME: PISAM_MILP_LOOP,
    PISAM_MILP_SINGLE_ROUND_ALGORITHM_NAME: PISAM_MILP_SINGLE_ROUND,
}


@dataclass(frozen=True)
class Source:
    """One mirrored root and the rows to take from it."""

    root: Path
    labels: Tuple[str, ...]
    experiments: str = "*"

    def selects(self, experiment: str, label: str) -> bool:
        return fnmatch.fnmatch(experiment, self.experiments) and label in self.labels


@dataclass
class MergePlan:
    """What one run would do, or did."""

    merged: Counter = field(default_factory=Counter)
    replaced: Counter = field(default_factory=Counter)
    skipped_cells: List[str] = field(default_factory=list)
    unselected: Counter = field(default_factory=Counter)


def load_sources(config_path: Path) -> Tuple[Path, List[Source]]:
    """``(target root, sources in merge order)`` from a YAML config."""
    raw = yaml.safe_load(Path(config_path).read_text()) or {}
    sources = [
        Source(Path(s["root"]), tuple(s["labels"]), s.get("experiments", "*"))
        for s in raw.get("sources", [])
    ]
    return Path(raw["target"]), sources


def iter_source_cells(root: Path) -> Iterator[Tuple[str, str, Path]]:
    """``(domain, experiment, cell dir)`` for every cell with a result file under ``root``."""
    for result in sorted(root.glob(f"*/*/testing/*/{FOLD_RESULT_FILENAME}")):
        cell = result.parent
        if is_cell_dir(cell):
            yield cell.parent.parent.parent.name, cell.parent.parent.name, cell


def same_fold(source_cell: Path, target_cell: Path) -> bool:
    """True when both cells describe the same trajectories and test problems."""
    try:
        source = json.loads((source_cell / "fold_info.json").read_text())
        target = json.loads((target_cell / "fold_info.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    keys = ("trajectories", "test_problems")
    return all(source.get(key) == target.get(key) for key in keys)


def _arm_name(label: str) -> str:
    return label.partition("__")[0]


def milp_work_dirname(label: str) -> Optional[str]:
    """The fold subdirectory of a ``pisam_milp_*`` row, or None for other arms."""
    key = _MILP_WORK_KEYS.get(_arm_name(label))
    if key is None:
        return None
    _, _, suffix = label.partition("__")
    return f"{key}__{suffix}" if suffix else key


def _copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def copy_artifacts(source_cell: Path, target_cell: Path, label: str) -> List[str]:
    """Copy the files that belong to ``label`` from one cell to the other; returns what was copied."""
    copied: List[str] = []
    model = source_cell / f"learned_domain_{label}.pddl"
    if model.is_file():
        shutil.copy2(model, target_cell / model.name)
        copied.append(model.name)
    baseline_dir = source_cell / BASELINE_MODELS_DIRNAME / label
    if baseline_dir.is_dir():
        _copy_tree(baseline_dir, target_cell / BASELINE_MODELS_DIRNAME / label)
        copied.append(f"{BASELINE_MODELS_DIRNAME}/{label}/")
    curve = source_cell / ROSAME_TRAINING_DIRNAME / f"{_arm_name(label)}.json"
    if curve.is_file():
        (target_cell / ROSAME_TRAINING_DIRNAME).mkdir(exist_ok=True)
        shutil.copy2(curve, target_cell / ROSAME_TRAINING_DIRNAME / f"{label}.json")
        copied.append(f"{ROSAME_TRAINING_DIRNAME}/{label}.json")
    work_dirname = milp_work_dirname(label)
    if work_dirname and (source_cell / work_dirname).is_dir():
        (target_cell / work_dirname).mkdir(exist_ok=True)
        for log in sorted((source_cell / work_dirname).glob("*.json")):
            shutil.copy2(log, target_cell / work_dirname / log.name)
            copied.append(f"{work_dirname}/{log.name}")
    return copied


def record_provenance(target_cell: Path, label: str, root: Path, copied: Sequence[str]) -> None:
    """Note in the target cell where ``label``'s row came from."""
    path = target_cell / PROVENANCE_FILENAME
    provenance = json.loads(path.read_text()) if path.is_file() else {}
    provenance[label] = {
        "source_root": str(root),
        "merged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifacts": list(copied),
    }
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True))


def merge_cell(
    source: Source, source_cell: Path, target_cell: Path, experiment: str, plan: MergePlan, apply: bool,
) -> None:
    """Merge the selected rows of one source cell into its target cell."""
    rows = json.loads((source_cell / FOLD_RESULT_FILENAME).read_text())
    target_result = target_cell / FOLD_RESULT_FILENAME
    existing = set()
    if target_result.is_file():
        existing = {r.get("algorithm") for r in json.loads(target_result.read_text())}
    for row in rows:
        label = row.get("algorithm")
        if not source.selects(experiment, label):
            plan.unselected[label] += 1
            continue
        plan.merged[label] += 1
        if label in existing:
            plan.replaced[label] += 1
        if apply:
            merge_row(target_result, row)
            copied = copy_artifacts(source_cell, target_cell, label)
            record_provenance(target_cell, label, source.root, copied)


def merge_source(source: Source, target: Path, apply: bool) -> MergePlan:
    """Merge every selected row of one root; returns what was (or would be) done."""
    plan = MergePlan()
    if not source.root.is_dir():
        plan.skipped_cells.append(f"{source.root}: root not found")
        return plan
    for domain, experiment, cell in iter_source_cells(source.root):
        if not fnmatch.fnmatch(experiment, source.experiments):
            continue
        target_cell = target / domain / experiment / "testing" / cell.name
        if not target_cell.is_dir():
            plan.skipped_cells.append(f"{domain}/{experiment}/{cell.name}: not in target")
            continue
        if not same_fold(cell, target_cell):
            plan.skipped_cells.append(f"{domain}/{experiment}/{cell.name}: fold_info differs")
            continue
        merge_cell(source, cell, target_cell, experiment, plan, apply)
    return plan


def _report(source: Source, plan: MergePlan, apply: bool) -> None:
    verb = "merged" if apply else "would merge"
    print(f"\n== {source.root} (experiments: {source.experiments})")
    for label in source.labels:
        n, replaced = plan.merged.get(label, 0), plan.replaced.get(label, 0)
        print(f"  {verb} {n:5d} rows of {label}" + (f"  ({replaced} replace an existing row)" if replaced else ""))
    if plan.unselected:
        print("  left behind: " + ", ".join(f"{label} x{n}" for label, n in sorted(plan.unselected.items())))
    for note in plan.skipped_cells[:10]:
        print(f"  [SKIP] {note}")
    if len(plan.skipped_cells) > 10:
        print(f"  ... and {len(plan.skipped_cells) - 10} more skipped cells")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Merge mirrored result roots into the main results tree.")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--roots", type=Path, nargs="*", default=None,
                    help="Only these roots of the config (default: all, in config order).")
    ap.add_argument("--dry-run", action="store_true", help="Report what would be merged; write nothing.")
    args = ap.parse_args(argv)

    target, sources = load_sources(args.config)
    if args.roots is not None:
        wanted = {Path(r) for r in args.roots}
        sources = [s for s in sources if s.root in wanted]
        missing = wanted - {s.root for s in sources}
        if missing:
            raise SystemExit(f"not in {args.config}: {', '.join(map(str, sorted(missing)))}")
    if not target.is_dir():
        raise SystemExit(f"target tree not found: {target}")

    total = Counter()
    skipped = 0
    for source in sources:
        plan = merge_source(source, target, apply=not args.dry_run)
        _report(source, plan, apply=not args.dry_run)
        total.update(plan.merged)
        skipped += len(plan.skipped_cells)
    print(f"\n{'Would merge' if args.dry_run else 'Merged'} {sum(total.values())} rows "
          f"({len(total)} labels) into {target}; {skipped} source cells skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
