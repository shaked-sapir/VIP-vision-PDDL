"""Rewrite superseded arm names in a result tree, and drop retired arms' rows.

Two renames are folded in here:

- the MILP arms ``cdps_milp`` / ``CDPS_MILP`` became ``pisam_milp`` /
  ``PISAM_MILP``;
- the ROSAME family gained the paper year it implements, so ``ROSAME`` became
  ``ROSAME_24`` and so on for the three variants built on it.

Result trees written before a rename carry the old spelling in four places:

- ``fold_result.json`` — the ``algorithm`` value of each affected row.
- ``run_params.json`` — the ``run_cdps_milp`` / ``run_cdps_milp_loop`` flags and
  any affected labels in the ``algorithms`` list.
- ``learned_domain_<label>.pddl`` — the filename.
- the arm's subdirectory — under the fold for our own learner, under
  ``baseline_models/`` for a baseline.

Labels in :data:`LABEL_PURGES` name arms that no longer exist. Their rows,
files and directories are *deleted*, not renamed: an arm the code can no longer
produce but whose rows are still on disk keeps rendering in the dashboard, since
the series list is built from the data present.

Ablated arms carry a ``__suffix`` (e.g. ``CDPS_MILP_SR__eq16=0.4``); the suffix
set is open, so names are matched on the part before ``__``.

Usage::

    python -m benchmark.migrate_arm_names --dry-run
    python -m benchmark.migrate_arm_names
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmark.algorithms import (  # noqa: E402
    PISAM_MILP_LOOP,
    PISAM_MILP_LOOP_ALGORITHM_NAME,
    PISAM_MILP_SINGLE_ROUND,
    PISAM_MILP_SINGLE_ROUND_ALGORITHM_NAME,
)

DEFAULT_ROOT = Path(__file__).resolve().parent / "running_results"

# Old -> new, for the three name spaces that appear on disk. The ROSAME right-hand
# sides are literals rather than imports because reaching the runners means
# importing torch, which a filesystem migration has no business needing.
LABEL_RENAMES = {
    "CDPS_MILP_SR": PISAM_MILP_SINGLE_ROUND_ALGORITHM_NAME,
    "CDPS_MILP_LOOP": PISAM_MILP_LOOP_ALGORITHM_NAME,
    "ROSAME": "ROSAME_24",
    "ROSAME_MILP": "ROSAME_MILP_24",
    "ROSAME_MILP_TAG": "ROSAME_MILP_24_TAG",
    "ROSAME-I": "ROSAME-I_24",
    "ROSAME-I_MILP": "ROSAME-I_MILP_24",
}
KEY_RENAMES = {
    "cdps_milp_single_round": PISAM_MILP_SINGLE_ROUND,
    "cdps_milp_loop": PISAM_MILP_LOOP,
}
RUN_PARAM_RENAMES = {
    "run_cdps_milp": "run_pisam_milp",
    "run_cdps_milp_loop": "run_pisam_milp_loop",
}

# Labels whose rows, files and directories are deleted outright.
LABEL_PURGES = frozenset({"ROSAME_MILP_BASE"})

_SUFFIX_SEP = "__"
_DOMAIN_PREFIX = "learned_domain_"
_DOMAIN_SUFFIX = ".pddl"


class MigrationConflict(RuntimeError):
    """Both the old and the new spelling exist, so the merge is ambiguous."""


def rename_suffixed(name: str, renames: Dict[str, str]) -> Optional[str]:
    """The new spelling of ``name``, or ``None`` if its stem is not renamed.

    ``name`` is either a bare stem or ``stem__suffix``; only the stem is looked
    up, and the suffix is carried over unchanged.

    Args:
        name: The on-disk or in-JSON name.
        renames: Old stem -> new stem.

    Returns:
        The renamed name, or ``None`` when the stem is absent from ``renames``.
    """
    stem, sep, suffix = name.partition(_SUFFIX_SEP)
    new_stem = renames.get(stem)
    if new_stem is None:
        return None
    return new_stem + sep + suffix


def is_purged(name: Optional[str]) -> bool:
    """Whether ``name``'s stem names an arm that no longer exists."""
    if not isinstance(name, str):
        return False
    return name.partition(_SUFFIX_SEP)[0] in LABEL_PURGES


def domain_filename_label(name: str) -> Optional[str]:
    """The label inside a ``learned_domain_<label>.pddl`` filename, or ``None``."""
    if not (name.startswith(_DOMAIN_PREFIX) and name.endswith(_DOMAIN_SUFFIX)):
        return None
    return name[len(_DOMAIN_PREFIX):-len(_DOMAIN_SUFFIX)]


def domain_filename(label: str) -> str:
    """The ``learned_domain_<label>.pddl`` filename for ``label``."""
    return f"{_DOMAIN_PREFIX}{label}{_DOMAIN_SUFFIX}"


@dataclass
class Plan:
    """Every change the migration would make, as (path, detail) pairs."""

    fold_results: List[Tuple[Path, List[Tuple[str, str]]]] = field(default_factory=list)
    run_params: List[Tuple[Path, List[Tuple[str, str]]]] = field(default_factory=list)
    files: List[Tuple[Path, Path]] = field(default_factory=list)
    dirs: List[Tuple[Path, Path]] = field(default_factory=list)
    dropped_labels: List[Tuple[Path, List[str]]] = field(default_factory=list)
    deletions: List[Path] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (
            self.fold_results or self.run_params or self.files or self.dirs
            or self.dropped_labels or self.deletions
        )


def plan_fold_result(path: Path) -> Tuple[List[Tuple[str, str]], List[str]]:
    """``(label rewrites, dropped labels)`` for one ``fold_result.json``.

    Raises:
        MigrationConflict: If rewriting would collide with a row that already
            carries the new label.
    """
    rows = json.loads(path.read_text())
    dropped = [row["algorithm"] for row in rows if is_purged(row.get("algorithm"))]
    survivors = [row for row in rows if not is_purged(row.get("algorithm"))]

    present = {row.get("algorithm") for row in survivors}
    changes: List[Tuple[str, str]] = []
    for row in survivors:
        old = row.get("algorithm")
        if not isinstance(old, str):
            continue
        new = rename_suffixed(old, LABEL_RENAMES)
        if new is None:
            continue
        if new in present:
            raise MigrationConflict(
                f"{path}: rows for both `{old}` and `{new}` exist; "
                f"delete whichever is stale before migrating."
            )
        changes.append((old, new))
    return changes, dropped


def apply_fold_result(path: Path) -> None:
    """Drop purged rows and rewrite the renamed labels in one result file."""
    rows = [
        row for row in json.loads(path.read_text())
        if not is_purged(row.get("algorithm"))
    ]
    for row in rows:
        old = row.get("algorithm")
        if isinstance(old, str):
            new = rename_suffixed(old, LABEL_RENAMES)
            if new is not None:
                row["algorithm"] = new
    path.write_text(json.dumps(rows, indent=2))


def plan_run_params(path: Path) -> Tuple[List[Tuple[str, str]], List[str]]:
    """``(key/label rewrites, dropped labels)`` for one ``run_params.json``.

    Raises:
        MigrationConflict: If a renamed key already exists under its new name.
    """
    params = json.loads(path.read_text())
    changes: List[Tuple[str, str]] = []
    for old_key, new_key in RUN_PARAM_RENAMES.items():
        if old_key not in params:
            continue
        if new_key in params:
            raise MigrationConflict(
                f"{path}: both `{old_key}` and `{new_key}` are set; "
                f"delete whichever is stale before migrating."
            )
        changes.append((old_key, new_key))

    dropped: List[str] = []
    for label in params.get("algorithms", []):
        if not isinstance(label, str):
            continue
        if is_purged(label):
            dropped.append(label)
            continue
        new_label = rename_suffixed(label, LABEL_RENAMES)
        if new_label is not None:
            changes.append((label, new_label))
    return changes, dropped


def apply_run_params(path: Path) -> None:
    """Rewrite the renamed flags and drop the purged labels in one file."""
    params = json.loads(path.read_text())
    migrated = {RUN_PARAM_RENAMES.get(k, k): v for k, v in params.items()}
    labels = migrated.get("algorithms")
    if isinstance(labels, list):
        migrated["algorithms"] = [
            rename_suffixed(label, LABEL_RENAMES) or label
            if isinstance(label, str) else label
            for label in labels
            if not is_purged(label)
        ]
    path.write_text(json.dumps(migrated, indent=2))


def build_plan(root: Path) -> Plan:
    """Walk ``root`` and collect every rename and rewrite it needs.

    Raises:
        MigrationConflict: If any old/new pair already coexists.
    """
    plan = Plan()
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        directory = Path(dirpath)

        if is_purged(directory.name):
            plan.deletions.append(directory)
            continue

        for filename in filenames:
            source = directory / filename
            if filename == "fold_result.json":
                changes, dropped = plan_fold_result(source)
                if changes:
                    plan.fold_results.append((source, changes))
                if dropped:
                    plan.dropped_labels.append((source, dropped))
                continue
            if filename == "run_params.json":
                changes, dropped = plan_run_params(source)
                if changes:
                    plan.run_params.append((source, changes))
                if dropped:
                    plan.dropped_labels.append((source, dropped))
                continue
            label = domain_filename_label(filename)
            if label is None:
                continue
            if is_purged(label):
                plan.deletions.append(source)
                continue
            new_label = rename_suffixed(label, LABEL_RENAMES)
            if new_label is not None:
                target = directory / domain_filename(new_label)
                _reject_existing(target, source)
                plan.files.append((source, target))

        # Our own learner's directories carry the registry key, a baseline's the
        # row label, so both maps are consulted.
        new_dirname = rename_suffixed(directory.name, KEY_RENAMES) or rename_suffixed(
            directory.name, LABEL_RENAMES
        )
        if new_dirname is not None:
            target = directory.parent / new_dirname
            _reject_existing(target, directory)
            plan.dirs.append((directory, target))

    return plan


def _reject_existing(target: Path, source: Path) -> None:
    if target.exists():
        raise MigrationConflict(
            f"{target} already exists, so `{source}` cannot be renamed onto it; "
            f"delete whichever is stale before migrating."
        )


def apply_plan(plan: Plan) -> None:
    """Execute a plan: delete, then rewrite JSON, then rename files and dirs."""
    for path in plan.deletions:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

    # A file can be listed under both a rewrite and a drop; rewrite it once.
    touched = {
        path
        for path, _ in plan.fold_results + plan.run_params + plan.dropped_labels
    }
    for path in sorted(touched):
        if path.name == "fold_result.json":
            apply_fold_result(path)
        else:
            apply_run_params(path)

    for source, target in plan.files:
        source.rename(target)
    for source, target in plan.dirs:
        source.rename(target)


def _tally(keys: List[str]) -> List[str]:
    """Indented ``key: count`` lines, sorted by key."""
    totals: Dict[str, int] = {}
    for key in keys:
        totals[key] = totals.get(key, 0) + 1
    return [f"    {k}: {v}" for k, v in sorted(totals.items())]


def summarize(plan: Plan, verbose: bool) -> str:
    """A human-readable report of what a plan changes."""
    label_totals = _tally(
        [f"{old} -> {new}" for _, changes in plan.fold_results for old, new in changes]
    )
    file_totals = _tally([f"{s.name} -> {t.name}" for s, t in plan.files])
    dir_totals = _tally([f"{s.name} -> {t.name}" for s, t in plan.dirs])
    drop_totals = _tally(
        [label for _, labels in plan.dropped_labels for label in labels]
    )
    deletion_totals = _tally([path.name for path in plan.deletions])

    lines = [
        f"fold_result.json files: {len(plan.fold_results)}",
        *label_totals,
        f"run_params.json files:  {len(plan.run_params)}",
        f"renamed .pddl files:    {len(plan.files)}",
        *file_totals,
        f"renamed directories:    {len(plan.dirs)}",
        *dir_totals,
        f"DROPPED labels:         {sum(len(l) for _, l in plan.dropped_labels)}",
        *drop_totals,
        f"DELETED paths:          {len(plan.deletions)}",
        *deletion_totals,
    ]
    if verbose:
        lines.append("")
        lines += [f"  DELETE {p}" for p in plan.deletions]
        lines += [f"  RENAME {s} -> {t}" for s, t in plan.files + plan.dirs]
        lines += [
            f"  REWRITE {p}"
            for p, _ in plan.fold_results + plan.run_params + plan.dropped_labels
        ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "root", nargs="?", type=Path, default=DEFAULT_ROOT,
        help=f"result tree to migrate (default: {DEFAULT_ROOT})",
    )
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    parser.add_argument("--verbose", action="store_true", help="list every path")
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        parser.error(f"{args.root} is not a directory")

    try:
        plan = build_plan(args.root)
    except MigrationConflict as error:
        print(f"ABORTED: {error}", file=sys.stderr)
        return 1

    if plan.is_empty:
        print(f"{args.root}: nothing to migrate.")
        return 0

    print(summarize(plan, args.verbose))
    if args.dry_run:
        print("\n(dry run — nothing written)")
        return 0

    apply_plan(plan)
    print("\nMigrated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
