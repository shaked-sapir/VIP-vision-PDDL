"""Rewrite pre-rename ``cdps_milp`` / ``CDPS_MILP`` arm names in a result tree.

The MILP arms were renamed to ``pisam_milp`` / ``PISAM_MILP``. Result trees
written before that carry the old spelling in four places:

- ``fold_result.json`` — the ``algorithm`` value of each MILP row.
- ``run_params.json`` — the ``run_cdps_milp`` / ``run_cdps_milp_loop`` flags and
  any MILP labels in the ``algorithms`` list.
- ``learned_domain_<label>.pddl`` — the filename.
- the arm's fold subdirectory — the directory name.

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

# Old -> new, for the three name spaces that appear on disk.
LABEL_RENAMES = {
    "CDPS_MILP_SR": PISAM_MILP_SINGLE_ROUND_ALGORITHM_NAME,
    "CDPS_MILP_LOOP": PISAM_MILP_LOOP_ALGORITHM_NAME,
}
KEY_RENAMES = {
    "cdps_milp_single_round": PISAM_MILP_SINGLE_ROUND,
    "cdps_milp_loop": PISAM_MILP_LOOP,
}
RUN_PARAM_RENAMES = {
    "run_cdps_milp": "run_pisam_milp",
    "run_cdps_milp_loop": "run_pisam_milp_loop",
}

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


def rename_domain_filename(name: str) -> Optional[str]:
    """The new ``learned_domain_<label>.pddl`` filename, or ``None``."""
    if not (name.startswith(_DOMAIN_PREFIX) and name.endswith(_DOMAIN_SUFFIX)):
        return None
    label = name[len(_DOMAIN_PREFIX):-len(_DOMAIN_SUFFIX)]
    new_label = rename_suffixed(label, LABEL_RENAMES)
    if new_label is None:
        return None
    return f"{_DOMAIN_PREFIX}{new_label}{_DOMAIN_SUFFIX}"


@dataclass
class Plan:
    """Every change the migration would make, as (path, detail) pairs."""

    fold_results: List[Tuple[Path, List[Tuple[str, str]]]] = field(default_factory=list)
    run_params: List[Tuple[Path, List[Tuple[str, str]]]] = field(default_factory=list)
    files: List[Tuple[Path, Path]] = field(default_factory=list)
    dirs: List[Tuple[Path, Path]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.fold_results or self.run_params or self.files or self.dirs)


def plan_fold_result(path: Path) -> Optional[List[Tuple[str, str]]]:
    """Label rewrites for one ``fold_result.json``, or ``None`` if unaffected.

    Raises:
        MigrationConflict: If rewriting would collide with a row that already
            carries the new label.
    """
    rows = json.loads(path.read_text())
    present = {row.get("algorithm") for row in rows}
    changes: List[Tuple[str, str]] = []
    for row in rows:
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
    return changes or None


def apply_fold_result(path: Path) -> None:
    """Rewrite the MILP ``algorithm`` labels in one ``fold_result.json``."""
    rows = json.loads(path.read_text())
    for row in rows:
        old = row.get("algorithm")
        if isinstance(old, str):
            new = rename_suffixed(old, LABEL_RENAMES)
            if new is not None:
                row["algorithm"] = new
    path.write_text(json.dumps(rows, indent=2))


def plan_run_params(path: Path) -> Optional[List[Tuple[str, str]]]:
    """Key and label rewrites for one ``run_params.json``, or ``None``.

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
    for label in params.get("algorithms", []):
        if isinstance(label, str):
            new_label = rename_suffixed(label, LABEL_RENAMES)
            if new_label is not None:
                changes.append((label, new_label))
    return changes or None


def apply_run_params(path: Path) -> None:
    """Rewrite the renamed flags and MILP labels in one ``run_params.json``."""
    params = json.loads(path.read_text())
    migrated = {RUN_PARAM_RENAMES.get(k, k): v for k, v in params.items()}
    labels = migrated.get("algorithms")
    if isinstance(labels, list):
        migrated["algorithms"] = [
            rename_suffixed(label, LABEL_RENAMES) or label
            if isinstance(label, str) else label
            for label in labels
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

        for filename in filenames:
            source = directory / filename
            if filename == "fold_result.json":
                changes = plan_fold_result(source)
                if changes:
                    plan.fold_results.append((source, changes))
                continue
            if filename == "run_params.json":
                changes = plan_run_params(source)
                if changes:
                    plan.run_params.append((source, changes))
                continue
            new_filename = rename_domain_filename(filename)
            if new_filename is not None:
                _reject_existing(directory / new_filename, source)
                plan.files.append((source, directory / new_filename))

        new_dirname = rename_suffixed(directory.name, KEY_RENAMES)
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
    """Execute a plan: rewrite JSON, then rename files, then rename directories."""
    for path, _ in plan.fold_results:
        apply_fold_result(path)
    for path, _ in plan.run_params:
        apply_run_params(path)
    for source, target in plan.files:
        source.rename(target)
    for source, target in plan.dirs:
        source.rename(target)


def summarize(plan: Plan, verbose: bool) -> str:
    """A human-readable report of what a plan changes."""
    label_totals: Dict[str, int] = {}
    for _, changes in plan.fold_results:
        for old, new in changes:
            label_totals[f"{old} -> {new}"] = label_totals.get(f"{old} -> {new}", 0) + 1

    dir_totals: Dict[str, int] = {}
    for source, target in plan.dirs:
        key = f"{source.name} -> {target.name}"
        dir_totals[key] = dir_totals.get(key, 0) + 1

    file_totals: Dict[str, int] = {}
    for source, target in plan.files:
        key = f"{source.name} -> {target.name}"
        file_totals[key] = file_totals.get(key, 0) + 1

    lines = [
        f"fold_result.json files: {len(plan.fold_results)}",
        *(f"    {k}: {v}" for k, v in sorted(label_totals.items())),
        f"run_params.json files:  {len(plan.run_params)}",
        f"renamed .pddl files:    {len(plan.files)}",
        *(f"    {k}: {v}" for k, v in sorted(file_totals.items())),
        f"renamed directories:    {len(plan.dirs)}",
        *(f"    {k}: {v}" for k, v in sorted(dir_totals.items())),
    ]
    if verbose:
        lines.append("")
        lines += [f"  RENAME {s} -> {t}" for s, t in plan.files + plan.dirs]
        lines += [f"  REWRITE {p}" for p, _ in plan.fold_results + plan.run_params]
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
