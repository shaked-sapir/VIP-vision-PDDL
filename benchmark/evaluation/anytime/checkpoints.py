"""Normalise each arm's intermediate models into one checkpoint stream.

Every arm already writes the models it produced along the way, but no two write
them the same way, and the differences are not cosmetic -- they follow from how
each arm works:

============================  =========================================  ==================================================
arm                           timestamps                                 models
============================  =========================================  ==================================================
``cdps`` / ``cdps_anchored``  ``conflict_free_solutions_log.json``        ``conflict_free_models/conflict_free_model_{i}/``
``pisam_milp_single_round``    same, or ``final_model/patch_details``      ``conflict_free_models/{conflict_free_model_0,final_model}/``
``pisam_milp_loop``            ``milp_loop_rounds.json``                   ``milp_loop_round_models/round_{i}/``
``ROSAME``                    ``anytime_snapshots/ROSAME/snapshots.json`` ``anytime_snapshots/ROSAME/snapshot_{i:04d}.pddl``
============================  =========================================  ==================================================

The single-round arm needs the extra fallback because it writes an *empty*
solutions log when it ends with conflicts, parking its model under
``final_model/`` instead. Dropping that case would silently erase the arm from
the plot in exactly the runs where it did worst, which is the one direction of
error a performance profile must not have.

All four clocks share an origin -- time since that arm began work on this fold --
which is what makes them plottable on one axis. They are not identical in what
they include: ROSAME's excludes its snapshot overhead (see
``benchmark.algorithm_adapters.anytime_snapshots``), while the CDPS and loop
timestamps include their own model writes. That asymmetry is left in rather than
modelled away because the CDPS-side writes are milliseconds against solves
measured in seconds, whereas ROSAME's were 12-17% of an epoch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

# Fold subdirectory per arm. ``cdps`` is the fold root itself: it ran first and
# owns the fold's top-level artifacts, which is also why its models are the ones
# the rest of the report stack reads without qualification.
ARM_SUBDIRS: Dict[str, str] = {
    "cdps": ".",
    "cdps_anchored": "cdps_anchored",
    "pisam_milp_single_round": "pisam_milp_single_round",
    "pisam_milp_loop": "pisam_milp_loop",
    "ROSAME": ".",
}

_SOLUTIONS_LOG = "conflict_free_solutions_log.json"
_MODELS_DIR = "conflict_free_models"
_ROUND_LOG = "milp_loop_rounds.json"
_ROUND_MODELS_DIR = "milp_loop_round_models"
_SNAPSHOT_DIR = "anytime_snapshots"
_SNAPSHOT_INDEX = "snapshots.json"


@dataclass(frozen=True)
class Checkpoint:
    """One model an arm had available, and when it had it.

    Attributes:
        arm: Which arm produced it, as keyed in :data:`ARM_SUBDIRS`.
        index: Position in that arm's own sequence, 0-based.
        elapsed_seconds: Seconds since the arm started work on this fold.
        model_path: The ``.pddl`` file. Not yet parsed -- scoring is a separate
            step so that reading a fold stays cheap.
    """

    arm: str
    index: int
    elapsed_seconds: float
    model_path: Path


def _load_json(path: Path) -> Optional[object]:
    """Parse a JSON artifact, or return ``None`` if it is absent or malformed.

    A truncated log means the run died mid-write; that arm simply has no curve
    in this fold, which is a fact about the run and not a reason to abort the
    whole report.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"  Warning: {path} is not valid JSON ({exc}); skipping")
        return None


def read_cdps_checkpoints(root: Path, arm: str) -> List[Checkpoint]:
    """Checkpoints from a CDPS-shaped artifact suite.

    Covers plain CDPS, the anchored variant, and the single-round MILP arm,
    which deliberately mirrors the same layout.
    """
    root = Path(root)
    entries = _load_json(root / _SOLUTIONS_LOG)
    checkpoints: List[Checkpoint] = []

    if isinstance(entries, list) and entries:
        for position, entry in enumerate(entries):
            elapsed = entry.get("wall_time_so_far")
            index = entry.get("index", position)
            model = root / _MODELS_DIR / f"conflict_free_model_{index}" / "model.pddl"
            if elapsed is None:
                print(f"  Warning: {arm} solution {index} has no wall_time_so_far; skipping")
                continue
            if not model.exists():
                print(f"  Warning: {arm} solution {index} has no model at {model}; skipping")
                continue
            checkpoints.append(Checkpoint(arm, position, float(elapsed), model))
        return checkpoints

    return _read_final_model_fallback(root, arm)


def _read_final_model_fallback(root: Path, arm: str) -> List[Checkpoint]:
    """The single conflicted model an arm parks under ``final_model/``.

    This is one point, not a curve, and that is the honest shape: the arm never
    had anything else to offer.
    """
    model = Path(root) / _MODELS_DIR / "final_model" / "model.pddl"
    if not model.exists():
        return []

    details = _load_json(model.parent / "patch_details.json")
    elapsed = details.get("wall_time_seconds") if isinstance(details, dict) else None
    if elapsed is None:
        print(f"  Warning: {arm} final model has no wall_time_seconds; skipping")
        return []
    return [Checkpoint(arm, 0, float(elapsed), model)]


def read_loop_checkpoints(root: Path, arm: str) -> List[Checkpoint]:
    """One checkpoint per MILP-loop round, losers included.

    Rounds that lost to the incumbent are kept: the curve's running-best will
    ignore them, but the per-round scatter the plan asks for is exactly the set
    of things the loop tried and rejected.
    """
    root = Path(root)
    payload = _load_json(root / _ROUND_LOG)
    if not isinstance(payload, dict):
        return []

    checkpoints: List[Checkpoint] = []
    for entry in payload.get("rounds", []):
        round_index = entry.get("round")
        elapsed = entry.get("elapsed_seconds")
        model = root / _ROUND_MODELS_DIR / f"round_{round_index}" / "model.pddl"
        if elapsed is None or round_index is None:
            print(f"  Warning: {arm} round entry is missing round/elapsed_seconds; skipping")
            continue
        if not model.exists():
            # Unsolved rounds produce no model; the log still records the attempt.
            continue
        checkpoints.append(Checkpoint(arm, int(round_index), float(elapsed), model))
    return checkpoints


def read_snapshot_checkpoints(root: Path, arm: str) -> List[Checkpoint]:
    """Checkpoints written by a :class:`SnapshotWriter`, e.g. ROSAME's."""
    snapshot_dir = Path(root) / _SNAPSHOT_DIR / arm
    payload = _load_json(snapshot_dir / _SNAPSHOT_INDEX)
    if not isinstance(payload, dict):
        return []

    checkpoints: List[Checkpoint] = []
    for entry in payload.get("snapshots", []):
        model = snapshot_dir / entry["path"]
        if not model.exists():
            print(f"  Warning: {arm} snapshot {entry['index']} missing at {model}; skipping")
            continue
        checkpoints.append(
            Checkpoint(arm, int(entry["index"]), float(entry["elapsed_seconds"]), model)
        )
    return checkpoints


# Which reader each arm needs. Adding an arm is one entry here plus one in
# ARM_SUBDIRS -- if it writes a shape one of these already reads, that is all.
_READERS: Dict[str, Callable[[Path, str], List[Checkpoint]]] = {
    "cdps": read_cdps_checkpoints,
    "cdps_anchored": read_cdps_checkpoints,
    "pisam_milp_single_round": read_cdps_checkpoints,
    "pisam_milp_loop": read_loop_checkpoints,
    "ROSAME": read_snapshot_checkpoints,
}


def read_fold_checkpoints(
    fold_dir: Path, arms: Optional[List[str]] = None
) -> Dict[str, List[Checkpoint]]:
    """Every checkpoint a fold holds, keyed by arm and sorted by time.

    Arms that left no artifacts are omitted rather than mapped to an empty list,
    so a caller can tell "did not run here" from "ran and produced nothing".

    Args:
        fold_dir: A ``testing/fold*`` directory.
        arms: Restrict to these arms. Defaults to all of :data:`ARM_SUBDIRS`.
    """
    fold_dir = Path(fold_dir)
    selected = arms if arms is not None else list(ARM_SUBDIRS)

    found: Dict[str, List[Checkpoint]] = {}
    for arm in selected:
        if arm not in _READERS:
            raise KeyError(f"unknown arm {arm!r}; known arms: {sorted(ARM_SUBDIRS)}")
        root = fold_dir / ARM_SUBDIRS[arm]
        checkpoints = _READERS[arm](root, arm)
        if checkpoints:
            found[arm] = sorted(checkpoints, key=lambda c: c.elapsed_seconds)
    return found
