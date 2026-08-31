"""Per-fold convergence series for the arms, read from their own artifacts.

Two arm families converge in different units and record it in different places:

* ``pisam_milp_loop*`` writes one row per loop round to
  ``<fold>/<arm_dir>/milp_loop_rounds.jsonl`` (streamed live) with ``v_raw`` --
  the ground-truth-free reconstruction score it minimises.
* ``rosame*`` writes one row per training epoch to
  ``<fold>/anytime_snapshots/<arm>/snapshots.json``, with ``loss`` and, for the
  MILP arms, the ``agreement`` known at that epoch.

Both are exposed on one x axis (round / epoch) so the dashboard can show a
family per panel and toggle which quantity is plotted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

ROUND_STREAM = "milp_loop_rounds.jsonl"
ROUND_FILE = "milp_loop_rounds.json"
SNAPSHOT_INDEX = "snapshots.json"
_INSTANCE_RE = re.compile(r"^fold(\d+)_numtrajs(\d+)_gtrate(\d+)$")


def _rows(path: Path) -> List[dict]:
    """JSONL rows, skipping a torn final line (the stream is written live)."""
    out: List[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def pisam_series(instance_dir: Path) -> Dict[str, dict]:
    """``{arm: {"x": [round], "v": [...], "improved": [...]}}`` for one fold.

    Prefers the live JSONL; falls back to the end-of-run JSON, which carries the
    same rows under ``rounds``.
    """
    out: Dict[str, dict] = {}
    for arm_dir in sorted(instance_dir.glob("pisam_milp_*")):
        if not arm_dir.is_dir():
            continue
        rows: List[dict] = []
        if (arm_dir / ROUND_STREAM).is_file():
            rows = _rows(arm_dir / ROUND_STREAM)
        elif (arm_dir / ROUND_FILE).is_file():
            try:
                rows = json.loads((arm_dir / ROUND_FILE).read_text()).get("rounds", [])
            except (json.JSONDecodeError, OSError):
                rows = []
        scored = [r for r in rows if r.get("v_raw") is not None]
        if not scored:
            continue
        out[arm_dir.name] = {
            "x": [r["round"] for r in scored],
            "v": [r["v_raw"] for r in scored],
            "improved": [bool(r.get("improved")) for r in scored],
        }
    return out


def rosame_series(instance_dir: Path) -> Dict[str, dict]:
    """``{arm: {"x": [epoch], "loss": [...], "agreement": [...|None]}}``.

    ``agreement`` is present only for the arms that run a solver; a DL-only arm
    leaves it null at every epoch.
    """
    out: Dict[str, dict] = {}
    root = instance_dir / "anytime_snapshots"
    if not root.is_dir():
        return out
    for arm_dir in sorted(root.iterdir()):
        index = arm_dir / SNAPSHOT_INDEX
        if not index.is_file():
            continue
        try:
            payload = json.loads(index.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        records = payload if isinstance(payload, list) else payload.get(
            "snapshots", payload.get("records", []))
        rows = [r for r in records if r.get("loss") is not None]
        if not rows:
            continue
        out[arm_dir.name] = {
            "x": [r["epoch"] for r in rows],
            "loss": [r["loss"] for r in rows],
            "agreement": [r.get("agreement") for r in rows],
        }
    return out


def pre_mip_epoch(instance_dir: Path) -> Optional[int]:
    """Epoch at which the MILP starts, for the boundary marker. ``None`` if absent."""
    marker = instance_dir / "fold_result.json"
    if not marker.is_file():
        return None
    try:
        rows = json.loads(marker.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    for row in rows:
        pre = (row.get("algorithm_specific") or {}).get("pre_mip_epochs")
        if pre is not None:
            return int(pre)
    return None


def cell_convergence(testing_dir: Path) -> dict:
    """Convergence series for one cell, grouped by family and training size.

    Returns ``{"pisam": {ntraj: {arm: series}}, "rosame": {...},
    "pre_mip": {ntraj: epoch}}`` -- folds are kept separate so the dashboard can
    aggregate them into a mean and a band itself, as it does for the learning
    curves.
    """
    pisam: Dict[int, Dict[str, list]] = {}
    rosame: Dict[int, Dict[str, list]] = {}
    pre_mip: Dict[int, int] = {}
    if not testing_dir.is_dir():
        return {"pisam": {}, "rosame": {}, "pre_mip": {}}

    for inst in sorted(testing_dir.iterdir()):
        m = _INSTANCE_RE.match(inst.name)
        if not m:
            continue
        ntraj = int(m.group(2))
        for arm, series in pisam_series(inst).items():
            pisam.setdefault(ntraj, {}).setdefault(arm, []).append(series)
        for arm, series in rosame_series(inst).items():
            rosame.setdefault(ntraj, {}).setdefault(arm, []).append(series)
        if ntraj not in pre_mip:
            pre = pre_mip_epoch(inst)
            if pre is not None:
                pre_mip[ntraj] = pre
    return {"pisam": pisam, "rosame": rosame, "pre_mip": pre_mip}
