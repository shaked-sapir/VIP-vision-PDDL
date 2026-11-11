"""Record a fold's realised corruption in its ``fold_info.json``.

An image fold has no nominal mask or noise rate: the VLM hid and flipped
whatever it hid and flipped. The fields written here are what place such a
fold on the simulated grid. They are measured against the corpus's
``gt_trajectories/`` by :func:`~benchmark.algorithm_adapters.lamanna.realised_rates.realised_corruption`,
the one measurement every arm's realised rate comes from.

``fold_info.json`` gains::

    "realised_mask_rate":   hidden atoms / all atoms, every frame
    "realised_flip_rate":   flipped atoms / visible atoms, every frame
    "flip_persistence":     of the flipped atoms with a next frame, the fraction
                            flipped again in it (null when nothing was flipped)
    "n_atoms":              atoms over every frame of the fold's observations
    "realised_corruption":  the counts behind those, frame 0 and frames 1..T
                            apart, and the GT root they were measured against

Two callers: ``run_fold.run_single_fold`` right after it prepares an image
fold, and ``benchmark/backfill_realised_corruption.py`` for cells that ran
before the fields existed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from benchmark.algorithm_adapters.lamanna.realised_rates import (
    RealisedCorruption,
    RealisedRates,
    realised_corruption,
)
from src.utils.atomic_json import read_json_or_none, write_json_atomic

FOLD_INFO_FILENAME = "fold_info.json"
CORRUPTION_KEY = "realised_corruption"

#: The fold_info.json keys this module owns, in the order they are written.
TOP_LEVEL_FIELDS = ("realised_mask_rate", "realised_flip_rate", "flip_persistence", "n_atoms")


def _rates_block(rates: RealisedRates) -> Dict[str, object]:
    """Counts and rates of one frame group, ``flip_rate`` spelled as the fold_info does."""
    return {
        "n_states": rates.n_states,
        "n_atoms": rates.n_atoms,
        "n_masked": rates.n_masked,
        "n_flipped": rates.n_flipped,
        "mask_rate": rates.mask_rate,
        "flip_rate": rates.noise_rate,
    }


def corruption_fields(
    corruption: RealisedCorruption, gt_root: Optional[Path] = None
) -> Dict[str, object]:
    """The ``fold_info.json`` fields for one measured fold."""
    return {
        "realised_mask_rate": corruption.total.mask_rate,
        "realised_flip_rate": corruption.total.noise_rate,
        "flip_persistence": corruption.flip_persistence,
        "n_atoms": corruption.total.n_atoms,
        CORRUPTION_KEY: {
            **_rates_block(corruption.total),
            "frame0": _rates_block(corruption.frame0),
            "later_frames": _rates_block(corruption.later_frames),
            "n_flipped_with_next": corruption.n_flipped_with_next,
            "n_flip_persisted": corruption.n_flip_persisted,
            "flip_persistence": corruption.flip_persistence,
            "gt_root": str(gt_root) if gt_root is not None else None,
        },
    }


def write_fold_corruption(
    fold_dir: Path, corruption: RealisedCorruption, gt_root: Optional[Path] = None
) -> Path:
    """Merge the corruption fields into ``<fold_dir>/fold_info.json``; other keys are kept."""
    path = Path(fold_dir) / FOLD_INFO_FILENAME
    metadata = read_json_or_none(path) or {}
    metadata.update(corruption_fields(corruption, gt_root))
    write_json_atomic(path, metadata)
    return path


def read_fold_corruption(fold_dir: Path) -> Optional[Dict[str, object]]:
    """The recorded ``realised_corruption`` block of a fold, or ``None`` if never measured."""
    metadata = read_json_or_none(Path(fold_dir) / FOLD_INFO_FILENAME) or {}
    block = metadata.get(CORRUPTION_KEY)
    return block if isinstance(block, dict) else None


def record_fold_corruption(
    fold_dir: Path,
    prepared_trajectories: Sequence[Tuple],
    domain_path: Path,
    gt_lookup: Mapping[str, Path],
    gt_root: Optional[Path] = None,
    tag: str = "fold",
) -> Optional[RealisedCorruption]:
    """Measure a fold's observations against the GT and record the result.

    A fold whose GT cannot be matched (a problem with no GT trajectory, or
    one of a different length) is reported and left unrecorded rather than
    failed: the fold's learning does not depend on this number.
    """
    try:
        corruption = realised_corruption(prepared_trajectories, domain_path, gt_lookup)
    except (KeyError, ValueError) as err:
        print(f"  [{tag}] realised corruption not recorded: {err}")
        return None
    write_fold_corruption(fold_dir, corruption, gt_root)
    print(
        f"  [{tag}] realised corruption: mask {corruption.total.mask_rate:.3f}, "
        f"flip {corruption.total.noise_rate:.3f} "
        f"(frame 0 {corruption.frame0.noise_rate:.3f}, frames 1..T "
        f"{corruption.later_frames.noise_rate:.3f}), persistence "
        f"{_fmt(corruption.flip_persistence)} over {corruption.total.n_atoms} atoms"
    )
    return corruption


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"
