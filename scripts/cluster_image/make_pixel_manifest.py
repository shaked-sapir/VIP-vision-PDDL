"""Write the pixel-arm fold manifest from the dashboard's current image experiments.

One row per (domain, fold) of every experiment under
``results_dashboard.image.small.experiment_dir`` in ``dashboard_config.yaml``:

    domain,data_dir,experiment,fold

``data_dir`` is the experiment's recorded data folder, rewritten relative to the
repository root, so the manifest is valid in any checkout.

    python -m scripts.cluster_image.make_pixel_manifest
    python -m scripts.cluster_image.make_pixel_manifest --domains blocksworld --out pilot.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

DASHBOARD_CONFIG = Path("benchmark/evaluation/cfm/dashboard_config.yaml")
DEFAULT_OUT = Path("scripts/cluster_image/pixel_fold_manifest.csv")
HEADER = ("domain", "data_dir", "experiment", "fold")

_DATA_MARKER = "benchmark/data/"
_FOLD_RE = re.compile(r"^fold(\d+)_")


def image_experiments(config_path: Path = DASHBOARD_CONFIG) -> Dict[str, Path]:
    """``{domain: experiment dir}`` the dashboard reads for the small image runs."""
    config = yaml.safe_load(config_path.read_text())
    block = config["results_dashboard"]["image"]["small"]["experiment_dir"]
    return {domain: Path(path.strip()) for domain, path in block.items()}


def repo_relative_data_dir(recorded: str) -> Path:
    """A recorded ``data_dir`` as a path under this repository's ``benchmark/data``.

    Raises:
        ValueError: if the recorded path does not pass through ``benchmark/data/``.
    """
    position = recorded.find(_DATA_MARKER)
    if position < 0:
        raise ValueError(f"data_dir {recorded!r} is not under {_DATA_MARKER}")
    return Path(recorded[position:])


def experiment_data_dir(experiment_dir: Path) -> Path:
    """The data folder an experiment recorded in its ``run_params.json``."""
    run_params = experiment_dir / "evaluation_results" / "run_params.json"
    return repo_relative_data_dir(json.loads(run_params.read_text())["data_dir"])


def experiment_folds(experiment_dir: Path) -> List[int]:
    """The fold indices present under an experiment's ``testing/``."""
    folds = set()
    for cell in (experiment_dir / "testing").iterdir():
        match = _FOLD_RE.match(cell.name)
        if cell.is_dir() and match:
            folds.add(int(match.group(1)))
    return sorted(folds)


def manifest_rows(
    experiments: Dict[str, Path], domains: Optional[Iterable[str]] = None
) -> List[Tuple[str, str, str, int]]:
    """One ``(domain, data_dir, experiment name, fold)`` row per fold."""
    wanted = set(domains) if domains else None
    rows: List[Tuple[str, str, str, int]] = []
    for domain, experiment_dir in sorted(experiments.items()):
        if wanted is not None and domain not in wanted:
            continue
        data_dir = experiment_data_dir(experiment_dir)
        for fold in experiment_folds(experiment_dir):
            rows.append((domain, data_dir.as_posix(), experiment_dir.name, fold))
    return rows


def write_manifest(rows: List[Tuple[str, str, str, int]], out: Path) -> None:
    """Write ``rows`` under the manifest header."""
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    rows = manifest_rows(image_experiments(), args.domains)
    write_manifest(rows, args.out)
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
