"""Write the fold manifests of an image experiment from its run config(s).

One row per (domain, fold) of every domain in the given ``source: image`` run
configs, in config order, so one domain is a block of ``n_folds`` rows. Two
files share that row order:

    <name>_symbolic_manifest.csv   domain,data_dir,experiment,fold,config
    <name>_pixel_manifest.csv      domain,data_dir,experiment,fold

The first feeds ``symbolic_fold.sbatch``, the second ``pixel_fold.sbatch``. The
experiment of an image cell is the config's ``run_name``.

    python -m scripts.cluster_image.make_image_manifests --name image_small \\
        --config benchmark/run_config_small_image.yaml benchmark/run_config_small_image_n9.yaml
    python -m scripts.cluster_image.make_image_manifests --name image_large \\
        --config benchmark/run_config_large_image.yaml
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Sequence, Tuple

import yaml

from scripts.cluster_image.make_pixel_manifest import HEADER, write_manifest

OUT_DIR = Path("scripts/cluster_image")
SYMBOLIC_HEADER = (*HEADER, "config")
DEFAULT_N_FOLDS = 5

SymbolicRow = Tuple[str, str, str, int, str]


def config_rows(config_path: Path) -> List[SymbolicRow]:
    """``(domain, data_dir, experiment, fold, config)`` for every fold of one config.

    Raises:
        ValueError: if the config is not an image config, names no run, or
            gives an absolute ``data_dir``.
    """
    config = yaml.safe_load(config_path.read_text())
    if config.get("source") != "image":
        raise ValueError(f"{config_path}: source must be 'image', got {config.get('source')!r}")
    run_name = config.get("run_name")
    if not run_name:
        raise ValueError(f"{config_path}: run_name is required")
    n_folds = int((config.get("shared") or {}).get("n_folds", DEFAULT_N_FOLDS))

    rows: List[SymbolicRow] = []
    for domain in config.get("domains") or []:
        data_dir = Path(domain["data_dir"])
        if data_dir.is_absolute():
            raise ValueError(f"{config_path}: data_dir {data_dir} must be relative to the repository")
        rows.extend(
            (domain["domain_key"], data_dir.as_posix(), run_name, fold, config_path.as_posix())
            for fold in range(n_folds)
        )
    return rows


def manifest_rows(config_paths: Sequence[Path]) -> List[SymbolicRow]:
    """The rows of every config, in the order given.

    Raises:
        ValueError: if two configs put the same domain in the same experiment.
    """
    rows: List[SymbolicRow] = []
    seen = set()
    for config_path in config_paths:
        for row in config_rows(config_path):
            cell = row[0], row[2], row[3]
            if cell in seen:
                raise ValueError(f"{row[0]} fold {row[3]} of {row[2]} appears in two configs")
            seen.add(cell)
            rows.append(row)
    return rows


def write_symbolic_manifest(rows: Sequence[SymbolicRow], out: Path) -> None:
    """Write ``rows`` under the symbolic manifest header."""
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SYMBOLIC_HEADER)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="Stem of the two manifest files.")
    parser.add_argument("--config", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    rows = manifest_rows(args.config)
    symbolic = args.out_dir / f"{args.name}_symbolic_manifest.csv"
    pixel = args.out_dir / f"{args.name}_pixel_manifest.csv"
    write_symbolic_manifest(rows, symbolic)
    write_manifest([row[:4] for row in rows], pixel)
    print(f"{len(rows)} rows -> {symbolic} and {pixel}")
    for domain in dict.fromkeys(row[0] for row in rows):
        indices = [index for index, row in enumerate(rows) if row[0] == domain]
        print(f"  {domain}: --array={indices[0]}-{indices[-1]}")


if __name__ == "__main__":
    main()
