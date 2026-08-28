"""Expand ``run_config_large.yaml`` into a one-cell-per-row SLURM manifest.

Unlike ``scripts/cluster/make_manifest.py`` (the small-data sweep), a row here is
one ``(domain, p_mask, p_noise)`` cell and the **whole L sweep runs inside it**.
Every training size and every arm of a cell therefore lands on one node, on one
CPU model, which is what makes them comparable: a wall-clock learning budget is
hardware-dependent, so an L=10 row on a different node than its L=2000 sibling
would not be.

    python scripts/cluster_large/make_manifest.py
    python scripts/cluster_large/make_manifest.py --domains blocksworld hanoi

Writes ``manifest.csv`` next to this file. The sbatch template reads a row and
passes it to ``benchmark_runner`` as ``--domains/--only-mask/--only-noise``, so
the config stays the single source of truth for arms, the L list, patience and
subset size.
"""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path
from typing import Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "benchmark" / "run_config_large.yaml"
HEADER = ["domain_key", "data_dir", "p_mask", "p_noise", "run_name"]


def load_config(path: Path) -> dict:
    """Parse the run config, or raise if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"run config not found: {path}")
    return yaml.safe_load(path.read_text())


def build_rows(config: dict, only_domains: List[str] | None) -> List[Dict[str, str]]:
    """One row per (domain, mask, noise). L is not expanded — it stays in the config.

    Raises:
        ValueError: on a non-simulated config, an empty grid, or a ``--domains``
            key the config does not define.
    """
    if config.get("source") != "simulated":
        raise ValueError(
            f"source must be 'simulated' for this sweep, got {config.get('source')!r}"
        )
    domains = config.get("domains") or []
    if only_domains:
        known = {d["domain_key"] for d in domains}
        unknown = sorted(set(only_domains) - known)
        if unknown:
            raise ValueError(f"--domains {unknown} not in the config: {sorted(known)}")
        domains = [d for d in domains if d["domain_key"] in only_domains]

    grid = (config.get("simulation") or {}).get("grid") or {}
    masking_ps, noising_ps = grid.get("masking_ps"), grid.get("noising_ps")
    if not masking_ps or not noising_ps:
        raise ValueError("simulation.grid must define masking_ps and noising_ps")

    run_name = config.get("run_name")
    if not run_name:
        raise ValueError("run config must define run_name")

    return [
        {"domain_key": d["domain_key"], "data_dir": d["data_dir"],
         "p_mask": mp, "p_noise": np_, "run_name": run_name}
        for d, mp, np_ in product(domains, masking_ps, noising_ps)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--domains", nargs="*", default=None, metavar="DOMAIN_KEY",
                        help="Subset of domain_keys (default: every one in the config)")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    rows = build_rows(load_config(args.config), args.domains)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "manifest.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)

    domains = sorted({r["domain_key"] for r in rows})
    print(f"Wrote {out} — {len(rows)} cell(s) over {len(domains)} domain(s): "
          f"{', '.join(domains)}")
    print(f"Submit with:  scripts/cluster_large/submit.sh")
    print(f"Array range:  0-{len(rows) - 1}")


if __name__ == "__main__":
    main()
