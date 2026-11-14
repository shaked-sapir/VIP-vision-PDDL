"""List the image experiments for the results dashboard, from their variant records.

    python -m benchmark.image_dashboard            # write the lists, then build the dashboard
    python -m benchmark.image_dashboard --no-build

Every image experiment that carries an ``experiment_variant.json`` becomes one
entry of the dashboard's "Image variant" selector, for the data size its name
family belongs to. The lists are written to
``benchmark/evaluation/cfm/generated/image_variants_<size>.yaml``, which
``dashboard_config.yaml`` names as ``image.<size>.variants_file``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from benchmark.corpus_variants import data_variant_labels
from benchmark.image_variant_config import corpus_family
from src.utils.atomic_json import read_json_or_none
from src.utils.corpus_variant import EXPERIMENT_VARIANT_FILE
from src.utils.gt_setup import GtSetup

GENERATED_DIR = Path("benchmark/evaluation/cfm/generated")
#: Experiment-name family of each data size's image runs.
FAMILIES = {"small": "TO=600__image-small", "large": "large-corpora__image-mini__N=100"}

SETUP_TEXT = {
    GtSetup.NONE: "no GT", GtSetup.INIT: "init GT",
    GtSetup.FINAL: "final GT", GtSetup.INIT_FINAL: "init+final GT",
}
#: Arms whose ground truth is the experiment's setup, keyed by row-label prefix.
SETUP_ARMS = {
    "PISAM_MILP_LOOP": "NPI-SAM", "PISAM": "PI-SAM",
    "ROSAME_MILP_24": "ROSAME+MILP", "ROSAME_24": "ROSAME",
}
#: Pixel arms, whose ground truth is their own and does not follow the setup.
PIXEL_ARMS = {
    "ROSAME-I_24": "ROSAME-I 24 (final GT)",
    "ROSAME-I_MILP_24": "ROSAME-I 24+MILP (init+final GT)",
    "ROSAME-I_26": "ROSAME-I 26 (init+final GT)",
    "ROSAME-I_MILP_26": "ROSAME-I 26+MILP (init+final GT)",
}


def variant_key(axes: Dict[str, str]) -> str:
    """A stable selector key for one set of axes."""
    return "__".join(f"{axis}={axes[axis]}" for axis in ("vlm_model", "reasoning", "example_protocol", "gt_setup"))


def variant_label(axes: Dict[str, str]) -> str:
    """The selector text for one set of axes."""
    return f"{axes['vlm_model']} · reasoning {axes['reasoning']} · {SETUP_TEXT[GtSetup.parse(axes['gt_setup'])]}"


#: Arms that fix both endpoint states unless their row records a setup.
BOTH_ENDS_BY_DEFAULT = ("ROSAME_MILP_24",)


def display_name(label: str, setup: GtSetup, row_setup: Optional[str] = None) -> str:
    """The legend name of a row label in an experiment run under ``setup``.

    ``row_setup`` is the ``gt_setup`` the row itself records, when it does. The
    label is returned unchanged when it names no known arm.
    """
    arm, *knobs = label.split("__")
    if arm in SETUP_ARMS:
        if row_setup is not None:
            setup = GtSetup.parse(row_setup)
        elif arm in BOTH_ENDS_BY_DEFAULT:
            setup = GtSetup.INIT_FINAL
        return f"{SETUP_ARMS[arm]} ({SETUP_TEXT[setup]})"
    if arm in PIXEL_ARMS:
        resolution = ", native res" if "res=native" in knobs else ""
        return PIXEL_ARMS[arm].replace(")", f"{resolution})")
    return label


def _labels(experiment: Path) -> Dict[str, Optional[str]]:
    """``{row label: the gt_setup its rows record, or None}`` for one experiment."""
    labels: Dict[str, Optional[str]] = {}
    for result in experiment.glob("testing/*/fold_result.json"):
        for row in read_json_or_none(result) or []:
            recorded = (row.get("algorithm_specific") or {}).get("gt_setup")
            labels[row["algorithm"]] = labels.get(row["algorithm"]) or recorded
    return dict(sorted(labels.items()))


def image_variants(results_root: Path, family: str, domains: List[str]) -> List[dict]:
    """One dashboard variant per recorded set of axes among ``family``'s experiments."""
    by_key: Dict[str, dict] = {}
    for domain in domains:
        for experiment in sorted((results_root / domain).glob(f"{family}*")):
            axes = read_json_or_none(experiment / "evaluation_results" / EXPERIMENT_VARIANT_FILE)
            if not axes or corpus_family(experiment.name) != family:
                continue
            setup = GtSetup.parse(axes["gt_setup"])
            variant = by_key.setdefault(variant_key(axes), {
                "key": variant_key(axes), "label": variant_label(axes), "experiment_dir": {},
                "display_names": {}, "include_algorithms": [], "exclude_algorithms": []})
            variant["experiment_dir"][domain] = experiment.as_posix()
            hidden = set(data_variant_labels(experiment))
            for label, row_setup in _labels(experiment).items():
                if label in hidden:
                    variant["exclude_algorithms"].append(label)
                    continue
                variant["display_names"][label] = display_name(label, setup, row_setup)
                if label.split("__")[0] in SETUP_ARMS:
                    variant["include_algorithms"].append(label)
    for variant in by_key.values():
        for field in ("include_algorithms", "exclude_algorithms"):
            variant[field] = sorted(set(variant[field]))
    return [by_key[key] for key in sorted(by_key)]


def write_variant_files(results_root: Path, domains: List[str]) -> Dict[str, Path]:
    """Write each data size's variant list; return ``{size: file}``."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for size, family in FAMILIES.items():
        path = GENERATED_DIR / f"image_variants_{size}.yaml"
        path.write_text(yaml.safe_dump(image_variants(results_root, family, domains), sort_keys=False))
        written[size] = path
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=Path("benchmark/evaluation/cfm/dashboard_config.yaml"))
    parser.add_argument("--no-build", action="store_true", help="write the lists only")
    parser.add_argument("--embed", action="store_true", help="also write the self-contained page")
    parser.add_argument("--regen-plots", action="store_true", help="passed to build_dashboard")
    parser.add_argument("--refresh-stats", action="store_true", help="passed to build_dashboard")
    args = parser.parse_args()

    dashboard = yaml.safe_load(args.config.read_text())["results_dashboard"]
    results_root = Path(dashboard.get("results_root", "benchmark/running_results"))
    for size, path in write_variant_files(results_root, dashboard["domains"]).items():
        print(f"{size}: {len(yaml.safe_load(path.read_text()) or [])} image variants -> {path}")
    if args.no_build:
        return 0
    command = [sys.executable, "-m", "benchmark.evaluation.cfm.build_dashboard", "--config", str(args.config)]
    command += ["--regen-plots"] * args.regen_plots + ["--refresh-stats"] * args.refresh_stats
    code = subprocess.call(command)
    if code == 0 and args.embed:
        code = subprocess.call(command + ["--embed"])
    return code


if __name__ == "__main__":
    sys.exit(main())
