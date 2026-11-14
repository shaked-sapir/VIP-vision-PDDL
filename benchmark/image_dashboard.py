"""Build an image dashboard page for one GT setup, with one or two classifier variants side by side.

    python -m benchmark.image_dashboard --gt-setup init_final \\
        --classifier vlm=gpt-5.2,reasoning=off --classifier vlm=gpt-5.2,reasoning=medium

The experiments are found by their ``experiment_variant.json``, not by name. The
page's second selector switches between the classifiers. The dashboard config
is written to ``benchmark/evaluation/cfm/generated/`` and built with
``benchmark.evaluation.cfm.build_dashboard``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from benchmark.image_variant_config import RunVariant, corpus_family
from src.utils.atomic_json import read_json_or_none
from src.utils.corpus_variant import (
    EXPERIMENT_VARIANT_FILE,
    REASONING_OFF,
    CorpusVariant,
    ExampleProtocol,
)
from src.utils.gt_setup import GtSetup

TEMPLATE = Path("benchmark/evaluation/cfm/dashboard_config_image_fs_shared.yaml")
GENERATED_DIR = Path("benchmark/evaluation/cfm/generated")
#: The dashboard's two data-size slots, reused for the classifiers.
SLOTS = ("small", "large")

SETUP_TEXT = {
    GtSetup.NONE: "no GT", GtSetup.INIT: "init GT",
    GtSetup.FINAL: "final GT", GtSetup.INIT_FINAL: "init+final GT",
}
#: Arms whose ground truth is the experiment's setup, keyed by row-label prefix.
SETUP_ARMS = {
    "PISAM_MILP_LOOP": "NPI-SAM", "PISAM": "PI-SAM",
    "ROSAME_MILP_24": "ROSAME+MILP", "ROSAME_24": "ROSAME",
}


def parse_classifier(text: str) -> CorpusVariant:
    """A classifier from ``vlm=<model>,reasoning=<r>[,fs=<protocol>]``.

    Raises:
        ValueError: on an unknown key or a missing ``vlm``.
    """
    fields = dict(part.split("=", 1) for part in text.split(","))
    unknown = set(fields) - {"vlm", "reasoning", "fs"}
    if unknown or "vlm" not in fields:
        raise ValueError(f"--classifier expects vlm=<model>[,reasoning=<r>][,fs=<protocol>]; got {text!r}")
    return CorpusVariant(
        vlm_model=fields["vlm"],
        reasoning=fields.get("reasoning", REASONING_OFF),
        example_protocol=ExampleProtocol(fields.get("fs", ExampleProtocol.SHARED.value)),
    )


def classifier_label(variant: CorpusVariant) -> str:
    """The selector text of a classifier."""
    return f"{variant.vlm_model}, reasoning {variant.reasoning}"


def find_experiments(results_root: Path, family: str, variant: RunVariant, domains: List[str]) -> Dict[str, str]:
    """``{domain: experiment dir}`` for the experiments of ``family`` recorded as ``variant``."""
    found: Dict[str, str] = {}
    for domain in domains:
        for candidate in sorted((results_root / domain).glob(f"{family}*")):
            recorded = read_json_or_none(candidate / "evaluation_results" / EXPERIMENT_VARIANT_FILE)
            if corpus_family(candidate.name) == family and recorded == variant.as_dict():
                found[domain] = candidate.as_posix()
    return found


def setup_display_names(labels: List[str], setup: GtSetup) -> Dict[str, str]:
    """Display names of the arms whose ground truth is the experiment's setup."""
    names: Dict[str, str] = {}
    for label in labels:
        arm = label.split("__")[0]
        if arm in SETUP_ARMS:
            names[label] = f"{SETUP_ARMS[arm]} ({SETUP_TEXT[setup]})"
    return names


def _labels(experiment_dirs: List[str]) -> List[str]:
    labels = set()
    for directory in experiment_dirs:
        for result in Path(directory).glob("testing/*/fold_result.json"):
            labels.update(row["algorithm"] for row in read_json_or_none(result) or [])
    return sorted(labels)


def build_config(
    template: dict, setup: GtSetup, classifiers: List[CorpusVariant],
    family: str, results_root: Path, name: str,
) -> dict:
    """The dashboard config of ``setup`` with each classifier in a selector slot.

    Raises:
        ValueError: if more classifiers are given than the page has slots.
        FileNotFoundError: if a classifier has no experiment under ``setup``.
    """
    if not 1 <= len(classifiers) <= len(SLOTS):
        raise ValueError(f"a page shows 1 to {len(SLOTS)} classifiers; got {len(classifiers)}")
    config = yaml.safe_load(yaml.safe_dump(template))
    dashboard = config["results_dashboard"]
    dashboard["output_html"] = (results_root / f"results_dashboard_image__{name}.html").as_posix()
    selector = dashboard["selectors"]["data_size"]
    selector["label"] = "Classifier"
    selector["options"] = {}
    dashboard["image"] = {}
    all_dirs: List[str] = []
    for slot, classifier in zip(SLOTS, classifiers):
        experiments = find_experiments(
            results_root, family, RunVariant(classifier, setup), dashboard["domains"])
        if not experiments:
            raise FileNotFoundError(
                f"no experiment of '{family}' is recorded as {RunVariant(classifier, setup).as_dict()}")
        dashboard["image"][slot] = {"experiment_dir": experiments}
        selector["options"][slot] = classifier_label(classifier)
        all_dirs.extend(experiments.values())
    for slot in SLOTS[len(classifiers):]:
        dashboard["image"][slot] = {"experiment_dir": {}}
        selector["options"][slot] = "(none)"
    dashboard.setdefault("display_names", {}).update(setup_display_names(_labels(all_dirs), setup))
    dashboard["selectors"]["default"] = {"mode": "image", "data_size": SLOTS[0], "view": "baselines"}
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gt-setup", required=True, choices=[s.value for s in GtSetup])
    parser.add_argument("--classifier", action="append", required=True,
                        help="vlm=<model>[,reasoning=<r>][,fs=<protocol>]; give one or two")
    parser.add_argument("--family", default="TO=600__image-small", help="experiment-name family")
    parser.add_argument("--results-root", type=Path, default=Path("benchmark/running_results"))
    parser.add_argument("--name", default=None, help="page name; default setup=<s>")
    parser.add_argument("--embed", action="store_true", help="also write the self-contained page")
    args = parser.parse_args()

    setup = GtSetup.parse(args.gt_setup)
    name = args.name or f"setup={setup.value}"
    config = build_config(
        yaml.safe_load(TEMPLATE.read_text()), setup, [parse_classifier(c) for c in args.classifier],
        args.family, args.results_root, name)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    config_path = GENERATED_DIR / f"dashboard_config_image__{name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    command = [sys.executable, "-m", "benchmark.evaluation.cfm.build_dashboard",
               "--config", str(config_path), "--regen-plots", "--refresh-stats"]
    code = subprocess.call(command)
    if code == 0 and args.embed:
        code = subprocess.call(command[:5] + ["--embed"])
    return code


if __name__ == "__main__":
    sys.exit(main())
