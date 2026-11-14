"""Run one image config under any variant: VLM model, reasoning, example protocol, GT setup.

A run config names each domain's base corpus. ``apply_variant`` rewrites it for
a variant: each ``data_dir`` becomes the corpus of that family classified under
the variant and prepared for the setup, ``run_name`` becomes that variant's
experiment, and ``shared`` gets the setup and the MILP anchoring that fixes its
states. ``benchmark_runner``, ``freeze_image_folds`` and
``scripts/cluster_image/make_image_manifests`` take the same four flags.
"""

from __future__ import annotations

import argparse
import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.atomic_json import read_json_or_none
from src.utils.corpus_variant import (
    EXPERIMENT_VARIANT_FILE,
    REASONING_OFF,
    CorpusVariant,
    ExampleProtocol,
    read_corpus_variant,
)
from src.utils.gt_setup import GtSetup

#: Name suffixes that spell a variant axis; what is left is the corpus family.
_AXIS_SUFFIX = re.compile(r"__(?:fs|reason|vlm|setup)=[^_]+(?:_[^_]+)*?(?=__|$)")
_SETUP_SUFFIX = re.compile(r"__setup=([a-z_]+?)(?=__|$)")

#: The ``pisam_milp.gt_anchoring`` value that fixes exactly a setup's states.
MILP_ANCHORING_VALUE = {
    GtSetup.NONE: "none",
    GtSetup.INIT: "init_only",
    GtSetup.FINAL: "gt_states_only",
    GtSetup.INIT_FINAL: "all_gt_states",
}


@dataclass(frozen=True)
class RunVariant:
    """The four axes one image run is identified by."""

    corpus: CorpusVariant
    gt_setup: GtSetup

    @property
    def name_suffix(self) -> str:
        """Every axis as a name suffix."""
        return f"{self.corpus.name_suffix}__setup={self.gt_setup.value}"

    def as_dict(self) -> Dict[str, str]:
        """The form ``experiment_variant.json`` stores."""
        return {**self.corpus.as_dict(), "gt_setup": self.gt_setup.value}


def add_variant_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--vlm``, ``--reasoning``, ``--example-protocol`` and ``--gt-setup``."""
    group = parser.add_argument_group(
        "image variant", "run an image config under another variant; all four default "
        "to the base variant once any one is given")
    group.add_argument("--vlm", default=None, help="VLM model, e.g. gpt-5.2 or qwen3.8-27b")
    group.add_argument("--reasoning", default=None, help="off, on, or a reasoning effort")
    group.add_argument("--example-protocol", default=None,
                       choices=[p.value for p in ExampleProtocol])
    group.add_argument("--gt-setup", default=None, choices=[s.value for s in GtSetup])


def requested_variant(args: argparse.Namespace, base_model: str = "gpt-5.2") -> Optional[RunVariant]:
    """The variant the flags ask for, or ``None`` when none of them was given."""
    given = (args.vlm, args.reasoning, args.example_protocol, args.gt_setup)
    if all(value is None for value in given):
        return None
    return RunVariant(
        corpus=CorpusVariant(
            vlm_model=args.vlm or base_model,
            reasoning=args.reasoning or REASONING_OFF,
            example_protocol=ExampleProtocol(args.example_protocol or ExampleProtocol.SHARED.value),
        ),
        gt_setup=GtSetup.parse(args.gt_setup),
    )


def corpus_family(name: str) -> str:
    """A corpus folder name without its variant suffixes."""
    return _AXIS_SUFFIX.sub("", name)


def _prepared_setup(name: str) -> GtSetup:
    match = _SETUP_SUFFIX.search(name)
    return GtSetup.parse(match.group(1) if match else None)


def resolve_corpus(base: Path, variant: RunVariant) -> Path:
    """The corpus of ``base``'s family classified under the variant and prepared for its setup.

    Raises:
        FileNotFoundError: if no sibling of ``base`` matches.
        ValueError: if more than one does.
    """
    family = corpus_family(base.name)
    matches = [
        sibling for sibling in sorted(base.parent.iterdir())
        if sibling.is_dir() and corpus_family(sibling.name) == family
        and read_corpus_variant(sibling) == variant.corpus
        and _prepared_setup(sibling.name) is variant.gt_setup
    ]
    if not matches:
        raise FileNotFoundError(
            f"no corpus of family '{family}' under {base.parent} is recorded as "
            f"{variant.as_dict()}; classify or stage one first"
        )
    if len(matches) > 1:
        raise ValueError(f"{len(matches)} corpora match {variant.as_dict()}: {[m.name for m in matches]}")
    return matches[0]


def experiment_name(run_name: str, variant: RunVariant, results_root: Path, domains: List[str]) -> str:
    """The experiment the variant's rows live in.

    An existing experiment of ``run_name``'s family recorded with the same four
    axes keeps its name; otherwise the name is the family plus every axis.

    Raises:
        ValueError: if the domains' existing experiments disagree on the name.
    """
    family = corpus_family(run_name)
    existing = set()
    for domain in domains:
        for candidate in sorted((results_root / domain).glob(f"{family}*")):
            recorded = read_json_or_none(candidate / "evaluation_results" / EXPERIMENT_VARIANT_FILE)
            if corpus_family(candidate.name) == family and recorded == variant.as_dict():
                existing.add(candidate.name)
    if len(existing) > 1:
        raise ValueError(f"experiments {sorted(existing)} all record {variant.as_dict()}")
    return existing.pop() if existing else f"{family}{variant.name_suffix}"


def apply_variant(
    config: Dict[str, Any], variant: RunVariant,
    results_root: Path = Path("benchmark/running_results"),
    domains: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """A copy of an image run config rewritten for ``variant``.

    Args:
        domains: Keep these domain keys only; ``None`` keeps every domain.

    Raises:
        ValueError: if the config's source is not ``image``, or a named domain
            is not in the config.
        FileNotFoundError: if a kept domain has no corpus for the variant.
    """
    if config.get("source") != "image":
        raise ValueError("a variant applies to an image run config only")
    rewritten = copy.deepcopy(config)
    if domains:
        unknown = set(domains) - {d["domain_key"] for d in rewritten["domains"]}
        if unknown:
            raise ValueError(f"domains not in the config: {sorted(unknown)}")
        rewritten["domains"] = [d for d in rewritten["domains"] if d["domain_key"] in domains]
    for domain in rewritten["domains"]:
        domain["data_dir"] = resolve_corpus(Path(domain["data_dir"]), variant).as_posix()
    rewritten["run_name"] = experiment_name(
        config["run_name"], variant, results_root, [d["domain_key"] for d in rewritten["domains"]])
    shared = rewritten.setdefault("shared", {})
    shared["gt_setup"] = variant.gt_setup.value
    if "pisam_milp" in shared:
        shared["pisam_milp"]["gt_anchoring"] = MILP_ANCHORING_VALUE[variant.gt_setup]
    return rewritten
