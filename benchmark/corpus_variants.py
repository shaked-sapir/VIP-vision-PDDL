"""Record and audit the variant axes of the image corpora and experiments.

    python -m benchmark.corpus_variants annotate   # write the missing records from folder names
    python -m benchmark.corpus_variants check      # list every corpus and experiment, flag problems

``annotate`` reads a folder name's suffixes (``__fs=shared``, ``__reason=<r>``,
``__vlm=<model>``, ``__setup=<s>``, and the ``model=<m>`` a corpus was first
generated with) and writes ``corpus_variant.json`` / ``experiment_variant.json``
where none exists; it never changes an existing record. ``check`` exits non-zero
when a corpus or experiment has no record, an experiment disagrees with its
corpus, or a row label names a data variant.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from src.utils.atomic_json import read_json_or_none, write_json_atomic
from src.utils.corpus_layout import IMAGE_POOL_NORMALIZED_SUBPATH, IMAGE_POOL_SUBPATH
from src.utils.corpus_variant import (
    CORPUS_VARIANT_FILE,
    EXPERIMENT_VARIANT_FILE,
    REASONING_OFF,
    REASONING_ON,
    THINKING_ALIAS_SUFFIX,
    CorpusVariant,
    ExampleProtocol,
    experiment_variant,
    read_corpus_variant,
    write_corpus_variant,
)
from src.utils.gt_setup import GtSetup

#: Row-label parts that name how the data was classified rather than the arm.
DATA_VARIANT_LABEL_PARTS = ("reasoning", "reason", "vlm", "fs", "setup")


def _suffix(name: str, key: str) -> Optional[str]:
    match = re.search(rf"__{key}=([^_]+(?:_[^_]+)*?)(?=__|$)", name)
    return match.group(1) if match else None


def variant_from_name(name: str, default_model: Optional[str] = None) -> Optional[CorpusVariant]:
    """The variant a corpus folder name spells.

    ``default_model`` stands in when the name names no model; without one such
    a name gives ``None``.
    """
    model = _suffix(name, "vlm") or _suffix(name, "model") or default_model
    if model is None:
        return None
    think = model.endswith(THINKING_ALIAS_SUFFIX)
    return CorpusVariant(
        vlm_model=model.removesuffix(THINKING_ALIAS_SUFFIX),
        reasoning=_suffix(name, "reason") or (REASONING_ON if think else REASONING_OFF),
        example_protocol=ExampleProtocol.SHARED if _suffix(name, "fs") == "shared"
        else ExampleProtocol.OWN_FRAME0,
    )


def name_clashes(name: str, variant: CorpusVariant) -> List[str]:
    """The axes a folder name spells differently from the recorded variant.

    An axis the name does not spell is not compared.
    """
    spelled = {
        "vlm_model": (_suffix(name, "vlm") or _suffix(name, "model") or "").removesuffix(THINKING_ALIAS_SUFFIX),
        "reasoning": _suffix(name, "reason") or "",
        "example_protocol": _suffix(name, "fs") or "",
    }
    recorded = variant.as_dict()
    return [f"its name spells {axis}={value}, its record {recorded[axis]}"
            for axis, value in spelled.items() if value and value != recorded[axis]]


def classified_corpora(data_root: Path) -> Iterator[Path]:
    """Every corpus under ``data_root`` whose image pool, or its normalized copy, holds an inferred trajectory."""
    for corpus in sorted(data_root.glob("*/*")):
        pools = (corpus / IMAGE_POOL_SUBPATH, corpus / IMAGE_POOL_NORMALIZED_SUBPATH)
        if any(pool.is_dir() and next(pool.glob("*/*.masking_info"), None) is not None for pool in pools):
            yield corpus


def image_experiments(results_root: Path) -> Iterator[Path]:
    """Every experiment under ``results_root`` whose run parameters name an image data source."""
    for params in sorted(results_root.glob("*/*/evaluation_results/run_params.json")):
        if (read_json_or_none(params) or {}).get("data_source_type") == "ImageDataSource":
            yield params.parent.parent


def _experiment_corpus(experiment: Path, data_root: Path) -> Optional[Path]:
    """The local corpus an experiment ran on, found by its recorded ``data_dir``'s last two parts."""
    recorded = (read_json_or_none(experiment / "evaluation_results" / "run_params.json") or {}).get("data_dir")
    if not recorded:
        return None
    local = data_root.joinpath(*Path(recorded).parts[-2:])
    return local if local.is_dir() else None


def _experiment_setup(experiment: Path) -> GtSetup:
    params = read_json_or_none(experiment / "evaluation_results" / "run_params.json") or {}
    return GtSetup.parse(params.get("gt_setup"))


def annotate(data_root: Path, results_root: Path, default_model: Optional[str] = None,
             only: str = "*") -> List[str]:
    """Write the missing records; return one line per record written or skipped.

    ``default_model`` is recorded for a corpus whose name names no model.
    ``only`` is a glob on the corpus folder name; other corpora are left alone.
    """
    lines: List[str] = []
    for corpus in classified_corpora(data_root):
        if read_corpus_variant(corpus) is not None or not fnmatch.fnmatch(corpus.name, only):
            continue
        variant = variant_from_name(corpus.name, default_model)
        if variant is None:
            lines.append(f"skipped  {corpus}: its name names no model")
            continue
        write_corpus_variant(corpus, variant)
        lines.append(f"recorded {corpus}: {variant.as_dict()}")
    for experiment in image_experiments(results_root):
        target = experiment / "evaluation_results" / EXPERIMENT_VARIANT_FILE
        corpus = _experiment_corpus(experiment, data_root)
        if target.exists() or corpus is None:
            continue
        axes = experiment_variant(corpus, _experiment_setup(experiment))
        if axes is not None:
            write_json_atomic(target, axes)
            lines.append(f"recorded {experiment}: {axes}")
    return lines


def data_variant_labels(experiment: Path) -> List[str]:
    """The row labels of an experiment that carry a data-variant part."""
    labels = set()
    for result in experiment.glob("testing/*/fold_result.json"):
        for row in read_json_or_none(result) or []:
            parts = {part.split("=")[0] for part in row["algorithm"].split("__")[1:]}
            if parts & set(DATA_VARIANT_LABEL_PARTS):
                labels.add(row["algorithm"])
    return sorted(labels)


def check(data_root: Path, results_root: Path) -> Dict[str, List[str]]:
    """The inventory lines and the problems found."""
    listing: List[str] = []
    problems: List[str] = []
    for corpus in classified_corpora(data_root):
        variant = read_corpus_variant(corpus)
        listing.append(f"corpus      {corpus.parent.name}/{corpus.name}: "
                       f"{json.dumps(variant.as_dict()) if variant else 'NO RECORD'}")
        if variant is None:
            problems.append(f"{corpus}: no {CORPUS_VARIANT_FILE}")
        else:
            problems.extend(f"{corpus}: {clash}" for clash in name_clashes(corpus.name, variant))
    for experiment in image_experiments(results_root):
        axes = read_json_or_none(experiment / "evaluation_results" / EXPERIMENT_VARIANT_FILE)
        listing.append(f"experiment  {experiment.parent.name}/{experiment.name}: "
                       f"{json.dumps(axes) if axes else 'NO RECORD'}")
        corpus = _experiment_corpus(experiment, data_root)
        if axes is None:
            problems.append(f"{experiment}: no {EXPERIMENT_VARIANT_FILE}"
                            + ("" if corpus else " (its corpus is not on this machine)"))
        elif corpus is not None and axes != experiment_variant(corpus, _experiment_setup(experiment)):
            problems.append(f"{experiment}: its record disagrees with its corpus's")
        labels = data_variant_labels(experiment)
        if labels:
            problems.append(f"{experiment}: row labels name a data variant: {labels}")
    return {"listing": listing, "problems": problems}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["annotate", "check"])
    parser.add_argument("--data-root", type=Path, default=Path("benchmark/data"))
    parser.add_argument("--results-root", type=Path, default=Path("benchmark/running_results"))
    parser.add_argument("--default-model", default=None,
                        help="annotate: the model of a corpus whose name names none")
    parser.add_argument("--only", default="*", help="annotate: glob on the corpus folder name")
    args = parser.parse_args()
    if args.command == "annotate":
        print("\n".join(annotate(args.data_root, args.results_root, args.default_model, args.only))
              or "nothing to record")
        return 0
    report = check(args.data_root, args.results_root)
    print("\n".join(report["listing"]))
    if report["problems"]:
        print(f"\n{len(report['problems'])} problem(s):", file=sys.stderr)
        print("\n".join(f"  {p}" for p in report["problems"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
