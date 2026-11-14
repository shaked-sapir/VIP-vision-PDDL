"""How an image corpus was classified, and the variant an experiment ran under.

A corpus is the output of one VLM, in one reasoning mode, under one example
protocol; ``corpus_variant.json`` in the corpus root records the three. An
experiment adds the ground-truth setup (``src/utils/gt_setup.py``);
``experiment_variant.json`` beside its ``run_params.json`` records all four.
Folder names may carry the same values as suffixes, and these files are what
code reads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.utils.atomic_json import read_json_or_none, write_json_atomic
from src.utils.gt_setup import GtSetup

CORPUS_VARIANT_FILE = "corpus_variant.json"
EXPERIMENT_VARIANT_FILE = "experiment_variant.json"

REASONING_OFF = "off"
REASONING_ON = "on"

#: Suffix of a served-model alias that names the reasoning mode, not the model.
THINKING_ALIAS_SUFFIX = "-think"


class ExampleProtocol(Enum):
    """Where the classifier's one-shot example comes from."""

    OWN_FRAME0 = "own_frame0"
    SHARED = "shared"


@dataclass(frozen=True)
class CorpusVariant:
    """The three axes a classified corpus is identified by.

    Attributes:
        vlm_model: The classifier's model name, as its vendor block names it.
        reasoning: ``"off"``, or the reasoning effort the model ran with.
        example_protocol: The one-shot example's source.
    """

    vlm_model: str
    reasoning: str
    example_protocol: ExampleProtocol

    @property
    def name_suffix(self) -> str:
        """The three axes as a name suffix, every axis written."""
        return (f"__vlm={self.vlm_model}__reason={self.reasoning}"
                f"__fs={self.example_protocol.value}")

    def as_dict(self) -> Dict[str, str]:
        """The JSON form."""
        return {**asdict(self), "example_protocol": self.example_protocol.value}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CorpusVariant":
        """A variant from its JSON form.

        Raises:
            KeyError: if an axis is missing.
            ValueError: on an unknown example protocol.
        """
        return cls(
            vlm_model=str(raw["vlm_model"]),
            reasoning=str(raw["reasoning"]),
            example_protocol=ExampleProtocol(raw["example_protocol"]),
        )

    @classmethod
    def from_vendor_block(
        cls, block: Mapping[str, Any], example_dir: Optional[Path]
    ) -> "CorpusVariant":
        """The variant a ``config.yaml`` vendor block classifies under."""
        return cls(
            vlm_model=str(block["fluent_classification_model"]["model_name"]).removesuffix(
                THINKING_ALIAS_SUFFIX),
            reasoning=reasoning_of(block.get("request_extras") or {}),
            example_protocol=ExampleProtocol.SHARED if example_dir else ExampleProtocol.OWN_FRAME0,
        )


def reasoning_of(request_extras: Mapping[str, Any]) -> str:
    """The reasoning mode a vendor block's ``request_extras`` ask for.

    A chat template's ``enable_thinking`` decides when present (``"on"`` /
    ``"off"``); otherwise ``reasoning_effort`` names the effort, and neither
    is ``"off"``.
    """
    template = (request_extras.get("extra_body") or {}).get("chat_template_kwargs") or {}
    if "enable_thinking" in template:
        return REASONING_ON if template["enable_thinking"] else REASONING_OFF
    return str(request_extras.get("reasoning_effort") or REASONING_OFF)


def read_corpus_variant(corpus: Path) -> Optional[CorpusVariant]:
    """The corpus's recorded variant, or ``None`` when it has none."""
    raw = read_json_or_none(Path(corpus) / CORPUS_VARIANT_FILE)
    return CorpusVariant.from_dict(raw) if raw else None


def write_corpus_variant(corpus: Path, variant: CorpusVariant) -> None:
    """Record ``variant`` in the corpus root.

    Raises:
        ValueError: if the corpus already records a different variant.
    """
    recorded = read_corpus_variant(corpus)
    if recorded is not None and recorded != variant:
        raise ValueError(
            f"{corpus} was classified as {recorded.as_dict()} and "
            f"{variant.as_dict()} was asked for; a corpus holds one variant"
        )
    write_json_atomic(Path(corpus) / CORPUS_VARIANT_FILE, variant.as_dict())


def experiment_variant(corpus: Path, gt_setup: GtSetup) -> Optional[Dict[str, str]]:
    """The four axes of an experiment on ``corpus``, or ``None`` for an unrecorded corpus."""
    variant = read_corpus_variant(corpus)
    if variant is None:
        return None
    return {**variant.as_dict(), "gt_setup": gt_setup.value}


def write_experiment_variant(evaluation_results_dir: Path, corpus: Path, gt_setup: GtSetup) -> None:
    """Record an experiment's four axes beside its run parameters, when the corpus has them."""
    axes = experiment_variant(corpus, gt_setup)
    if axes is not None:
        write_json_atomic(Path(evaluation_results_dir) / EXPERIMENT_VARIANT_FILE, axes)
