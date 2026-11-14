"""The recorded axes of a classified corpus and of an experiment on it.

    python -m pytest src/utils/test_corpus_variant.py
"""

from pathlib import Path

import pytest

from src.utils.atomic_json import read_json_or_none
from src.utils.corpus_variant import (
    EXPERIMENT_VARIANT_FILE,
    CorpusVariant,
    ExampleProtocol,
    experiment_variant,
    read_corpus_variant,
    reasoning_of,
    write_corpus_variant,
    write_experiment_variant,
)
from src.utils.gt_setup import GtSetup

GPT = CorpusVariant("gpt-5.2", "off", ExampleProtocol.SHARED)


@pytest.mark.parametrize("extras, expected", [
    ({}, "off"),
    ({"reasoning_effort": "medium"}, "medium"),
    ({"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}, "on"),
    ({"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}, "off"),
    ({"reasoning_effort": "medium", "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}, "on"),
    ({"reasoning_effort": "medium", "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}, "off"),
])
def test_reasoning_of_a_vendor_blocks_extras(extras: dict, expected: str) -> None:
    assert reasoning_of(extras) == expected


def test_a_vendor_block_gives_the_variant(tmp_path: Path) -> None:
    block = {"fluent_classification_model": {"model_name": "qwen3.8-27b"},
             "request_extras": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}}
    assert CorpusVariant.from_vendor_block(block, tmp_path) == CorpusVariant(
        "qwen3.8-27b", "off", ExampleProtocol.SHARED)
    assert CorpusVariant.from_vendor_block(block, None).example_protocol is ExampleProtocol.OWN_FRAME0


def test_a_thinking_alias_names_the_model_without_its_suffix(tmp_path: Path) -> None:
    block = {"fluent_classification_model": {"model_name": "qwen3.8-27b-think"},
             "request_extras": {"reasoning_effort": "medium",
                                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}}
    assert CorpusVariant.from_vendor_block(block, tmp_path) == CorpusVariant(
        "qwen3.8-27b", "on", ExampleProtocol.SHARED)


def test_the_suffix_writes_every_axis() -> None:
    assert GPT.name_suffix == "__vlm=gpt-5.2__reason=off__fs=shared"


def test_a_variant_round_trips_through_its_file(tmp_path: Path) -> None:
    assert read_corpus_variant(tmp_path) is None
    write_corpus_variant(tmp_path, GPT)
    assert read_corpus_variant(tmp_path) == GPT
    write_corpus_variant(tmp_path, GPT)


def test_a_corpus_refuses_a_second_variant(tmp_path: Path) -> None:
    write_corpus_variant(tmp_path, GPT)
    with pytest.raises(ValueError, match="a corpus holds one variant"):
        write_corpus_variant(tmp_path, CorpusVariant("gpt-5.2", "medium", ExampleProtocol.SHARED))


def test_an_experiment_adds_the_setup(tmp_path: Path) -> None:
    corpus, results = tmp_path / "corpus", tmp_path / "results"
    corpus.mkdir(); results.mkdir()
    assert experiment_variant(corpus, GtSetup.INIT) is None
    write_experiment_variant(results, corpus, GtSetup.INIT)
    assert not (results / EXPERIMENT_VARIANT_FILE).exists()

    write_corpus_variant(corpus, GPT)
    write_experiment_variant(results, corpus, GtSetup.INIT_FINAL)
    assert read_json_or_none(results / EXPERIMENT_VARIANT_FILE) == {
        "vlm_model": "gpt-5.2", "reasoning": "off", "example_protocol": "shared", "gt_setup": "init_final"}
