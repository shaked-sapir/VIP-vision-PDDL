"""Annotating and auditing the variant records of image corpora and experiments.

    python -m pytest benchmark/test_corpus_variants.py
"""

import json
from pathlib import Path

import pytest

from benchmark.corpus_variants import annotate, check, variant_from_name
from src.utils.corpus_layout import IMAGE_POOL_SUBPATH
from src.utils.corpus_variant import CorpusVariant, ExampleProtocol, read_corpus_variant

BASE = "multi__model=gpt-5.2__steps=100"


@pytest.mark.parametrize("name, expected", [
    (f"{BASE}__fs=shared", CorpusVariant("gpt-5.2", "off", ExampleProtocol.SHARED)),
    (f"{BASE}__fs=shared__reason=medium__setup=init_final", CorpusVariant("gpt-5.2", "medium", ExampleProtocol.SHARED)),
    (f"{BASE}__fs=shared__vlm=qwen3.8-27b-think", CorpusVariant("qwen3.8-27b", "on", ExampleProtocol.SHARED)),
    (BASE, CorpusVariant("gpt-5.2", "off", ExampleProtocol.OWN_FRAME0)),
])
def test_variant_from_a_folder_name(name: str, expected: CorpusVariant) -> None:
    assert variant_from_name(name) == expected


def test_a_name_without_a_model_needs_a_default() -> None:
    assert variant_from_name("blocks__fs=shared") is None
    assert variant_from_name("blocks__fs=shared", "gpt-5.2").vlm_model == "gpt-5.2"


def _tree(tmp_path: Path, corpus_name: str, labels) -> tuple:
    data, results = tmp_path / "data", tmp_path / "results"
    corpus = data / "gripper" / corpus_name
    folder = corpus / IMAGE_POOL_SUBPATH / "problem0"
    folder.mkdir(parents=True)
    (folder / "problem0.masking_info").write_text("")
    experiment = results / "gripper" / "exp"
    (experiment / "evaluation_results").mkdir(parents=True)
    (experiment / "evaluation_results" / "run_params.json").write_text(json.dumps({
        "data_source_type": "ImageDataSource", "gt_setup": "init_final",
        "data_dir": f"/elsewhere/benchmark/data/gripper/{corpus_name}"}))
    cell = experiment / "testing" / "fold0_numtrajs3_gtrate0"
    cell.mkdir(parents=True)
    (cell / "fold_result.json").write_text(json.dumps([{"algorithm": l} for l in labels]))
    return data, results, corpus, experiment


def test_annotate_records_corpus_and_experiment_and_check_passes(tmp_path: Path) -> None:
    data, results, corpus, experiment = _tree(tmp_path, f"{BASE}__fs=shared__reason=medium", ["PISAM"])
    assert len(check(data, results)["problems"]) == 2

    annotate(data, results)

    assert read_corpus_variant(corpus) == CorpusVariant("gpt-5.2", "medium", ExampleProtocol.SHARED)
    assert json.loads((experiment / "evaluation_results" / "experiment_variant.json").read_text()) == {
        "vlm_model": "gpt-5.2", "reasoning": "medium", "example_protocol": "shared", "gt_setup": "init_final"}
    assert check(data, results)["problems"] == []


def test_annotate_never_changes_an_existing_record(tmp_path: Path) -> None:
    data, results, corpus, _ = _tree(tmp_path, f"{BASE}__fs=shared", ["PISAM"])
    annotate(data, results)
    corpus.rename(corpus.parent / f"{BASE}__fs=shared__reason=medium")
    annotate(data, results)
    assert read_corpus_variant(corpus.parent / f"{BASE}__fs=shared__reason=medium").reasoning == "off"


def test_check_flags_a_row_label_that_names_a_data_variant(tmp_path: Path) -> None:
    data, results, _, _ = _tree(tmp_path, f"{BASE}__fs=shared", ["PISAM", "PISAM__reasoning=off", "ROSAME-I_24__res=native"])
    annotate(data, results)
    problems = check(data, results)["problems"]
    assert len(problems) == 1 and "PISAM__reasoning=off" in problems[0] and "res=native" not in problems[0]
