"""Rewriting a base image run config for a variant.

    python -m pytest benchmark/test_image_variant_config.py
"""

import argparse
from pathlib import Path

import pytest

from benchmark.image_variant_config import (
    RunVariant,
    add_variant_arguments,
    apply_variant,
    corpus_family,
    experiment_name,
    requested_variant,
    resolve_corpus,
)
from src.utils.atomic_json import write_json_atomic
from src.utils.corpus_variant import CorpusVariant, ExampleProtocol, write_corpus_variant
from src.utils.gt_setup import GtSetup

GPT_OFF = CorpusVariant("gpt-5.2", "off", ExampleProtocol.SHARED)
GPT_MED = CorpusVariant("gpt-5.2", "medium", ExampleProtocol.SHARED)


def _corpus(root: Path, name: str, variant: CorpusVariant) -> Path:
    corpus = root / "gripper" / name
    corpus.mkdir(parents=True)
    write_corpus_variant(corpus, variant)
    return corpus


@pytest.fixture
def data(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    _corpus(root, "multi__model=gpt-5.2__fs=shared", GPT_OFF)
    _corpus(root, "multi__model=gpt-5.2__fs=shared__setup=init_final", GPT_OFF)
    _corpus(root, "multi__model=gpt-5.2__fs=shared__reason=medium", GPT_MED)
    _corpus(root, "other__model=gpt-5.2__fs=shared__reason=medium__setup=init_final", GPT_MED)
    return root


def _config(data: Path) -> dict:
    return {"source": "image", "run_name": "TO=600__image-small__fs=shared",
            "shared": {"pisam_milp": {"gt_anchoring": "init_only"}},
            "domains": [{"domain_key": "gripper",
                         "data_dir": str(data / "gripper" / "multi__model=gpt-5.2__fs=shared")}]}


def test_a_family_is_the_name_without_axis_suffixes() -> None:
    assert corpus_family("multi__model=gpt-5.2__steps=100__fs=shared__reason=medium__setup=init_final") \
        == "multi__model=gpt-5.2__steps=100"
    assert corpus_family("TO=600__image-small__fs=shared__vlm=qwen3.8-27b-think") == "TO=600__image-small"


@pytest.mark.parametrize("variant, expected", [
    (RunVariant(GPT_OFF, GtSetup.INIT), "multi__model=gpt-5.2__fs=shared"),
    (RunVariant(GPT_OFF, GtSetup.INIT_FINAL), "multi__model=gpt-5.2__fs=shared__setup=init_final"),
    (RunVariant(GPT_MED, GtSetup.INIT), "multi__model=gpt-5.2__fs=shared__reason=medium"),
])
def test_resolve_corpus_reads_the_record_and_the_setup_suffix(data: Path, variant, expected: str) -> None:
    base = data / "gripper" / "multi__model=gpt-5.2__fs=shared"
    assert resolve_corpus(base, variant).name == expected


def test_a_variant_with_no_corpus_in_the_family_raises(data: Path) -> None:
    base = data / "gripper" / "multi__model=gpt-5.2__fs=shared"
    with pytest.raises(FileNotFoundError, match="classify or stage one first"):
        resolve_corpus(base, RunVariant(GPT_MED, GtSetup.INIT_FINAL))


def test_an_existing_experiment_keeps_its_name(tmp_path: Path) -> None:
    variant = RunVariant(GPT_OFF, GtSetup.INIT_FINAL)
    results = tmp_path / "results"
    existing = results / "gripper" / "TO=600__image-small__fs=shared__setup=init_final" / "evaluation_results"
    existing.mkdir(parents=True)
    write_json_atomic(existing / "experiment_variant.json", variant.as_dict())

    assert experiment_name("TO=600__image-small__fs=shared", variant, results, ["gripper"]) \
        == "TO=600__image-small__fs=shared__setup=init_final"
    assert experiment_name("TO=600__image-small__fs=shared", RunVariant(GPT_MED, GtSetup.FINAL), results, ["gripper"]) \
        == "TO=600__image-small__vlm=gpt-5.2__reason=medium__fs=shared__setup=final"


def test_apply_variant_rewrites_data_run_name_setup_and_anchoring(data: Path, tmp_path: Path) -> None:
    config = _config(data)

    out = apply_variant(config, RunVariant(GPT_OFF, GtSetup.INIT_FINAL), results_root=tmp_path / "none")

    assert out["domains"][0]["data_dir"].endswith("__fs=shared__setup=init_final")
    assert out["run_name"] == "TO=600__image-small__vlm=gpt-5.2__reason=off__fs=shared__setup=init_final"
    assert out["shared"]["gt_setup"] == "init_final"
    assert out["shared"]["pisam_milp"]["gt_anchoring"] == "all_gt_states"
    assert config["shared"]["pisam_milp"]["gt_anchoring"] == "init_only"


def test_apply_variant_refuses_a_simulated_config_and_an_unknown_domain(data: Path) -> None:
    with pytest.raises(ValueError, match="image run config only"):
        apply_variant({"source": "simulated"}, RunVariant(GPT_OFF, GtSetup.INIT))
    with pytest.raises(ValueError, match="not in the config"):
        apply_variant(_config(data), RunVariant(GPT_OFF, GtSetup.INIT), domains=["depot"])


def test_flags_default_to_the_base_variant_once_one_is_given() -> None:
    parser = argparse.ArgumentParser()
    add_variant_arguments(parser)
    assert requested_variant(parser.parse_args([])) is None
    assert requested_variant(parser.parse_args(["--gt-setup", "final"])) == RunVariant(GPT_OFF, GtSetup.FINAL)
    assert requested_variant(parser.parse_args(["--vlm", "qwen3.8-27b", "--reasoning", "on"])) == RunVariant(
        CorpusVariant("qwen3.8-27b", "on", ExampleProtocol.SHARED), GtSetup.INIT)
