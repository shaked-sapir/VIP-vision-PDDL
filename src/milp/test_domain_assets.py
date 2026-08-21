"""Tests for the generated ``Convertor`` domain assets.

    python -m pytest src/milp/test_domain_assets.py

The assets under ``vendor/planning_structs/specs/`` and ``vendor/pddl/`` are
generated from ``src/domains/*.pddl`` by ``src.milp.domain_assets``. They are
checked in, so they can drift from their source. These tests fail when they do.

Covers:
  1. Both assets exist for all five domains and the spec loads.
  2. The checked-in spec equals one regenerated from the current domain PDDL.
  3. The checked-in reference PDDL is byte-identical to the project's.
  4. The spec vocabulary matches the parsed domain's predicates and schemas.
  5. Directories are named by bench key, so `Convertor`'s hardcoded `"blocks"`
     alias stays inert and the vendored file needs no edit.
"""

from __future__ import annotations

import pytest

import src.milp  # noqa: F401  (vendor sys.path bootstrap)
from src.milp.domain_assets import (
    BENCH_DOMAINS,
    build_domain,
    reference_pddl_path,
    source_pddl_path,
    spec_path,
)

from planning_structs.util import dump_domain_to_json, load_domain_from_json


@pytest.mark.parametrize("domain_key", BENCH_DOMAINS)
def test_assets_exist_and_spec_loads(domain_key: str) -> None:
    assert spec_path(domain_key).is_file()
    assert reference_pddl_path(domain_key).is_file()

    domain = load_domain_from_json(spec_path(domain_key))
    assert domain.predicates
    assert domain.action_schemas


@pytest.mark.parametrize("domain_key", BENCH_DOMAINS)
def test_spec_is_not_stale(domain_key: str) -> None:
    """Regenerating from the current domain PDDL must reproduce the checked-in spec."""
    regenerated = dump_domain_to_json(build_domain(domain_key))
    assert regenerated == spec_path(domain_key).read_text(encoding="utf-8"), (
        f"{domain_key}'s spec is stale; run `python -m src.milp.domain_assets {domain_key}`"
    )


@pytest.mark.parametrize("domain_key", BENCH_DOMAINS)
def test_reference_pddl_is_not_stale(domain_key: str) -> None:
    assert reference_pddl_path(domain_key).read_bytes() == source_pddl_path(domain_key).read_bytes(), (
        f"{domain_key}'s reference PDDL is stale; run `python -m src.milp.domain_assets {domain_key}`"
    )


@pytest.mark.parametrize("domain_key", BENCH_DOMAINS)
def test_spec_vocabulary_matches_the_parsed_domain(domain_key: str) -> None:
    spec = load_domain_from_json(spec_path(domain_key))
    parsed = build_domain(domain_key)

    assert {p.name for p in spec.predicates} == {p.name for p in parsed.predicates}
    assert {a.name for a in spec.action_schemas} == {a.name for a in parsed.action_schemas}
    for lhs, rhs in zip(spec.predicates, parsed.predicates):
        assert (lhs.name, [t.name for t in lhs.types]) == (rhs.name, [t.name for t in rhs.types])
    for lhs, rhs in zip(spec.action_schemas, parsed.action_schemas):
        assert (lhs.name, [t.name for t in lhs.types]) == (rhs.name, [t.name for t in rhs.types])


def test_asset_dirs_are_named_by_bench_key() -> None:
    """`Convertor` maps only "blocks"; every other name it receives it uses verbatim."""
    assert "blocksworld" in BENCH_DOMAINS
    assert not (spec_path("blocks").parent).exists(), (
        "a spec dir named 'blocks' would be shadowed by Convertor's alias"
    )
