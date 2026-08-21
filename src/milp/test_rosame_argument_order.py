"""ICAPS-26 grounds predicate and schema arguments in *sorted type* order.

    python -m pytest src/milp/test_rosame_argument_order.py

``dl/util/ROSAME/rosame.py`` stores a signature as a ``{Type: count}`` map, not
as an ordered list. ``Predicate.__init__`` then does::

    self.params_types = sorted(params.keys(), key=lambda x: x.name)

and ``Predicate.ground`` emits one ``itertools.permutations`` group per key in
that order. So a PDDL signature that is not already grouped and name-sorted is
grounded as a **permutation of itself**, and the DL head's proposition list is
not positionally comparable to the CP proposition list built from the PDDL.

This is a divergence between the two upstreams, not a quirk of ours. AMLGym's
ICAPS-24 fork carries the same line commented out::

    # self.params_types = sorted(params.keys(), key=lambda x: x.name)
    self.params_types = list(params.keys())

so the 24 arms never see the reordering and the 26 arms always do. That is why
``benchmark/algorithm_adapters/test_check_predicate.py`` passes without a
permutation step and this file exists separately.

The reconciliation itself is :mod:`src.milp.head_alignment`, tested in
``test_head_alignment.py``; this file only pins the disagreement it exists to
absorb. A head aligned by predicate name alone mis-associates arguments on four
of our five domains.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import pytest

import src.milp  # noqa: F401  (vendor sys.path bootstrap)
from src.milp.domain_assets import (
    BENCH_DOMAINS,
    build_domain,
    rosame_argument_order,
    rosame_argument_permutation,
    type_counts,
    write_grounding_assets,
)

from dl.util.ROSAME.rosame import get_domain_model

# Measured, not asserted from memory: the number of predicates plus action
# schemas whose PDDL argument order differs from ROSAME's grounding order.
# blocksworld is 0, which is exactly why a blocksworld-only test would prove
# nothing — every other gate in this package uses blocksworld.
REORDERED_SIGNATURES: Dict[str, int] = {
    "blocksworld": 0,
    "depot": 10,
    "gripper": 2,
    "hanoi": 2,
    "npuzzle": 2,
}

# One object per type, named so its type is readable off the object name.
OBJECTS_BY_DOMAIN: Dict[str, Dict[str, List[str]]] = {
    "npuzzle": {"tile": ["tile1", "tile2"], "position": ["pos1", "pos2"]},
    "hanoi": {"disc": ["disc1", "disc2"], "peg": ["peg1", "peg2"]},
    "gripper": {
        "ball": ["ball1", "ball2"],
        "room": ["room1", "room2"],
        "gripper": ["grip1", "grip2"],
    },
}


def _signatures(domain_key: str):
    """``(kind, name, pddl_types)`` for every predicate and schema of a domain."""
    domain = build_domain(domain_key)
    for kind, items in (("predicate", domain.predicates), ("schema", domain.action_schemas)):
        for item in items:
            yield kind, item.name, [t.name for t in item.types]


@pytest.mark.parametrize("domain_key", BENCH_DOMAINS)
def test_reordered_signature_count_is_pinned(domain_key: str) -> None:
    """A domain PDDL edit that changes head alignment must fail here, loudly."""
    reordered = [
        f"{kind} {name}: {types} -> {rosame_argument_order(_types_of(domain_key, kind, name))}"
        for kind, name, types in _signatures(domain_key)
        if types != rosame_argument_order(_types_of(domain_key, kind, name))
    ]
    assert len(reordered) == REORDERED_SIGNATURES[domain_key], (
        f"{domain_key} now reorders {len(reordered)} signatures, pinned at "
        f"{REORDERED_SIGNATURES[domain_key]}; the Phase-2 head alignment must be "
        f"rechecked. Reordered: {reordered}"
    )


def _types_of(domain_key: str, kind: str, name: str) -> Sequence:
    domain = build_domain(domain_key)
    items = domain.predicates if kind == "predicate" else domain.action_schemas
    return next(item.types for item in items if item.name == name)


def test_the_reordering_is_real_on_four_of_five_domains() -> None:
    """Guards the table above against being quietly zeroed into vacuity."""
    affected = [key for key, count in REORDERED_SIGNATURES.items() if count > 0]
    assert sorted(affected) == ["depot", "gripper", "hanoi", "npuzzle"]
    assert REORDERED_SIGNATURES["blocksworld"] == 0


@pytest.mark.parametrize("domain_key", BENCH_DOMAINS)
def test_permutation_is_a_bijection_so_no_argument_identity_is_lost(
    domain_key: str,
) -> None:
    """The count-map collapse is recoverable: the mapping is onto every index."""
    for kind, name, types in _signatures(domain_key):
        permutation = rosame_argument_permutation(_types_of(domain_key, kind, name))
        assert sorted(permutation) == list(range(len(types))), (
            f"{domain_key} {kind} {name}: {permutation} is not a permutation"
        )


@pytest.mark.parametrize("domain_key", BENCH_DOMAINS)
def test_permutation_agrees_with_rosames_sorted_count_map(domain_key: str) -> None:
    """``rosame_argument_order`` must equal what ``sorted(params.keys())`` yields."""
    for kind, name, _ in _signatures(domain_key):
        types = _types_of(domain_key, kind, name)
        counts = type_counts(types)
        from_count_map = [n for n in sorted(counts) for _ in range(counts[n])]
        assert rosame_argument_order(types) == from_count_map


def test_hanoi_repeats_a_type_at_non_adjacent_positions() -> None:
    """``move_peg_disc(disc, peg, disc)`` — the case a naive pairing gets wrong.

    The stable sort keeps the two ``disc`` arguments in PDDL order, so slot 0
    is PDDL argument 0 and slot 1 is PDDL argument 2. ``ground`` permutes rather
    than combines, so the two slots are genuinely distinct and this is lossless.
    """
    types = _types_of("hanoi", "schema", "move_peg_disc")
    assert [t.name for t in types] == ["disc", "peg", "disc"]
    assert rosame_argument_permutation(types) == [0, 2, 1]
    assert rosame_argument_order(types) == ["disc", "disc", "peg"]


@pytest.mark.parametrize("domain_key", sorted(OBJECTS_BY_DOMAIN))
def test_the_vendored_grounder_emits_objects_in_permuted_order(
    domain_key: str, tmp_path
) -> None:
    """The end-to-end pin: check the real ``Domain_Model``, not our helper.

    Objects are named after their type, so the type order of a grounded
    proposition or action can be read straight off the emitted string. Both are
    covered: gripper reorders only in its schemas, hanoi and npuzzle in both.
    """
    objects = OBJECTS_BY_DOMAIN[domain_key]
    write_grounding_assets(domain_key, objects, tmp_path)
    model = get_domain_model(domain_key, str(tmp_path))

    type_of_object = {
        name: type_name for type_name, names in objects.items() for name in names
    }
    grounded: Dict[str, Dict[str, List[List[str]]]] = {"predicate": {}, "schema": {}}
    for kind, source in (("predicate", model.propositions), ("schema", model.actions)):
        for entry in source:
            head, *arguments = entry.split()
            grounded[kind].setdefault(head, []).append(
                [type_of_object[a] for a in arguments]
            )

    checked_reordered = 0
    for kind, name, pddl_types in _signatures(domain_key):
        if name not in grounded[kind]:
            continue
        expected = rosame_argument_order(_types_of(domain_key, kind, name))
        if pddl_types != expected:
            checked_reordered += 1
        for observed in grounded[kind][name]:
            assert observed == expected, (
                f"{domain_key} {kind} {name}: the vendored grounder emitted "
                f"{observed}, but the permutation predicts {expected}"
            )

    assert checked_reordered > 0, (
        f"{domain_key} grounded nothing whose argument order actually differs, so "
        "this run of the test confirms only that a no-op is a no-op"
    )


def test_npuzzle_at_grounds_position_before_tile(tmp_path) -> None:
    """A concrete, readable instance of the reordering, end to end."""
    objects = OBJECTS_BY_DOMAIN["npuzzle"]
    write_grounding_assets("npuzzle", objects, tmp_path)
    model = get_domain_model("npuzzle", str(tmp_path))

    assert [t.name for t in _types_of("npuzzle", "predicate", "at")] == ["tile", "position"]
    grounded = [p for p in model.propositions if p.split()[0] == "at"]
    assert grounded, "npuzzle's `at` predicate did not ground"
    for proposition in grounded:
        _, first, second = proposition.split()
        assert first.startswith("pos") and second.startswith("tile"), (
            f"expected `at <position> <tile>`, got {proposition!r}"
        )
