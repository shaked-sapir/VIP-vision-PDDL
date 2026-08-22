"""Tests for :mod:`src.milp.schema_row_alignment`, on all five benchmark domains.

    python -m pytest src/milp/test_schema_row_alignment.py

The gate is :class:`TestBijection`: the head's rows must land on the CP domain's
own bindings, one-to-one and onto. :class:`TestEachLayerIsNeeded` is the
non-vacuity check — it rebuilds the mapping with each of the three orderings left
un-undone and asserts the bijection *breaks*, so a green suite cannot mean "the
orderings happened to agree". blocksworld is the domain that agrees anyway, and
it is asserted to survive every mutation, which is why a blocksworld-only test
would prove nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pytest

import src.milp  # noqa: F401  (vendor sys.path bootstrap)
from src.milp.domain_assets import (
    BENCH_DOMAINS,
    build_domain,
    rosame_argument_permutation,
    write_grounding_assets,
)
from src.milp.schema_row_alignment import (
    BindingRow,
    check_bijective,
    head_binding_rows,
    parameter_variables,
    schema_row_keys,
)

from dl.util.ROSAME.rosame import get_domain_model

#: Domains with at least one reordered *action schema* — layer 1 is needed here.
REORDERED_SCHEMAS = {"depot", "gripper", "hanoi", "npuzzle"}

#: Domains with at least one reordered *predicate* — layer 2 is needed here.
#: gripper is the instructive gap: 2 reordered schemas, 0 reordered predicates,
#: so a predicate-only check is vacuous there exactly as a blocksworld-only
#: check is vacuous everywhere.
REORDERED_PREDICATES = {"depot", "hanoi", "npuzzle"}

_OBJECTS = {
    "blocksworld": {"block": ["a", "b"]},
    "hanoi": {"disc": ["d1", "d2"], "peg": ["p1"]},
    "depot": {
        "truck": ["t1"], "depot": ["d1"], "crane": ["c1"],
        "package": ["p1"], "pile": ["pl1"],
    },
    "gripper": {"ball": ["b1"], "room": ["r1"], "gripper": ["g1"]},
    "npuzzle": {"tile": ["t1"], "position": ["p1", "p2"]},
}

# A second, larger universe: the mapping is over parameter positions, so it must
# not move when the object count does.
_LARGER = {
    "blocksworld": {"block": [f"b{i}" for i in range(1, 5)]},
    "hanoi": {"disc": [f"d{i}" for i in range(1, 5)], "peg": ["p1", "p2", "p3"]},
    "depot": {
        "truck": ["t1", "t2"], "depot": ["d1", "d2"], "crane": ["c1", "c2"],
        "package": ["p1", "p2", "p3"], "pile": ["pl1", "pl2"],
    },
    "gripper": {
        "ball": ["b1", "b2", "b3"], "room": ["r1", "r2"], "gripper": ["g1", "g2"],
    },
    "npuzzle": {
        "tile": ["t1", "t2", "t3"], "position": [f"p{i}" for i in range(1, 5)],
    },
}


def _head(domain_key: str, root: Path, objects=None):
    write_grounding_assets(domain_key, objects or _OBJECTS[domain_key], root)
    return get_domain_model(domain_key, str(root))


# ── the pieces ──────────────────────────────────────────────────────────


class TestParameterVariables:
    def test_positions_are_one_based_pddl_order(self) -> None:
        schema = build_domain("gripper").get_action_schema("pick")
        by_type, position = parameter_variables(schema)

        assert [t.name for t in schema.types] == ["ball", "room", "gripper"]
        assert position == {"v1": 1, "v2": 2, "v3": 3}
        assert by_type == {"ball": ["v1"], "room": ["v2"], "gripper": ["v3"]}

    def test_a_repeated_type_groups_in_pddl_order(self) -> None:
        """hanoi's ``move_peg_disc(disc, peg, disc)`` — slots 1 and 3."""
        schema = build_domain("hanoi").get_action_schema("move_peg_disc")
        by_type, _ = parameter_variables(schema)

        assert by_type["disc"] == ["v1", "v3"]
        assert by_type["peg"] == ["v2"]


# ── the gate ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("domain_key", BENCH_DOMAINS)
class TestBijection:
    def test_every_schema_aligns(self, domain_key, tmp_path) -> None:
        keys = schema_row_keys(_head(domain_key, tmp_path), build_domain(domain_key))
        assert set(keys) == {a.name for a in build_domain(domain_key).action_schemas}

    def test_rows_match_the_head_tensor_width(self, domain_key, tmp_path) -> None:
        model = _head(domain_key, tmp_path)
        keys = schema_row_keys(model, build_domain(domain_key))
        for schema in model.action_schemas:
            assert len(keys[schema.name]) == int(schema.randn.shape[0]), schema.name

    def test_rows_cover_the_cp_bindings_exactly(self, domain_key, tmp_path) -> None:
        model = _head(domain_key, tmp_path)
        ps_domain = build_domain(domain_key)
        for schema in model.action_schemas:
            rows = head_binding_rows(schema, ps_domain)
            ps_schema = ps_domain.get_action_schema(schema.name)
            expected = {
                (predicate.name, tuple(binding))
                for predicate in schema.predicates
                for binding in ps_domain.predicate_arguments[
                    (ps_schema, ps_domain.get_predicate(predicate.name))
                ]
            }
            assert set(rows) == expected, schema.name
            assert len(set(rows)) == len(rows), schema.name

    def test_the_alignment_does_not_depend_on_the_object_universe(
        self, domain_key, tmp_path
    ) -> None:
        """``predicate_arguments`` is keyed on parameter positions, not objects."""
        small = schema_row_keys(
            _head(domain_key, tmp_path / "s"), build_domain(domain_key)
        )
        large = schema_row_keys(
            _head(domain_key, tmp_path / "l", _LARGER[domain_key]),
            build_domain(domain_key),
        )
        assert small == large


# ── non-vacuity ─────────────────────────────────────────────────────────


def _rows_without_predicate_reordering(
    head_schema: object, ps_domain
) -> List[BindingRow]:
    """The mapping with layer 2 left un-undone — i.e. ``binding_table``'s rule.

    This is exactly what the ICAPS-24 arm's bridge computes, and it is correct
    *there* because AMLGym's fork disables the predicate sort.
    """
    ps_schema = ps_domain.get_action_schema(head_schema.name)
    by_type, position = parameter_variables(ps_schema)
    variables = {t: by_type[t.name] for t in head_schema.params_types}
    return [
        (predicate.name, tuple(position[v] for v in proposition.split()[1:]))
        for predicate in head_schema.predicates
        for proposition in predicate.ground(variables)
    ]


def _rows_by_slot_number(head_schema: object, ps_domain) -> List[BindingRow]:
    """The mapping with layer 1 left un-undone — head slot numbers taken at face value."""
    variables: Dict[object, List[str]] = {}
    counter = 1
    for one_type in head_schema.params_types:
        count = head_schema.params[one_type]
        variables[one_type] = [str(counter + i) for i in range(count)]
        counter += count
    return [
        (predicate.name, tuple(int(v) for v in proposition.split()[1:]))
        for predicate in head_schema.predicates
        for proposition in predicate.ground(variables)
    ]


@pytest.mark.parametrize("domain_key", BENCH_DOMAINS)
class TestEachLayerIsNeeded:
    """Drop a layer and the bijection must break — on exactly the right domains.

    The two layers are needed on *different* subsets, which is what makes this
    stronger than "it breaks somewhere": layer 1 on the four domains with a
    reordered schema, layer 2 on the three with a reordered predicate. gripper
    needs the first and not the second; blocksworld needs neither, which is why
    every other gate in this package being a blocksworld gate proves nothing.
    """

    def test_ignoring_the_predicate_sort_breaks_exactly_the_right_domains(
        self, domain_key, tmp_path
    ) -> None:
        model = _head(domain_key, tmp_path)
        ps_domain = build_domain(domain_key)
        broken = any(
            set(_rows_without_predicate_reordering(schema, ps_domain))
            != set(head_binding_rows(schema, ps_domain))
            for schema in model.action_schemas
        )
        assert broken == (domain_key in REORDERED_PREDICATES), domain_key

    def test_taking_head_slot_numbers_at_face_value_breaks_exactly_the_right_domains(
        self, domain_key, tmp_path
    ) -> None:
        model = _head(domain_key, tmp_path)
        ps_domain = build_domain(domain_key)
        broken = any(
            set(_rows_by_slot_number(schema, ps_domain))
            != set(head_binding_rows(schema, ps_domain))
            for schema in model.action_schemas
        )
        assert broken == (domain_key in REORDERED_SCHEMAS), domain_key

    def test_the_24_arms_rule_is_insufficient_for_the_26_head(
        self, domain_key, tmp_path
    ) -> None:
        """``model_bridge.binding_table``'s rule, applied to a sorted head.

        Correct for the ICAPS-24 arm, whose fork disables the sort; incomplete
        here. Pinned so "reuse the 24 bridge" cannot be revisited without
        meeting this.
        """
        model = _head(domain_key, tmp_path)
        ps_domain = build_domain(domain_key)
        wrong = sum(
            1
            for schema in model.action_schemas
            for naive, correct in zip(
                _rows_without_predicate_reordering(schema, ps_domain),
                head_binding_rows(schema, ps_domain),
            )
            if naive != correct
        )
        assert (wrong > 0) == (domain_key in REORDERED_PREDICATES), domain_key

    def test_identity_is_wrong_wherever_a_signature_reorders(
        self, domain_key, tmp_path
    ) -> None:
        """The concrete claim that corrects plan §0.1."""
        ps_domain = build_domain(domain_key)
        reorders = [
            a.name
            for a in ps_domain.action_schemas
            if rosame_argument_permutation(a.types) != list(range(len(a.types)))
        ]
        assert bool(reorders) == (domain_key in REORDERED_SCHEMAS), domain_key


class TestTheAlignmentRaisesRatherThanMislabels:
    def test_a_head_for_another_domain_raises(self, tmp_path) -> None:
        model = _head("gripper", tmp_path)
        with pytest.raises(KeyError):
            schema_row_keys(model, build_domain("blocksworld"))

    def test_a_width_mismatch_raises(self, tmp_path) -> None:
        model = _head("blocksworld", tmp_path)
        schema = model.action_schemas[0]
        rows = head_binding_rows(schema, build_domain("blocksworld"))
        with pytest.raises(ValueError, match="grounds 1 rows but its head emits"):
            check_bijective(rows[:1], schema, build_domain("blocksworld"))

    def test_a_repeated_row_raises(self, tmp_path) -> None:
        model = _head("blocksworld", tmp_path)
        schema = model.action_schemas[0]
        rows = head_binding_rows(schema, build_domain("blocksworld"))
        with pytest.raises(ValueError, match="not injective"):
            check_bijective(
                [rows[0]] * len(rows), schema, build_domain("blocksworld")
            )
