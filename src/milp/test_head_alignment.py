"""Tests for :mod:`src.milp.head_alignment`, on all five benchmark domains.

    python -m pytest src/milp/test_head_alignment.py

``test_rosame_argument_order.py`` pins that the two groundings *disagree*. This
file pins that :mod:`~src.milp.head_alignment` reconciles them, and — through
:class:`TestNameOnlyAlignmentIsCaught` — that a build without the permutation
step fails on the four domains that reorder. That mutation check is the point of
the file: blocksworld's permutation is the identity and would pass a broken
implementation, and every other gate in this package is a blocksworld gate.

The DL side is the real vendored ``Domain_Model``, not a reimplementation of its
ordering rule, so a change to ``Predicate.ground`` surfaces here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pytest

import src.milp  # noqa: F401  (vendor sys.path bootstrap)
from src.milp import head_alignment
from src.milp.domain_assets import (
    BENCH_DOMAINS,
    build_domain,
    write_grounding_assets,
)
from src.milp.head_alignment import (
    action_head_indices,
    head_key,
    invert,
    pddl_argument_order,
    proposition_head_indices,
)

from dl.util.ROSAME.rosame import get_domain_model
from planning_structs.instance import Instance as PSInstance

_OBJECTS_PER_TYPE = 2


def _leaf_types(domain_key: str):
    """Types with no declared child; objects of a parent type would not ground."""
    types = list(build_domain(domain_key).types_set)
    parents = {t.parent.name for t in types if t.parent}
    leaves = [t for t in types if t.name not in parents and t.name != "object"]
    return leaves or [t for t in types if t.name != "object"] or types


def _objects(domain_key: str) -> Dict[str, List[str]]:
    return {
        one_type.name: [f"{one_type.name}{i}" for i in range(1, _OBJECTS_PER_TYPE + 1)]
        for one_type in _leaf_types(domain_key)
    }


def _both_groundings(domain_key: str, root: Path):
    """``(cp_instance, dl_domain_model)`` over one shared object universe."""
    objects = _objects(domain_key)
    instance = PSInstance(
        build_domain(domain_key),
        [(name, type_name) for type_name, names in objects.items() for name in names],
    )
    write_grounding_assets(domain_key, objects, root)
    return instance, get_domain_model(domain_key, str(root))


# ── the key formula ─────────────────────────────────────────────────────


class _FakeType:
    def __init__(self, name: str) -> None:
        self.name = name


class TestHeadKey:
    def test_a_signature_already_sorted_is_left_alone(self) -> None:
        types = [_FakeType("crate"), _FakeType("truck")]
        assert head_key("in", ["c1", "t1"], types) == "in c1 t1"

    def test_an_unsorted_signature_is_permuted(self) -> None:
        types = [_FakeType("tile"), _FakeType("position")]
        assert head_key("at", ["t1", "p1"], types) == "at p1 t1"

    def test_a_repeated_type_keeps_its_pddl_order(self) -> None:
        """hanoi's ``move_peg_disc(disc, peg, disc)`` — slots 0 and 2 stay ordered."""
        types = [_FakeType("disc"), _FakeType("peg"), _FakeType("disc")]
        assert head_key("move", ["d1", "p1", "d2"], types) == "move d1 d2 p1"

    def test_a_nullary_predicate_is_its_bare_name(self) -> None:
        assert head_key("handempty", [], []) == "handempty"

    def test_an_arity_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="takes 2 arguments, got 1"):
            head_key("at", ["t1"], [_FakeType("tile"), _FakeType("position")])


class TestPddlArgumentOrder:
    @pytest.mark.parametrize(
        "type_names,args",
        [
            (["tile", "position"], ["t1", "p1"]),
            (["disc", "peg", "disc"], ["d1", "p1", "d2"]),
            (["crate", "truck"], ["c1", "t1"]),
        ],
    )
    def test_it_undoes_head_key(self, type_names, args) -> None:
        types = [_FakeType(n) for n in type_names]
        head_args = head_key("p", args, types).split()[1:]
        assert pddl_argument_order(head_args, types) == args

    def test_an_arity_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="takes 2 arguments, got 3"):
            pddl_argument_order(["a", "b", "c"], [_FakeType("x"), _FakeType("y")])


# ── the alignment, against the real vendored head ───────────────────────


@pytest.mark.parametrize("domain_key", BENCH_DOMAINS)
class TestAlignmentAgainstTheVendoredHead:
    def test_propositions_align_bijectively(self, domain_key, tmp_path) -> None:
        instance, model = _both_groundings(domain_key, tmp_path)
        aligned = proposition_head_indices(instance, model)

        assert len(aligned) == len(instance.propositions) == len(model.propositions)
        assert sorted(aligned) == list(range(len(aligned)))

    def test_actions_align_bijectively(self, domain_key, tmp_path) -> None:
        instance, model = _both_groundings(domain_key, tmp_path)
        aligned = action_head_indices(instance, model)

        assert len(aligned) == len(instance.actions) == len(model.actions)
        assert sorted(aligned) == list(range(len(aligned)))

    def test_the_aligned_head_key_names_the_same_objects(
        self, domain_key, tmp_path
    ) -> None:
        """Bijectivity alone would be satisfied by any permutation; this pins
        that each CP proposition lands on the head column holding its objects."""
        instance, model = _both_groundings(domain_key, tmp_path)
        by_index = {index: key for key, index in model.propositions.items()}

        for cp_index, head_index in enumerate(
            proposition_head_indices(instance, model)
        ):
            proposition = instance.propositions[cp_index]
            name, *head_args = by_index[head_index].split()
            assert name == proposition.predicate.name
            assert pddl_argument_order(head_args, proposition.predicate.types) == [
                obj.name for obj in proposition.args
            ]

    def test_inversion_round_trips(self, domain_key, tmp_path) -> None:
        instance, model = _both_groundings(domain_key, tmp_path)
        aligned = proposition_head_indices(instance, model)
        assert [aligned[cp] for cp in invert(aligned)] == list(range(len(aligned)))


# ── the mutation check ──────────────────────────────────────────────────

_REORDERING_DOMAINS = ["depot", "gripper", "hanoi", "npuzzle"]


@pytest.fixture
def identity_permutation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the permutation step — i.e. align by predicate name alone."""
    monkeypatch.setattr(
        head_alignment,
        "rosame_argument_permutation",
        lambda types: list(range(len(types))),
    )


class TestNameOnlyAlignmentIsCaught:
    """Without the permutation the build must fail, not quietly mis-associate."""

    @pytest.mark.parametrize("domain_key", _REORDERING_DOMAINS)
    def test_it_raises_on_every_domain_that_reorders(
        self, domain_key, tmp_path, identity_permutation
    ) -> None:
        instance, model = _both_groundings(domain_key, tmp_path)
        with pytest.raises(KeyError):
            action_head_indices(instance, model)

    @pytest.mark.parametrize("domain_key", ["depot", "hanoi", "npuzzle"])
    def test_propositions_too_where_predicates_reorder(
        self, domain_key, tmp_path, identity_permutation
    ) -> None:
        instance, model = _both_groundings(domain_key, tmp_path)
        with pytest.raises(KeyError):
            proposition_head_indices(instance, model)

    def test_gripper_reorders_only_its_schemas(
        self, tmp_path, identity_permutation
    ) -> None:
        """The mutation is invisible in gripper's predicates; the count table in
        ``test_rosame_argument_order.py`` attributes both its entries to schemas."""
        instance, model = _both_groundings("gripper", tmp_path)
        assert proposition_head_indices(instance, model) == list(
            range(len(instance.propositions))
        )

    def test_blocksworld_cannot_detect_the_mutation(
        self, tmp_path, identity_permutation
    ) -> None:
        """Pinned so the parametrisation above is never trimmed to blocksworld."""
        instance, model = _both_groundings("blocksworld", tmp_path)
        proposition_head_indices(instance, model)
        action_head_indices(instance, model)


# ── a real fold's object union ──────────────────────────────────────────

_REAL_CELLS = {
    "depot": Path(
        "benchmark/data/depot/multi_problem_04-07-2026T01:32:08"
        "__model=gpt-5.2__steps=100/training/trajectories"
    ),
    "blocksworld": Path(
        "benchmark/data/blocksworld/blocks_predefined_problems1-10_final-version"
        "/training/trajectories"
    ),
}


def _real_object_union(domain_key: str) -> Dict[str, List[str]]:
    """``{type: [object]}`` over every problem of a checked-in cell.

    Real problems name objects arbitrarily, so an argument order that is wrong
    but type-valid would produce a key that *exists*. The synthetic universe
    above names objects after their type and cannot reach that case.
    """
    from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser

    from src.utils.config import load_config

    pddl_domain = DomainParser(
        Path(load_config()["domains"][domain_key]["domain_file"]), partial_parsing=True
    ).parse_domain()

    by_type: Dict[str, List[str]] = {}
    for problem_dir in sorted(_REAL_CELLS[domain_key].iterdir()):
        problem_pddl = problem_dir / f"{problem_dir.name}.pddl"
        if not problem_pddl.exists():
            continue
        problem = ProblemParser(problem_pddl, pddl_domain).parse_problem()
        for name, obj in problem.objects.items():
            names = by_type.setdefault(obj.type.name, [])
            if name not in names:
                names.append(name)
    return by_type


@pytest.mark.parametrize("domain_key", sorted(_REAL_CELLS))
class TestOnARealObjectUnion:
    def test_both_groundings_align(self, domain_key, tmp_path) -> None:
        if not _REAL_CELLS[domain_key].exists():
            pytest.skip(f"{domain_key} image cell absent")

        by_type = _real_object_union(domain_key)
        instance = PSInstance(
            build_domain(domain_key),
            sorted((name, t) for t, names in by_type.items() for name in names),
        )
        write_grounding_assets(domain_key, by_type, tmp_path)
        model = get_domain_model(domain_key, str(tmp_path))

        propositions = proposition_head_indices(instance, model)
        actions = action_head_indices(instance, model)

        assert sorted(propositions) == list(range(len(instance.propositions)))
        assert sorted(actions) == list(range(len(instance.actions)))


# ── failure modes ───────────────────────────────────────────────────────


class _Model:
    def __init__(self, propositions, actions=None) -> None:
        self.propositions = propositions
        self.actions = actions or {}


class TestAlignmentRejectsMismatchedGroundings:
    def test_a_head_grounded_over_other_objects_raises(self, tmp_path) -> None:
        instance, model = _both_groundings("blocksworld", tmp_path)
        narrower = dict(list(model.propositions.items())[:-1])

        with pytest.raises(ValueError, match="both must be built over the same"):
            proposition_head_indices(instance, _Model(narrower))

    def test_a_same_sized_head_missing_one_key_raises(self, tmp_path) -> None:
        instance, model = _both_groundings("blocksworld", tmp_path)
        renamed = dict(model.propositions)
        victim = next(iter(renamed))
        renamed["not_a_proposition"] = renamed.pop(victim)

        with pytest.raises(KeyError, match="does not carry"):
            proposition_head_indices(instance, _Model(renamed))


class TestInvert:
    def test_a_permutation_inverts(self) -> None:
        assert invert([2, 0, 1]) == [1, 2, 0]

    def test_a_repeated_target_raises(self) -> None:
        with pytest.raises(ValueError, match="claimed twice"):
            invert([0, 0])

    def test_an_out_of_range_target_raises(self) -> None:
        with pytest.raises(ValueError, match="outside 0..1"):
            invert([0, 5])
