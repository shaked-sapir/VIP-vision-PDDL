"""Tests for :mod:`patch_accounting`.

The premise under test is that a patch is a *toggle*, not an assignment:
``flip_fluent_in_state`` looks the fluent up under ``{fluent, negate(fluent)}``
and inverts whatever it finds, so the polarity in the patch string is inert.
:func:`test_opposite_polarities_are_one_toggle_each` pins that premise against
the real state helper, so this module's arithmetic stops being true the moment
the premise does.
"""
from __future__ import annotations

from pddl_plus_parser.models import GroundedPredicate, PDDLType, State

from src.plan_denoising.typings import FluentLevelPatch
from src.utils.pddl_state import flip_fluent_in_state
from src.plan_denoising.patch_accounting import (
    net_patch_count,
    net_patch_count_from_records,
    patch_key,
    record_key,
)


def patch(fluent: str, obs: int = 0, comp: int = 0, state_type: str = "next") -> FluentLevelPatch:
    return FluentLevelPatch(
        observation_index=obs, component_index=comp, state_type=state_type, fluent=fluent,
    )


def record(fluent: str, obs: int = 0, comp: int = 0, state_type: str = "next") -> dict:
    return {
        "observation_index": obs, "component_index": comp,
        "state_type": state_type, "fluent": fluent,
    }


class TestKeys:
    def test_polarity_does_not_change_the_key(self):
        assert patch_key(patch("(on a b)")) == patch_key(patch("(not (on a b))"))

    def test_different_state_of_the_same_fluent_is_a_different_key(self):
        assert patch_key(patch("(on a b)", comp=0)) != patch_key(patch("(on a b)", comp=1))
        assert patch_key(patch("(on a b)", state_type="prev")) != patch_key(patch("(on a b)"))
        assert patch_key(patch("(on a b)", obs=0)) != patch_key(patch("(on a b)", obs=1))

    def test_record_key_matches_patch_key(self):
        assert record_key(record("(on a b)")) == patch_key(patch("(on a b)"))

    def test_record_key_survives_stringified_indices(self):
        """A log round trip can hand back "0" where the patch held 0."""
        stringified = {"observation_index": "0", "component_index": "0",
                       "state_type": "next", "fluent": "(not (on a b))"}
        assert record_key(stringified) == patch_key(patch("(on a b)"))


class TestNetCount:
    def test_empty(self):
        assert net_patch_count([]) == 0

    def test_distinct_patches_all_count(self):
        patches = {patch("(on a b)"), patch("(clear a)"), patch("(on a b)", comp=1)}
        assert net_patch_count(patches) == 3

    def test_opposite_polarity_pair_cancels_to_zero(self):
        """The bug this module exists for: two set members, zero edits."""
        patches = {patch("(on a b)"), patch("(not (on a b))")}
        assert len(patches) == 2
        assert net_patch_count(patches) == 0

    def test_cancelling_pair_does_not_hide_its_neighbours(self):
        patches = {patch("(on a b)"), patch("(not (on a b))"), patch("(clear a)")}
        assert net_patch_count(patches) == 1

    def test_records_and_patches_agree(self):
        fluents = ["(on a b)", "(not (on a b))", "(clear a)"]
        assert net_patch_count_from_records([record(f) for f in fluents]) == \
            net_patch_count({patch(f) for f in fluents})

    def test_odd_repetition_still_counts_once(self):
        """A list (unlike a set) can repeat; three toggles is still one edit."""
        records = [record("(on a b)"), record("(not (on a b))"), record("(on a b)")]
        assert net_patch_count_from_records(records) == 1


class TestTogglePremise:
    """The claim that makes cancellation arithmetic legitimate."""

    @staticmethod
    def _single_predicate_state() -> State:
        block = PDDLType(name="block")
        predicate = GroundedPredicate(
            name="clear",
            signature={"?x": block},
            object_mapping={"?x": "a"},
            is_positive=True,
        )
        return State(predicates={"(clear ?x)": {predicate}}, fluents={})

    def test_opposite_polarities_are_one_toggle_each(self):
        """Both spellings hit the same predicate, so applying both restores it."""
        state = self._single_predicate_state()
        original = {p.is_positive for p in next(iter(state.state_predicates.values()))}

        flip_fluent_in_state(state, "(clear a)")
        flipped = {p.is_positive for p in next(iter(state.state_predicates.values()))}
        assert flipped != original

        flip_fluent_in_state(state, "(not (clear a))")
        restored = {p.is_positive for p in next(iter(state.state_predicates.values()))}
        assert restored == original
