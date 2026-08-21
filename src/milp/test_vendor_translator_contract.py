"""Gate 3 (MILP side) for the vendored ``convertor/translator.py``.

    python -m pytest src/milp/test_vendor_translator_contract.py

Plan §9.3 lists four MILP-parity obligations. Two of them — that the §0.1
identity mappings are what reach ``extract_sol_*``, and that ``run_fixer``
agrees with a direct translator call — need the Phase-2 adapter and are not
here. The other two need no solver at all, because they are properties of the
translator's own arithmetic:

  * ``trans_full_state``'s ``zip`` is index-aligned with the DL symbol vector
    (§4.2) — and ``zip`` is the wrong tool for an alignment that matters,
    because it truncates in silence.
  * ``state_label`` / ``action_label`` are sized from the single scalar
    ``problem.max_t``, so a **ragged** bundle mis-sizes every trace but the one
    that set it (§6.1).

Plus the third silent-failure path §6.1 flags: ``extract_sol_label``'s
``chosen = 0`` fallback, which fabricates action 0 when no action variable is
true and is indistinguishable from action 0 being genuinely chosen.

These tests do not assert that the vendored behaviour is *correct*. It is not:
each one below documents a way the vendored code fails quietly. They pin the
behaviour so the adapter is written against what the code does, and so that the
guards §6.1 asks for can be shown to be load-bearing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pytest

torch = pytest.importorskip("torch")

import src.milp  # noqa: F401  (vendor sys.path bootstrap)

from convertor.translator import Translator
from planning_structs.domain import Domain as PSDomain
from planning_structs.instance import Instance as PSInstance


def _micro_world(n_rooms: int = 2):
    """move(?from ?to) / at(?r) over ``n_rooms`` rooms — one predicate, one schema."""
    domain = PSDomain(
        [("room", None)],
        [("at", ["room"])],
        [("move", ["room", "room"])],
        name="micro",
    )
    objects = [(f"r{i}", "room") for i in range(1, n_rooms + 1)]
    return domain, PSInstance(domain, objects)


@pytest.fixture()
def translator() -> Tuple[Translator, PSInstance]:
    _, instance = _micro_world()
    return Translator(mip_domain=None, rosame_domain=None, instance=instance), instance


# --------------------------------------------------------------------------
# trans_full_state — the §4.2 zip alignment
# --------------------------------------------------------------------------


def test_trans_full_state_selects_by_position_not_by_name(translator) -> None:
    """The DL vector carries no names; index *is* the only association."""
    trans, instance = translator
    state = torch.zeros(len(instance.propositions))
    state[0] = 1
    assert trans.trans_full_state(state) == [instance.propositions[0]]


def test_a_short_state_vector_is_truncated_in_silence(translator) -> None:
    """``zip`` stops at the shorter side, so missing symbols read as false.

    A DL head narrower than the CP proposition list therefore does not raise —
    it silently asserts that every proposition past its width is absent, which
    for the goal row is a fabricated anchor.
    """
    trans, instance = translator
    assert len(instance.propositions) > 1

    full = torch.ones(len(instance.propositions))
    assert trans.trans_full_state(full) == list(instance.propositions)

    short = torch.ones(1)
    assert trans.trans_full_state(short) == [instance.propositions[0]], (
        "a one-element state silently yields a one-proposition state, not an error"
    )


def test_a_long_state_vector_is_truncated_in_silence(translator) -> None:
    """The other direction: a wider DL head drops its tail without complaint."""
    trans, instance = translator
    long = torch.ones(len(instance.propositions) + 5)
    assert trans.trans_full_state(long) == list(instance.propositions)


def test_a_permuted_state_vector_is_silently_wrong_not_detectably_wrong(
    translator,
) -> None:
    """Why §4.2 matters, and the link to the sorted-type reordering.

    A permutation preserves length, so no truncation occurs and no shape error
    is raised — the result is simply the wrong set of propositions, at the right
    size. ``src/milp/test_rosame_argument_order.py`` shows the ICAPS-26 head
    really is permuted relative to the PDDL order on four of five domains, so
    this is the failure mode the Phase-2 head alignment exists to prevent.
    """
    trans, instance = translator
    state = torch.zeros(len(instance.propositions))
    state[0] = 1

    permuted = torch.zeros(len(instance.propositions))
    permuted[-1] = 1

    correct = trans.trans_full_state(state)
    wrong = trans.trans_full_state(permuted)
    assert len(correct) == len(wrong) == 1
    assert correct != wrong, "same size, different meaning, no error raised"


# --------------------------------------------------------------------------
# extract_sol_label — ragged max_t and the chosen = 0 fallback
# --------------------------------------------------------------------------


@dataclass
class _FakeSolvedProblem:
    """The duck type ``extract_sol_label`` reads: see ``Translator``'s docstring."""

    instance: PSInstance
    n_traces: int
    max_t: int
    true_holds: set = field(default_factory=set)
    true_acts: set = field(default_factory=set)

    @property
    def hol(self) -> Dict[Tuple[int, int, Any], Any]:
        return _Lookup(self.true_holds)

    @property
    def act(self) -> Dict[Tuple[int, int, Any], Any]:
        return _Lookup(self.true_acts)

    @staticmethod
    def _bool_value(var) -> bool:
        return bool(var)


class _Lookup:
    """A ``problem.hol``-alike: subscripting returns the truth of that key."""

    def __init__(self, true_keys: set) -> None:
        self._true_keys = true_keys

    def __getitem__(self, key) -> bool:
        return key in self._true_keys


def test_labels_are_sized_from_the_single_max_t_scalar(translator) -> None:
    """§6.1: ``max_t - 2`` rows of state and ``max_t - 1`` of action, per trace."""
    trans, instance = translator
    max_t = 5
    problem = _FakeSolvedProblem(instance=instance, n_traces=2, max_t=max_t)

    state_labels, action_labels = trans.extract_sol_label(problem, None, None)

    assert len(state_labels) == len(action_labels) == 2
    for state_label, action_label in zip(state_labels, action_labels):
        assert tuple(state_label.shape) == (max_t - 2, len(instance.propositions))
        assert tuple(action_label.shape) == (max_t - 1,)


def test_a_ragged_bundle_mis_sizes_every_trace_but_the_one_that_set_max_t(
    translator,
) -> None:
    """The §6.1 hazard, made concrete.

    ``max_t`` is one scalar for the whole bundle, so both traces get labels
    sized for the longest. ``loss_pseudo_s`` then compares a ``[max_t-2, S]``
    label against a ``[T_i, S]`` row of ``z``, which for the shorter trace is a
    shape error at training time rather than a solver error here. That is what
    §6.1's pad-and-mask decision exists to prevent.
    """
    trans, instance = translator
    longest, shorter = 6, 4
    problem = _FakeSolvedProblem(instance=instance, n_traces=2, max_t=longest)

    state_labels, _ = trans.extract_sol_label(problem, None, None)
    assert all(label.shape[0] == longest - 2 for label in state_labels)

    z_for_the_shorter_trace = torch.zeros(shorter - 2, len(instance.propositions))
    with pytest.raises(ValueError, match="target size"):
        torch.nn.functional.binary_cross_entropy(
            torch.sigmoid(z_for_the_shorter_trace), state_labels[1]
        )


def test_the_chosen_zero_fallback_fabricates_action_zero_in_silence(
    translator,
) -> None:
    """§6.1: no action true at a step is indistinguishable from action 0 chosen.

    Under option B the actions are hard-constrained, so this branch should be
    unreachable — which is exactly why the adapter must assert it, since if it
    ever fires the fabricated label is fed to ``loss_pseudo_a`` at full weight.
    """
    trans, instance = translator
    max_t = 4
    genuine = _FakeSolvedProblem(
        instance=instance,
        n_traces=1,
        max_t=max_t,
        true_acts={(1, t, instance.actions[0]) for t in range(1, max_t)},
    )
    nothing_true = _FakeSolvedProblem(instance=instance, n_traces=1, max_t=max_t)

    _, chosen_labels = trans.extract_sol_label(genuine, None, None)
    _, fallback_labels = trans.extract_sol_label(nothing_true, None, None)

    assert torch.equal(chosen_labels[0], fallback_labels[0]), (
        "the fallback is byte-identical to a genuine action-0 solution, so it "
        "cannot be detected downstream — the assertion has to be at extraction"
    )
    assert torch.equal(fallback_labels[0], torch.zeros(max_t - 1, dtype=torch.long))


def test_a_non_zero_action_is_recovered_so_the_fallback_test_is_not_vacuous(
    translator,
) -> None:
    """Guards the test above: extraction really can return something other than 0."""
    trans, instance = translator
    assert len(instance.actions) > 1
    max_t = 4
    problem = _FakeSolvedProblem(
        instance=instance,
        n_traces=1,
        max_t=max_t,
        true_acts={(1, t, instance.actions[1]) for t in range(1, max_t)},
    )
    _, action_labels = trans.extract_sol_label(problem, None, None)
    assert torch.equal(action_labels[0], torch.ones(max_t - 1, dtype=torch.long))


def test_holds_are_read_from_steps_two_to_max_t_exclusive(translator) -> None:
    """Init (t=1) and goal (t=max_t+1) are anchors, not label rows."""
    trans, instance = translator
    max_t = 5
    every_step = {
        (1, t, p) for t in range(0, max_t + 2) for p in instance.propositions
    }
    problem = _FakeSolvedProblem(
        instance=instance, n_traces=1, max_t=max_t, true_holds=every_step
    )
    state_labels, _ = trans.extract_sol_label(problem, None, None)
    assert torch.equal(
        state_labels[0], torch.ones(max_t - 2, len(instance.propositions))
    )

    interior_only = {
        (1, t, p) for t in range(2, max_t) for p in instance.propositions
    }
    same = _FakeSolvedProblem(
        instance=instance, n_traces=1, max_t=max_t, true_holds=interior_only
    )
    assert torch.equal(
        trans.extract_sol_label(same, None, None)[0][0], state_labels[0]
    ), "t=1 and t=max_t..max_t+1 must not reach the state label"
