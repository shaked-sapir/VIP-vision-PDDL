"""Tests for ``model_bridge.extract_state_labels``.

The MILP hands back one set of true Propositions per frame; the CV head wants a
binary matrix in *its own* column order, covering the interior frames only.
"""

from __future__ import annotations

import torch

import benchmark.algorithm_adapters.rosame_milp  # noqa: F401  (vendor sys.path)
from benchmark.algorithm_adapters.rosame_milp.model_bridge import extract_state_labels

from planning_structs.domain import Domain as PSDomain
from planning_structs.instance import Instance as PSInstance


def _world():
    """move(?from ?to) / at(?r) over rooms r1, r2."""
    domain = PSDomain(
        [("room", None)],
        [("at", ["room"])],
        [("move", ["room", "room"])],
        name="micro",
    )
    return PSInstance(domain, [("r1", "room"), ("r2", "room")])


def _prop(instance, name, args):
    pred = instance.domain.get_predicate(name)
    return instance.get_proposition(pred, tuple(instance.get_object(a) for a in args))


def test_covers_interior_frames_in_the_given_column_order():
    instance = _world()
    at_r1 = _prop(instance, "at", ["r1"])
    at_r2 = _prop(instance, "at", ["r2"])
    # Four frames: r1, r1, r2, r2. Only the two interior ones become labels.
    states = [{at_r1}, {at_r1}, {at_r2}, {at_r2}]

    labels = extract_state_labels(instance, states, ["at r2", "at r1"])

    assert labels.shape == (2, 2)
    assert torch.equal(labels, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))


def test_a_column_the_instance_lacks_is_zero():
    instance = _world()
    at_r1 = _prop(instance, "at", ["r1"])
    states = [{at_r1}, {at_r1}, {at_r1}]

    # A union-grounded CV head carries columns for objects this problem lacks.
    labels = extract_state_labels(instance, states, ["at r1", "at r3", "holding r1"])

    assert labels.shape == (1, 3)
    assert torch.equal(labels, torch.tensor([[1.0, 0.0, 0.0]]))


def test_absent_propositions_are_false_under_cwa():
    instance = _world()
    states = [set(), set(), set()]

    labels = extract_state_labels(instance, states, ["at r1", "at r2"])

    assert torch.equal(labels, torch.zeros(1, 2))


def test_a_single_step_trace_has_no_interior_frames():
    instance = _world()
    at_r1 = _prop(instance, "at", ["r1"])
    states = [{at_r1}, {at_r1}]

    labels = extract_state_labels(instance, states, ["at r1", "at r2"])

    assert labels.shape == (0, 2)
