"""Tests for :func:`src.milp.converter.cv_predictions_to_trace` (torch-free).

    python -m src.milp.test_cv_predictions_to_trace

Covers:
  1. Endpoints are hard rows from init/goal; the CV's own endpoint rows are
     never read.
  2. Interior rows carry the clamped CV probabilities, column-mapped by name.
  3. A proposition the CV head has no column for gets the neutral 0.5.
  4. A CV column with no instance counterpart is dropped, not misaligned.
  5. A soft goal drops the row at T+1 rather than inventing one.
  6. Shape mismatches and ungroundable action calls raise.
  7. It agrees with `observation_to_trace` on the propositions they share.
  8. The encoder solves a trace built this way and recovers the GT model.
"""

from __future__ import annotations

import src.milp  # noqa: F401  (vendor sys.path bootstrap)
from src.milp.converter import _EPS, cv_predictions_to_trace, observation_to_trace
from src.milp.encoder import CPSATObservedActions

from planning_structs.domain import Domain as PSDomain
from planning_structs.instance import Instance as PSInstance
from planning_structs.traces import Traces


# ---------------------------------------------------------------- helpers

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


def _probs_by_name(instance, row):
    """{proposition name string: prob} for one obs_p row."""
    return {str(op.proposition): op.prob for op in row}


def _key(domain, schema_name, pred_name, x):
    return (domain.get_action_schema(schema_name), domain.get_predicate(pred_name), x)


_NAMES = ["at r1", "at r2"]


# ---------------------------------------------------------------- tests

def test_endpoints_are_hard_and_cv_endpoints_unread():
    _domain, instance = _micro_world()
    # r1 -> r2 -> r1. The CV is confidently WRONG at both endpoints.
    probs = [[0.0, 1.0], [0.2, 0.8], [0.0, 1.0]]
    calls = [("move", ["r1", "r2"]), ("move", ["r2", "r1"])]

    trace = cv_predictions_to_trace(
        instance, _NAMES, probs, calls,
        init_fluents={("at", ("r1",))},
        goal_fluents={("at", ("r1",))},
    )

    first = _probs_by_name(instance, trace.obs_p[1])
    last = _probs_by_name(instance, trace.obs_p[3])
    assert first == {"at_r1": 1 - _EPS, "at_r2": _EPS}, first
    assert last == {"at_r1": 1 - _EPS, "at_r2": _EPS}, last
    assert [str(p) for p in trace.init] == ["at_r1"]
    assert [str(p) for p in trace.goal] == ["at_r1"]
    print("PASS  endpoints are hard rows; probs[0] and probs[T] are not read")


def test_interior_rows_are_clamped_cv_values():
    _domain, instance = _micro_world()
    # Out-of-range values: our CV head emits raw logits, not sigmoid outputs.
    probs = [[0.0, 0.0], [-3.0, 4.0], [0.0, 0.0]]
    calls = [("move", ["r1", "r2"]), ("move", ["r2", "r1"])]

    trace = cv_predictions_to_trace(
        instance, _NAMES, probs, calls,
        init_fluents={("at", ("r1",))},
        goal_fluents={("at", ("r1",))},
    )

    interior = _probs_by_name(instance, trace.obs_p[2])
    assert interior == {"at_r1": _EPS, "at_r2": 1 - _EPS}, interior
    print("PASS  interior rows are the CV values, clamped to [eps, 1-eps]")


def test_missing_cv_column_is_neutral():
    _domain, instance = _micro_world()
    # The CV head knows nothing about at(r2).
    probs = [[0.0], [0.9], [0.0]]
    calls = [("move", ["r1", "r2"]), ("move", ["r2", "r1"])]

    trace = cv_predictions_to_trace(
        instance, ["at r1"], probs, calls,
        init_fluents={("at", ("r1",))},
        goal_fluents={("at", ("r1",))},
    )

    interior = _probs_by_name(instance, trace.obs_p[2])
    assert interior["at r1".replace(" ", "_")] == 0.9
    assert interior["at_r2"] == 0.5, interior
    print("PASS  a proposition with no CV column is neutral (weight 0)")


def test_unknown_cv_column_is_dropped_without_shifting():
    _domain, instance = _micro_world()
    # 'at r3' has no object r3 in the instance; it must not consume at_r2's value.
    names = ["at r1", "at r3", "at r2"]
    probs = [[0.0, 0.0, 0.0], [0.1, 0.99, 0.7], [0.0, 0.0, 0.0]]
    calls = [("move", ["r1", "r2"]), ("move", ["r2", "r1"])]

    trace = cv_predictions_to_trace(
        instance, names, probs, calls,
        init_fluents={("at", ("r1",))},
        goal_fluents={("at", ("r1",))},
    )

    interior = _probs_by_name(instance, trace.obs_p[2])
    assert interior == {"at_r1": 0.1, "at_r2": 0.7}, interior
    print("PASS  an unmappable CV column is dropped, columns stay aligned")


def test_soft_goal_drops_the_final_row():
    _domain, instance = _micro_world()
    probs = [[0.0, 1.0], [0.2, 0.8], [0.0, 1.0]]
    calls = [("move", ["r1", "r2"]), ("move", ["r2", "r1"])]

    trace = cv_predictions_to_trace(
        instance, _NAMES, probs, calls, init_fluents={("at", ("r1",))}, goal_fluents=None
    )

    assert trace.goal is None
    assert sorted(trace.obs_p) == [1, 2], sorted(trace.obs_p)
    print("PASS  a soft goal leaves T+1 unobserved instead of inventing a row")


def test_shape_and_grounding_guards():
    _domain, instance = _micro_world()
    calls = [("move", ["r1", "r2"])]
    init = {("at", ("r1",))}

    for bad, why in [
        ([[0.0, 1.0]], "too few rows"),
        ([[0.0], [1.0]], "wrong width"),
    ]:
        try:
            cv_predictions_to_trace(instance, _NAMES, bad, calls, init)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {why}")

    try:
        cv_predictions_to_trace(
            instance, _NAMES, [[0.0, 1.0], [0.0, 1.0]], [("move", ["r1", "r3"])], init
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an ungroundable action call")
    print("PASS  shape mismatches and ungroundable calls raise")


class _FakeGP:
    """The duck-typed surface the converter reads off a grounded predicate."""

    def __init__(self, name, arg_names, is_positive=True, is_masked=False):
        self.name = name
        self.signature = {f"?x{i}": "room" for i in range(len(arg_names))}
        self.object_mapping = dict(zip(self.signature, arg_names))
        self.is_positive = is_positive
        self.is_masked = is_masked


class _FakeState:
    def __init__(self, gps):
        self.state_predicates = {"at": list(gps)}


class _FakeComponent:
    def __init__(self, previous_state, action_name, parameters, next_state):
        self.previous_state = previous_state
        self.next_state = next_state
        self.grounded_action_call = type(
            "Call", (), {"name": action_name, "parameters": list(parameters)}
        )()


class _FakeObservation:
    def __init__(self, components):
        self.components = components


def _cwa_observation():
    """r1 -> r2 -> r3, every state stating all three rooms; at(r1) masked at t=2."""
    def state(true_room, masked=()):
        return _FakeState([
            _FakeGP("at", [r], is_positive=(r == true_room), is_masked=(r in masked))
            for r in ("r1", "r2", "r3")
        ])

    states = [state("r1"), state("r2", masked=("r1",)), state("r3")]
    return _FakeObservation([
        _FakeComponent(states[0], "move", ["r1", "r2"], states[1]),
        _FakeComponent(states[1], "move", ["r2", "r3"], states[2]),
    ])


def test_parity_with_observation_to_trace_on_the_shared_subset():
    _domain, instance = _micro_world(3)
    goal = {("at", ("r3",))}
    symbolic = observation_to_trace(instance, _cwa_observation(), goal_fluents=goal)

    # Ternarized CV output: the same probabilities, minus a column for at(r3).
    names = ["at r1", "at r2"]
    rows = [_probs_by_name(instance, symbolic.obs_p[t]) for t in (1, 2, 3)]
    probs = [[row[n.replace(" ", "_")] for n in names] for row in rows]

    cv = cv_predictions_to_trace(
        instance, names, probs, [("move", ["r1", "r2"]), ("move", ["r2", "r3"])],
        init_fluents={("at", ("r1",))}, goal_fluents=goal,
    )

    interior = _probs_by_name(instance, cv.obs_p[2])
    shared = {k: v for k, v in interior.items() if k in ("at_r1", "at_r2")}
    assert shared == {k: v for k, v in rows[1].items() if k in shared}, shared
    assert interior["at_r3"] == 0.5, "a proposition outside the CV head must be free"
    assert [str(p) for p in cv.init] == [str(p) for p in symbolic.init]
    assert {str(p) for p in cv.goal} == {str(p) for p in symbolic.goal}
    assert cv.actions == symbolic.actions
    print("PASS  the two entry points agree on the propositions they share")


def test_encoder_recovers_gt_model_from_cv_trace():
    domain, instance = _micro_world()
    # r1 -> r2 -> r1, with the interior frame only weakly (and wrongly) predicted.
    probs = [[9.0, -9.0], [0.55, 0.45], [9.0, -9.0]]
    calls = [("move", ["r1", "r2"]), ("move", ["r2", "r1"])]

    trace = cv_predictions_to_trace(
        instance, _NAMES, probs, calls,
        init_fluents={("at", ("r1",))},
        goal_fluents={("at", ("r1",))},
    )

    enc = CPSATObservedActions(domain, Traces(instance=None, obs_m=None, obs_t=[trace]), {"state"})
    assert enc.solve(time_limit=30), "CV-derived trace must be solvable"
    sol = enc.action_model_sol()
    assert sol.add[_key(domain, "move", "at", (2,))] == 1, "add at(?to) expected"
    assert sol.dele[_key(domain, "move", "at", (1,))] == 1, "del at(?from) expected"

    repaired = enc.repaired_states(1)  # traces are 1-indexed
    at_r2 = instance.get_proposition(
        domain.get_predicate("at"), (instance.get_object("r2"),)
    )
    assert at_r2 in repaired[1], "the weakly-predicted interior frame is repaired"
    print("PASS  encoder solves a CV-derived trace and recovers the GT model")


if __name__ == "__main__":
    test_endpoints_are_hard_and_cv_endpoints_unread()
    test_interior_rows_are_clamped_cv_values()
    test_missing_cv_column_is_neutral()
    test_unknown_cv_column_is_dropped_without_shifting()
    test_soft_goal_drops_the_final_row()
    test_shape_and_grounding_guards()
    test_parity_with_observation_to_trace_on_the_shared_subset()
    test_encoder_recovers_gt_model_from_cv_trace()
    print("ALL TESTS PASSED")
