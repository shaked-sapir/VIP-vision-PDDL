"""Tests for the ROSAME+MILP adapter (torch-free; runnable standalone).

    python -m benchmark.algorithm_adapters.rosame_milp.test_rosame_milp

Covers:
  1. Encoder recovers the GT model of a micro-domain from a single clean trace.
  2. Masked fluents (prob 0.5) carry zero objective weight yet stay consistent.
  3. ``forbid_redundant_adds`` reproduces the infeasibility we predicted for
     legal redundant adds (and turning the flag off restores feasibility).
  4. Bridge binding table maps ``forward()`` rows to PDDL parameter positions.
  5. ``MilpEncodingConfig.tag`` (add-only schema rule + no redundant-add ban)
     makes the legal-redundant-add scenario feasible.
"""

from __future__ import annotations

import itertools
from collections import OrderedDict, defaultdict

import benchmark.algorithm_adapters.rosame_milp  # noqa: F401  (vendor sys.path)
from src.milp.encoder import CPSATObservedActions
from src.milp.encoding_config import MilpEncodingConfig
from benchmark.algorithm_adapters.rosame_milp.model_bridge import binding_table

from planning_structs.domain import Domain as PSDomain
from planning_structs.instance import Instance as PSInstance
from planning_structs.traces import ObservationP, ObservationT, Traces

_EPS = 1e-5


# ---------------------------------------------------------------- helpers

def _micro_world():
    """move(?from ?to) / at(?r) over rooms r1, r2."""
    domain = PSDomain(
        [("room", None)],
        [("at", ["room"])],
        [("move", ["room", "room"])],
        name="micro",
    )
    instance = PSInstance(domain, [("r1", "room"), ("r2", "room")])
    return domain, instance


def _prop(instance, name, args):
    pred = instance.domain.get_predicate(name)
    return instance.get_proposition(pred, tuple(instance.get_object(a) for a in args))


def _action(instance, name, args):
    schema = instance.domain.get_action_schema(name)
    return instance.get_action(schema, tuple(instance.get_object(a) for a in args))


def _trace(instance, states, actions, goal_index=-1, masked=frozenset()):
    """Build an ObservationT from lists of positive-fluent sets + action tuples.

    ``masked``: (t, prop) pairs to encode at prob 0.5.
    """
    step = len(actions)
    obs_p = {}
    for t, positives in enumerate(states, start=1):
        entries = []
        for p in instance.propositions:
            if (t, p) in masked:
                entries.append(ObservationP(p, 0.5))
            else:
                entries.append(ObservationP(p, 1 - _EPS if p in positives else _EPS))
        obs_p[t] = entries
    init = sorted(states[0], key=str)
    goal = sorted(states[goal_index], key=str) if goal_index is not None else None
    trace = ObservationT(step, init, obs_p, goal, obs_a={})
    trace.instance = instance
    trace.actions = {t: a for t, a in enumerate(actions, start=1)}
    return trace


def _flat_obs_m(domain):
    """Uninformative model prior (0.5 everywhere -> objective weight 0)."""
    pre, add, dele = {}, {}, {}
    for a in domain.action_schemas:
        for p in domain.predicates:
            for x in domain.predicate_arguments[(a, p)]:
                pre[(a, p, x)] = 0.5
                add[(a, p, x)] = 0.5
                dele[(a, p, x)] = 0.5
    return type("ObsM", (), {"pre": pre, "add": add, "dele": dele})()


def _key(domain, schema_name, pred_name, x):
    return (domain.get_action_schema(schema_name), domain.get_predicate(pred_name), x)


# ---------------------------------------------------------------- tests

def test_encoder_recovers_gt_model():
    domain, instance = _micro_world()
    at_r1, at_r2 = _prop(instance, "at", ["r1"]), _prop(instance, "at", ["r2"])
    move = _action(instance, "move", ["r1", "r2"])

    trace = _trace(instance, [{at_r1}, {at_r2}], [move])
    traces = Traces(instance=None, obs_m=_flat_obs_m(domain), obs_t=[trace])

    enc = CPSATObservedActions(domain, traces, {"state", "model"})
    assert enc.solve(time_limit=30), "micro problem must be solvable"
    sol = enc.action_model_sol()

    assert sol.add[_key(domain, "move", "at", (2,))] == 1, "add at(?to) expected"
    assert sol.dele[_key(domain, "move", "at", (1,))] == 1, "del at(?from) expected"
    assert sol.pre[_key(domain, "move", "at", (1,))] == 1, "pre at(?from) (del=>pre)"
    assert sol.add[_key(domain, "move", "at", (1,))] == 0
    print("PASS  encoder recovers GT model on micro-domain")


def test_masked_fluent_free():
    """A masked intermediate value must not block the consistent solution."""
    domain, instance = _micro_world()
    at_r1, at_r2 = _prop(instance, "at", ["r1"]), _prop(instance, "at", ["r2"])
    move_12 = _action(instance, "move", ["r1", "r2"])
    move_21 = _action(instance, "move", ["r2", "r1"])

    # r1 -> r2 -> r1; the middle state's at(r2) is masked (prob 0.5)
    states = [{at_r1}, {at_r2}, {at_r1}]
    trace = _trace(instance, states, [move_12, move_21], masked={(2, at_r2)})
    traces = Traces(instance=None, obs_m=_flat_obs_m(domain), obs_t=[trace])

    enc = CPSATObservedActions(domain, traces, {"state", "model"})
    assert enc.solve(time_limit=30)
    sol = enc.action_model_sol()
    # the repaired middle state must still set at(r2) (needed by move_21's frame)
    repaired = enc.repaired_states(1)
    assert at_r2 in repaired[1], "masked at(r2) should be repaired to true"
    assert sol.add[_key(domain, "move", "at", (2,))] == 1
    print("PASS  masked fluents are free and consistently repaired")


def test_forbid_redundant_adds_infeasibility():
    """Our depot argument: with the flag on, a redundant add makes GT infeasible."""
    domain, instance = _micro_world()
    at_r1, at_r2 = _prop(instance, "at", ["r1"]), _prop(instance, "at", ["r2"])
    move = _action(instance, "move", ["r1", "r2"])

    # at(r2) already true before move(r1,r2): the add of at(?to) is redundant.
    states = [{at_r1, at_r2}, {at_r2}]
    trace = _trace(instance, states, [move])

    # flag ON (upstream behavior): nonempty-add + no-redundant-add + goal
    # constraints admit no model -> infeasible
    traces = Traces(instance=None, obs_m=_flat_obs_m(domain), obs_t=[trace])
    enc_on = CPSATObservedActions(domain, traces, {"state", "model"},
                                  forbid_redundant_adds=True)
    assert not enc_on.solve(time_limit=30), "expected infeasibility with flag on"

    # flag OFF: the redundant add at(?to) is admissible
    traces = Traces(instance=None, obs_m=_flat_obs_m(domain), obs_t=[trace])
    enc_off = CPSATObservedActions(domain, traces, {"state", "model"},
                                   forbid_redundant_adds=False)
    assert enc_off.solve(time_limit=30), "expected feasibility with flag off"
    sol = enc_off.action_model_sol()
    assert sol.add[_key(domain, "move", "at", (2,))] == 1
    assert sol.dele[_key(domain, "move", "at", (1,))] == 1
    print("PASS  forbid_redundant_adds on->infeasible / off->feasible")


def test_tag_config_allows_redundant_add():
    """MilpEncodingConfig.tag (add-only schema rule + no redundant-add ban)
    makes the legal-redundant-add scenario feasible."""
    domain, instance = _micro_world()
    at_r1, at_r2 = _prop(instance, "at", ["r1"]), _prop(instance, "at", ["r2"])
    move = _action(instance, "move", ["r1", "r2"])

    # at(r2) already true before move(r1,r2): the add of at(?to) is redundant —
    # infeasible under upstream rules, admissible under the tag config.
    states = [{at_r1, at_r2}, {at_r2}]
    trace = _trace(instance, states, [move])

    traces = Traces(instance=None, obs_m=_flat_obs_m(domain), obs_t=[trace])
    enc = CPSATObservedActions(domain, traces, {"state", "model"},
                               config=MilpEncodingConfig.tag())
    assert enc.solve(time_limit=30), "expected feasibility under tag config"
    sol = enc.action_model_sol()
    assert sol.add[_key(domain, "move", "at", (2,))] == 1
    assert sol.dele[_key(domain, "move", "at", (1,))] == 1
    print("PASS  tag config admits the legal redundant add")


# ------------------------------------------------- bridge (torch-free fakes)

class _FakeType:
    def __init__(self, name):
        self.name = name

    def __str__(self):  # pddl_plus PDDLType stringifies to its name
        return self.name

    def is_child(self, other):
        return self.name == other.name


class _FakePredicate:
    """Mimics AMLGym rosame.Predicate.ground's enumeration order."""

    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.params_types = list(params.keys())

    def ground(self, objects):
        per_type = {pt: [] for pt in self.params_types}
        for pt in self.params_types:
            for ot, objs in objects.items():
                if ot.is_child(pt):
                    per_type[pt].extend(objs)
        props = []
        for combo in itertools.product(
            *[itertools.permutations(per_type[pt], self.params[pt]) for pt in self.params_types]
        ):
            names = [o for group in combo for o in group]
            props.append((self.name + " " + " ".join(names)).strip())
        return props


class _FakeSchema:
    def __init__(self, name, params, predicates):
        self.name = name
        self.params = params
        self.params_types = list(params.keys())
        self.predicates = predicates


class _FakeRunner:
    def __init__(self, actions_signatures):
        acts = {}
        for name, sig in actions_signatures.items():
            acts[name] = type("A", (), {"signature": OrderedDict(sig)})()
        self.domain = type("D", (), {"actions": acts})()

    def get_params_names(self, action):
        out = defaultdict(list)
        for pname, ptype in action.signature.items():
            out[str(ptype)].append(pname[1:])
        return out


def test_binding_table_positions():
    room = _FakeType("room")
    at = _FakePredicate("at", {room: 1})
    schema = _FakeSchema("move", {room: 2}, [at])
    runner = _FakeRunner({"move": [("?from", room), ("?to", room)]})

    rows = binding_table(runner, schema)
    assert rows == [("at", (1,)), ("at", (2,))], f"unexpected rows: {rows}"
    print("PASS  binding table maps rows to PDDL parameter positions")


if __name__ == "__main__":
    test_binding_table_positions()
    test_encoder_recovers_gt_model()
    test_masked_fluent_free()
    test_forbid_redundant_adds_infeasibility()
    test_tag_config_allows_redundant_add()
    print("ALL TESTS PASSED")
