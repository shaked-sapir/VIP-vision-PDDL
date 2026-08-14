"""Tests for the ``cdps_milp_*`` MILP denoiser (standalone; no PDDL parsing).

    python -m src.plan_denoising.milp_denoiser.test_cdps_milp

The suite is organised around the two claims the whole design rests on
(``docs/cdps-milp-denoiser-design.md`` §3-§4):

1. **The CDPS dialect cannot exclude a legal ground-truth model.** Three
   constraint families were dropped for that reason; each test below exhibits a
   legal world that the family makes *infeasible* and the dialect solves. Each
   compares ``cdps_dialect()`` against ``cdps_dialect()`` with exactly one family
   switched back on, so a failure names the guilty family.
2. **The solve returns the globally cheapest repair**, and the vocabulary it
   repairs over is the same one PI-SAM will see (the repeated-argument
   groundings).

Plus the cheap-but-error-prone plumbing: config validation and the
state-index -> (component, prev/next) mapping of the repair log.
"""

from __future__ import annotations

import dataclasses

from planning_structs.domain import Domain as PSDomain
from planning_structs.instance import Instance as PSInstance
from planning_structs.traces import ObservationP, ObservationT, Traces

from src.plan_denoising.milp_denoiser.config import (
    CdpsMilpConfig,
    MilpSolver,
)
from src.milp.converter import (
    GtAnchoring,
    RepeatedArgsInstance,
    build_ps_instance_from_objects,
    observation_to_trace,
    try_observation_to_trace,
)
from src.milp.encoder import CPSATObservedActions
from src.milp.encoding_config import (
    MilpEncodingConfig,
    PriorWeightMode,
    SchemaNonemptyRule,
)
from src.plan_denoising.milp_denoiser.trajectory_extraction import (
    ExtractionResult,
    FluentFlip,
)

_EPS = 1e-5
_TIME_LIMIT = 30


# ---------------------------------------------------------------- helpers

def _micro_world(n_rooms: int = 2):
    """``move(?from ?to)`` / ``at(?r)`` over ``r1..rN``.

    Small enough that every forced model bit can be reasoned about by hand,
    which is what makes the infeasibility claims below checkable.
    """
    domain = PSDomain(
        [("room", None)],
        [("at", ["room"])],
        [("move", ["room", "room"])],
        name="micro",
    )
    instance = PSInstance(domain, [(f"r{i}", "room") for i in range(1, n_rooms + 1)])
    return domain, instance


def _prop(instance, name, args):
    pred = instance.domain.get_predicate(name)
    return instance.get_proposition(pred, tuple(instance.get_object(a) for a in args))


def _action(instance, name, args):
    schema = instance.domain.get_action_schema(name)
    return instance.get_action(schema, tuple(instance.get_object(a) for a in args))


def _trace(instance, states, actions, goal_index=-1, masked=frozenset()):
    """ObservationT from positive-fluent sets + observed actions.

    ``goal_index=None`` leaves the final state *soft* (repairable, priced by the
    objective) — that is how our traces normally arrive, since only the initial
    state is GT by assumption. ``masked``: ``(t, prop)`` pairs encoded at 0.5.
    """
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
    trace = ObservationT(len(actions), init, obs_p, goal, obs_a={})
    trace.instance = instance
    trace.actions = {t: a for t, a in enumerate(actions, start=1)}
    return trace


def _solve(domain, traces_list, config):
    """Encode + solve with the state objective only (single round has no prior)."""
    traces = Traces(instance=None, obs_m=None, obs_t=list(traces_list))
    encoder = CPSATObservedActions(domain, traces, {"state"}, config=config)
    return encoder, encoder.solve(time_limit=_TIME_LIMIT)


def _with(config: MilpEncodingConfig, **overrides) -> MilpEncodingConfig:
    """The dialect with one family switched back on — isolates a single cause."""
    return dataclasses.replace(config, **overrides)


def _disagreements(encoder, trace_index: int, trace) -> int:
    """How many *observed* (unmasked) fluents the solution flipped in one trace.

    The in-encoder analogue of ``ExtractionResult.repair_cost``: masked slots sit
    at prob 0.5 and are excluded, exactly as ``trajectory_extraction`` excludes
    them from T'.
    """
    count = 0
    for t, entries in trace.obs_p.items():
        for op in entries:
            if op.prob == 0.5:
                continue
            observed = op.prob > 0.5
            if bool(encoder.hol[(trace_index, t, op.proposition)].value()) != observed:
                count += 1
    return count


# ------------------------------------------- dialect: dropped families

def test_dialect_allows_delete_without_precondition():
    """Paper eq. 18 (``del => pre``) excludes a legal model.

    Trace 1 pins ``del[move, at, (1,)] = 1`` (``at(?from)`` goes true -> false
    between two hard-fixed states). Trace 2 then executes ``move(r1, r3)`` from a
    hard-fixed state where ``at(r1)`` is *false* — legal PDDL (deleting a fluent
    that is not a precondition is a no-op), but ``del => pre`` forces
    ``pre[move, at, (1,)] = 1``, hence ``at(r1)`` true, hence infeasible.
    """
    domain, instance = _micro_world(3)
    at1, at2, at3 = (_prop(instance, "at", [r]) for r in ("r1", "r2", "r3"))

    t1 = _trace(instance, [{at1}, {at2}], [_action(instance, "move", ["r1", "r2"])])
    t2 = _trace(instance, [{at2}, {at2, at3}], [_action(instance, "move", ["r1", "r3"])])

    dialect = MilpEncodingConfig.cdps_dialect()
    _, ok = _solve(domain, [t1, t2], dialect)
    assert ok, "the CDPS dialect must admit a delete of a non-precondition"

    _, ok_eq18 = _solve(domain, [t1, t2], _with(dialect, delete_implies_precondition=True))
    assert not ok_eq18, "del => pre was expected to exclude this legal model"
    print("PASS  dialect admits delete-without-precondition (eq. 18 does not)")


def test_dialect_allows_delete_only_schema():
    """The upstream per-schema non-empty rule excludes a legal model.

    ``move(r1, r2)`` runs between two hard-fixed states and leaves *nothing*
    true, so every add bit is forced to 0 — a legal delete-only action.
    ``PreIsNotEmpty``/``AddIsNotEmpty`` demand at least one add effect anyway.
    """
    domain, instance = _micro_world(2)
    at1 = _prop(instance, "at", ["r1"])

    trace = _trace(instance, [{at1}, set()], [_action(instance, "move", ["r1", "r2"])])

    dialect = MilpEncodingConfig.cdps_dialect()
    _, ok = _solve(domain, [trace], dialect)
    assert ok, "the CDPS dialect must admit a delete-only action schema"

    _, ok_nonempty = _solve(
        domain, [trace], _with(dialect, schema_nonempty=SchemaNonemptyRule.PRE_AND_ADD)
    )
    assert not ok_nonempty, "the non-empty rule was expected to exclude this legal model"
    print("PASS  dialect admits a delete-only schema (non-empty rule does not)")


def test_dialect_allows_redundant_add():
    """The upstream redundant-add ban (``stepadd & hol``) excludes a legal model.

    Trace 1 forces ``add[move, at, (2,)] = 1``. Trace 2 executes the same schema
    where ``at(?to)`` is *already* true — a legal redundant add, which the ban
    forbids outright.
    """
    domain, instance = _micro_world(2)
    at1, at2 = _prop(instance, "at", ["r1"]), _prop(instance, "at", ["r2"])
    move = _action(instance, "move", ["r1", "r2"])

    t1 = _trace(instance, [{at1}, {at2}], [move])
    t2 = _trace(instance, [{at1, at2}, {at2}], [move])

    dialect = MilpEncodingConfig.cdps_dialect()
    _, ok = _solve(domain, [t1, t2], dialect)
    assert ok, "the CDPS dialect must admit a redundant add"

    _, ok_ban = _solve(domain, [t1, t2], _with(dialect, forbid_redundant_adds=True))
    assert not ok_ban, "the redundant-add ban was expected to exclude this legal model"
    print("PASS  dialect admits a redundant add (the upstream ban does not)")


# ------------------------------------------------- optimality of the repair

def test_minimal_repair_flips_exactly_one_fluent():
    """A one-noisy-fluent world must cost exactly one flip.

    Trace 1 is fully hard-fixed and forces ``del[at,(1,)] = add[at,(2,)] = 1``.
    Trace 2's final state is soft and observed as ``{at(r2), at(r3)}`` after
    ``move(r2, r3)`` — ``at(r2)`` is the noise. The forced delete makes flipping
    it unavoidable, and nothing else disagrees, so the optimum is cost 1.
    """
    domain, instance = _micro_world(3)
    at1, at2, at3 = (_prop(instance, "at", [r]) for r in ("r1", "r2", "r3"))

    t1 = _trace(instance, [{at1}, {at2}], [_action(instance, "move", ["r1", "r2"])])
    t2 = _trace(
        instance,
        [{at2}, {at2, at3}],  # at(r2) still true after leaving r2: the noise
        [_action(instance, "move", ["r2", "r3"])],
        goal_index=None,  # soft final state — this is the repairable one
    )

    encoder, ok = _solve(domain, [t1, t2], MilpEncodingConfig.cdps_dialect())
    assert ok, "the repair problem must be feasible"

    repaired = encoder.repaired_states(2)
    assert at2 not in repaired[1], "the noisy at(r2) must be repaired away"
    assert at3 in repaired[1], "the observed at(r3) must survive"
    assert _disagreements(encoder, 1, t1) == 0, "the hard-fixed trace cannot be repaired"
    assert _disagreements(encoder, 2, t2) == 1, "the optimal repair costs exactly 1 flip"
    print("PASS  minimal repair flips exactly the one noisy fluent")


def test_masked_fluent_is_free():
    """A masked slot carries zero objective weight, so it is not a repair.

    Same world as above with the noisy fluent masked instead: the solution must
    still remove it from the state, but at cost 0 — masked slots are the family
    of consistent completions, not evidence (design §4.2).
    """
    domain, instance = _micro_world(3)
    at1, at2, at3 = (_prop(instance, "at", [r]) for r in ("r1", "r2", "r3"))

    t1 = _trace(instance, [{at1}, {at2}], [_action(instance, "move", ["r1", "r2"])])
    t2 = _trace(
        instance,
        [{at2}, {at2, at3}],
        [_action(instance, "move", ["r2", "r3"])],
        goal_index=None,
        masked={(2, at2)},
    )

    encoder, ok = _solve(domain, [t1, t2], MilpEncodingConfig.cdps_dialect())
    assert ok
    assert at2 not in encoder.repaired_states(2)[1], "the completion must respect the delete"
    assert _disagreements(encoder, 2, t2) == 0, "a masked slot must never count as a repair"
    print("PASS  masked fluents are free (completed, not repaired)")


# ------------------------------------------------- repeated-argument grounding

def _mini_blocks():
    """``stack(?x ?y)`` / ``on(?x ?y)`` over blocks a, b — the arity-2 case."""
    return PSDomain(
        [("block", None)],
        [("on", ["block", "block"])],
        [("stack", ["block", "block"])],
        name="mini_blocks",
    )


def test_repeated_args_instance_grounds_reflexive_tuples():
    """``(on a a)`` / ``stack(a, a)`` exist only under ``RepeatedArgsInstance``.

    Upstream grounds with ``permutations`` (distinct objects only). Our
    observations are completed with ``product``, so without this widening a
    reflexive fluent would get no ``hol`` variable, pass through the repair
    untouched, and reach PI-SAM unconstrained — the hole design §4.1 forbids.
    """
    domain = _mini_blocks()
    objects = [("a", "block"), ("b", "block")]
    plain, widened = PSInstance(domain, objects), RepeatedArgsInstance(domain, objects)

    on, stack = domain.get_predicate("on"), domain.get_action_schema("stack")
    aa_plain = (plain.get_object("a"), plain.get_object("a"))
    aa_wide = (widened.get_object("a"), widened.get_object("a"))

    assert plain.get_proposition(on, aa_plain) is None, "upstream must not ground (on a a)"
    assert widened.get_proposition(on, aa_wide) is not None, "(on a a) must be grounded"
    assert plain.get_action(stack, aa_plain) is None, "upstream must not ground stack(a,a)"
    assert widened.get_action(stack, aa_wide) is not None, "stack(a,a) must be grounded"

    # n!/(n-k)! = 2 vs n^k = 4 for each of the predicate and the schema
    assert len(plain.propositions) == 2 and len(widened.propositions) == 4
    print("PASS  RepeatedArgsInstance grounds reflexive tuples")


def test_repeated_args_action_unifies_both_bindings():
    """``(on a a)`` under ``stack(a, a)`` must OR *both* lifted bindings.

    The lifted vocabulary is not widened (PI-SAM's matcher enumerates distinct
    parameter slots), so the encoder reproduces PI-SAM's behaviour by unifying
    the reflexive proposition with ``(on ?x ?y)`` *and* ``(on ?y ?x)``.
    """
    domain = _mini_blocks()
    instance = RepeatedArgsInstance(domain, [("a", "block"), ("b", "block")])
    a = instance.get_object("a")
    on_aa = instance.get_proposition(domain.get_predicate("on"), (a, a))
    stack_aa = instance.get_action(domain.get_action_schema("stack"), (a, a))

    trace = _trace(instance, [set(), {on_aa}], [stack_aa], goal_index=None)
    encoder, _ = _solve(domain, [trace], MilpEncodingConfig.cdps_dialect())

    bindings = encoder._bindings(stack_aa, on_aa)
    assert sorted(bindings) == [(1, 2), (2, 1)], f"unexpected bindings: {bindings}"
    print("PASS  a reflexive action unifies both lifted bindings")


# ------------------------------------------------------------ gt anchoring

class _FakeGP:
    """The duck-typed surface ``converter`` reads off a grounded predicate."""

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


def _three_state_observation():
    """r1 -> r2 -> r3; the middle state's ``at(r2)`` is observed (unmasked)."""
    states = [
        _FakeState([_FakeGP("at", ["r1"]), _FakeGP("at", ["r2"], is_positive=False)]),
        _FakeState([_FakeGP("at", ["r2"]), _FakeGP("at", ["r1"], is_positive=False)]),
        _FakeState([_FakeGP("at", ["r3"]), _FakeGP("at", ["r2"], is_positive=False)]),
    ]
    return _FakeObservation([
        _FakeComponent(states[0], "move", ["r1", "r2"], states[1]),
        _FakeComponent(states[1], "move", ["r2", "r3"], states[2]),
    ])


def test_gt_anchoring_modes():
    """``ALL_GT_STATES`` pins the extra GT states; ``INIT_ONLY`` pins none.

    Also guards the off-by-one: ``gt_states_by_obs`` uses 0-based *state*
    indices, the encoder uses 1-based *time* indices, so state 1 must land at
    time 2 (and never in ``hard_states`` under ``INIT_ONLY``, where the encoder
    supplies the initial state itself).
    """
    _, instance = _micro_world(3)
    observation = _three_state_observation()

    init_only = observation_to_trace(
        instance, observation, gt_state_indices={1}, gt_anchoring=GtAnchoring.INIT_ONLY
    )
    assert init_only.hard_states == {}, "INIT_ONLY must not pin intermediate states"

    anchored = observation_to_trace(
        instance, observation, gt_state_indices={1}, gt_anchoring=GtAnchoring.ALL_GT_STATES
    )
    assert set(anchored.hard_states) == {2}, "state index 1 must map to time index 2"
    assert anchored.hard_states[2] == {_prop(instance, "at", ["r2"])}
    print("PASS  gt_anchoring modes pin the right states (0-based -> 1-based)")


def test_gt_anchoring_out_of_range_index_ignored():
    """A GT index past the end of the trace must be dropped, not crash later.

    ``hard_states`` indices are range-checked by the encoder, so silently
    letting one through would fail deep inside the solve instead of here.
    """
    _, instance = _micro_world(3)
    trace = observation_to_trace(
        instance,
        _three_state_observation(),
        gt_state_indices={0, 2, 99},
        gt_anchoring=GtAnchoring.ALL_GT_STATES,
    )
    assert set(trace.hard_states) == {1, 3}, "only in-range state indices may be pinned"
    print("PASS  out-of-range GT state indices are ignored")


# --------------------------------------------------------- object typing

def _depot_lite():
    """depot's shape in miniature: ``drop(?p - package ?pl - pile)``, ``clear`` untyped.

    ``clear`` must hold of both packages and piles, which share no supertype, so
    depot leaves its one slot untyped. That is the whole reason the real bug
    exists — see :func:`converter.problem_object_types`.
    """
    return PSDomain(
        [("pile", None), ("package", None)],
        [("on-pile", ["package", "pile"]), ("clear", ["object"])],
        [("drop", ["package", "pile"])],
        name="depot_lite",
    )


class _FakePDDLObject:
    """The duck-typed surface the instance builders read: just ``.type``."""

    def __init__(self, type_name: str):
        self.type = type_name


class _NoConstants:
    constants: dict = {}


def _pile2_typed_object():
    """What ``TrajectoryParser`` infers when ``(clear pile2)`` is the last mention."""
    return {"p1": _FakePDDLObject("package"), "pile2": _FakePDDLObject("object")}


def test_inferred_object_type_makes_an_action_ungroundable():
    """The bug, reproduced: ``pile2:object`` kills every ``drop`` mentioning it.

    ``.trajectory`` files carry no ``(:objects ...)``, so types are inferred
    from predicate slots and an untyped slot wins if it comes last. ``object``
    is not a child of ``pile``, so ``_type_match`` rejects the grounding and
    ``get_action`` returns ``None``. In production this silently cost 250 of
    320 depot MILP folds one training trajectory each.
    """
    domain = _depot_lite()
    instance = build_ps_instance_from_objects(
        domain, _NoConstants(), _pile2_typed_object(), include_repeated_args=True
    )
    drop = domain.get_action_schema("drop")
    args = (instance.get_object("p1"), instance.get_object("pile2"))

    assert instance.get_action(drop, args) is None, "the mistyped grounding must be absent"
    print("PASS  an inferred 'object' type makes the observed action ungroundable")


def test_object_types_overlay_restores_the_grounding():
    """The fix: declared types from the problem file make ``drop`` groundable.

    The overlay must move types only. The object *set* is what the observation's
    states are written over, so changing it would leave propositions with no
    observed entry; the proposition vocabulary, by contrast, is *supposed* to
    grow — the trace's own ``(on-pile p1 pile2)`` is exactly what the wrong type
    had excluded.
    """
    domain = _depot_lite()
    objects = _pile2_typed_object()
    inferred = build_ps_instance_from_objects(
        domain, _NoConstants(), objects, include_repeated_args=True
    )
    declared = build_ps_instance_from_objects(
        domain, _NoConstants(), objects, include_repeated_args=True,
        object_types={"pile2": "pile"},
    )

    drop = domain.get_action_schema("drop")
    args = (declared.get_object("p1"), declared.get_object("pile2"))
    assert declared.get_action(drop, args) is not None, "drop(p1, pile2) must ground"

    assert {o.name for o in inferred.objects} == {o.name for o in declared.objects}, \
        "the overlay must not add or remove objects"
    assert declared.get_object("pile2").type.name == "pile"
    on_pile = domain.get_predicate("on-pile")
    pair = (declared.get_object("p1"), declared.get_object("pile2"))
    assert inferred.get_proposition(on_pile, (inferred.get_object("p1"),
                                              inferred.get_object("pile2"))) is None
    assert declared.get_proposition(on_pile, pair) is not None, \
        "the corrected type must admit the fluent the trace actually contains"
    print("PASS  the object_types overlay restores the grounding, objects unchanged")


def test_object_types_overlay_ignores_unknown_names():
    """Names the observation does not have must not become objects.

    The overlay is fed a whole problem's ``:objects``, and a problem may declare
    objects its trajectory never mentions. Grounding those would add
    propositions no state of the observation has an entry for — a change to the
    MILP's shape rather than a fix to its types.
    """
    domain = _depot_lite()
    instance = build_ps_instance_from_objects(
        domain, _NoConstants(), _pile2_typed_object(), include_repeated_args=True,
        object_types={"pile2": "pile", "p9": "package", "pile9": "pile"},
    )
    assert {o.name for o in instance.objects} == {"p1", "pile2"}, \
        "an overlay-only name must not be grounded"
    print("PASS  the overlay never introduces objects")


def _drop_observation():
    """One step: ``drop(p1, pile2)``, over the depot-lite vocabulary."""
    before = _FakeState([_FakeGP("clear", ["pile2"])])
    after = _FakeState([_FakeGP("clear", ["pile2"], is_positive=False)])
    return _FakeObservation([_FakeComponent(before, "drop", ["p1", "pile2"], after)])


def test_unmatched_action_raises_with_the_type_diagnosis():
    """An ungroundable observed action must raise, naming the offending slot.

    It used to ``print`` and ``return None``, which turned a mistyped object
    into a quietly smaller trace set. Three defects land on the same ``None`` —
    unknown action, ungrounded object, type mismatch — and only the message can
    tell them apart.
    """
    domain = _depot_lite()
    instance = build_ps_instance_from_objects(
        domain, _NoConstants(), _pile2_typed_object(), include_repeated_args=True
    )
    _expect_error(
        lambda: observation_to_trace(instance, _drop_observation()),
        ValueError,
        "argument type mismatch",
    )
    _expect_error(
        lambda: observation_to_trace(instance, _drop_observation()),
        ValueError,
        "pile2:object (slot 2 wants pile)",
    )
    print("PASS  an ungroundable action raises and names the mistyped slot")


def test_try_observation_to_trace_still_returns_none():
    """The tolerant wrapper keeps its contract — the loop's warm starts need it.

    A hint trace is built from a *repair*; failing to re-encode one costs
    nothing but the warm start, so that caller must not be made to raise.
    """
    domain = _depot_lite()
    instance = build_ps_instance_from_objects(
        domain, _NoConstants(), _pile2_typed_object(), include_repeated_args=True
    )
    assert try_observation_to_trace(instance, _drop_observation()) is None
    print("PASS  try_observation_to_trace still absorbs the failure")


# ------------------------------------------------------------ config surface

def _expect_error(fn, exc_type, needle: str):
    try:
        fn()
    except exc_type as error:
        assert needle in str(error), f"expected {needle!r} in {error!r}"
        return
    raise AssertionError(f"expected {exc_type.__name__} mentioning {needle!r}")


def test_config_validation():
    """A bad ``cdps_milp:`` block must fail at parse time, not three hours in."""
    _expect_error(lambda: CdpsMilpConfig.from_dict({"eq_16": True}), ValueError, "eq_16")
    _expect_error(
        lambda: CdpsMilpConfig.from_dict({"gt_anchoring": "all"}), ValueError, "all_gt_states"
    )
    _expect_error(
        lambda: CdpsMilpConfig.from_dict({"eq16": "maybe"}), ValueError, "on, off"
    )

    assert CdpsMilpConfig.from_dict(None) == CdpsMilpConfig(), "empty block = defaults"
    assert CdpsMilpConfig.from_dict({"eq16": "on"}).eq16 is True, "YAML 'on' must parse"
    assert CdpsMilpConfig.from_dict({"eq16": "off"}).eq16 is False, "YAML 'off' must parse"
    print("PASS  config rejects bad keys/values and accepts YAML booleans")


def test_config_derived_settings():
    """The three things the config decides for the run."""
    config = CdpsMilpConfig()

    encoding = config.encoding_config(has_prior=False)
    assert encoding.schema_nonempty is SchemaNonemptyRule.NONE
    assert not encoding.forbid_redundant_adds
    assert not encoding.delete_implies_precondition
    assert encoding.prior_weighting is PriorWeightMode.NONE, "single round has no prior"

    # An unset budget inherits the fold's CDPS budget — that is what makes the
    # head-to-head lower-bound comparison fair.
    assert config.resolve_time_limit(600) == 600
    assert CdpsMilpConfig(time_limit_seconds=30).resolve_time_limit(600) == 30

    assert config.solver_encoder_key() == "cp-sat-observed"
    _expect_error(
        lambda: CdpsMilpConfig(solver=MilpSolver.GUROBI).solver_encoder_key(),
        NotImplementedError,
        "no backend yet",
    )
    print("PASS  config derives the encoding rule set, budget and backend")


# ------------------------------------------------------------- repair log

def test_flip_to_patch_index_mapping():
    """State index -> CDPS's (component, prev/next) addressing.

    State 0 is component 0's ``prev``; state s > 0 is component s-1's ``next``.
    Getting this wrong would misplace every entry of ``milp_repair_log.json``
    relative to CDPS's patch logs, silently breaking every comparison.
    """
    first = FluentFlip(0, 0, "(at r1)", original_is_positive=True)
    later = FluentFlip(2, 3, "(at r2)", original_is_positive=False)

    assert first.as_patch_dict() == {
        "observation_index": 0, "component_index": 0,
        "state_type": "prev", "fluent": "(at r1)",
    }
    assert later.as_patch_dict() == {
        "observation_index": 2, "component_index": 2,
        "state_type": "next", "fluent": "(at r2)",
    }
    assert first.as_dict()["from"] is True and first.as_dict()["to"] is False

    result = ExtractionResult(observations=[], flips=[first, later])
    assert result.repair_cost == 2, "repair cost is the number of observed flips"
    print("PASS  flips map onto CDPS's (component, prev/next) addressing")


if __name__ == "__main__":
    test_dialect_allows_delete_without_precondition()
    test_dialect_allows_delete_only_schema()
    test_dialect_allows_redundant_add()
    test_minimal_repair_flips_exactly_one_fluent()
    test_masked_fluent_is_free()
    test_repeated_args_instance_grounds_reflexive_tuples()
    test_repeated_args_action_unifies_both_bindings()
    test_gt_anchoring_modes()
    test_gt_anchoring_out_of_range_index_ignored()
    test_inferred_object_type_makes_an_action_ungroundable()
    test_object_types_overlay_restores_the_grounding()
    test_object_types_overlay_ignores_unknown_names()
    test_unmatched_action_raises_with_the_type_diagnosis()
    test_try_observation_to_trace_still_returns_none()
    test_config_validation()
    test_config_derived_settings()
    test_flip_to_patch_index_mapping()
    print("ALL TESTS PASSED")
