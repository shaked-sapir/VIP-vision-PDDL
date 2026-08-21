"""Phase 1½ gate: per-problem grounding vs. the upstream one-Instance grounding.

    python -m pytest src/milp/test_grounding_scope_equivalence.py

The ``rosame_milp*`` adapters build one ``PSInstance`` **per problem**
(``converter.build_ps_instance``); upstream ROSAME builds one shared Instance
over the object union of the whole corpus, because its network's proposition
head has a single fixed width. The ICAPS-26 arms follow upstream, so a trace
from a 4-block problem is grounded on the union — and carries propositions over
objects its problem never mentions.

Those extra propositions were predicted to be **inert**: pinned false by the
hard init anchor and the frame axioms, and decoupled from the lifted schema
variables the action model is read off. Inert-ness is what makes the union
grounding a fidelity choice with no measurement cost, so it needs checking
rather than asserting: if a phantom proposition could float, or could force a
lifted variable, the choice would change the numbers and would have to be
registered as a deviation instead.

The setup is one clean 4-block blocksworld trace exercising all four schemas,
grounded twice — on its own four blocks, and on those plus a fifth block that
appears nowhere in it.

Non-vacuity: ``test_the_two_encodings_are_genuinely_different_problems`` shows
the two solves really are two problems, and
``test_the_recovered_model_does_not_depend_on_the_solver_seed`` shows the
equality is a property of the encoding rather than of CP-SAT tie-breaking.

The Phase-1½ half above hands every phantom to the solver at ``eps``, so a
phantom is false because the *observation* says so and the frame argument never
has to carry the result. ``TestPhantomObservationValue`` is the Phase-2.5 half:
it sweeps the phantom rows across the whole ``[eps, 1 - eps]`` range a CV head
could emit, which settles the question for any head rather than for one head's
particular outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import pytest

import src.milp  # noqa: F401  (vendor sys.path bootstrap)

from pddl_plus_parser.lisp_parsers import DomainParser
from planning_structs.instance import Instance as PSInstance
from planning_structs.traces import ObservationP, ObservationT, Traces

from src.milp.converter import build_ps_domain
from src.milp.encoder import CPSATObservedActions
from src.utils.config import load_config

_EPS = 1e-5

PROBLEM_BLOCKS = ("a", "b", "c", "d")
PHANTOM_BLOCK = "e"

#: A clean plan touching every schema: pick_up, stack, unstack, put_down.
_STATES: Sequence[Sequence[Tuple[str, ...]]] = (
    (("ontable", "a"), ("ontable", "b"), ("ontable", "c"), ("ontable", "d"),
     ("clear", "a"), ("clear", "b"), ("clear", "c"), ("clear", "d"), ("handempty",)),
    (("ontable", "b"), ("ontable", "c"), ("ontable", "d"),
     ("clear", "b"), ("clear", "c"), ("clear", "d"), ("holding", "a")),
    (("ontable", "b"), ("ontable", "c"), ("ontable", "d"),
     ("clear", "a"), ("clear", "c"), ("clear", "d"), ("handempty",), ("on", "a", "b")),
    (("ontable", "b"), ("ontable", "c"), ("ontable", "d"),
     ("clear", "b"), ("clear", "c"), ("clear", "d"), ("holding", "a")),
    (("ontable", "a"), ("ontable", "b"), ("ontable", "c"), ("ontable", "d"),
     ("clear", "a"), ("clear", "b"), ("clear", "c"), ("clear", "d"), ("handempty",)),
)
_ACTIONS: Sequence[Tuple[str, ...]] = (
    ("pick_up", "a"), ("stack", "a", "b"), ("unstack", "a", "b"), ("put_down", "a"),
)


# -------------------------------------------------------------- construction


def _ps_domain():
    pddl_domain = DomainParser(
        Path(load_config()["domains"]["blocksworld"]["domain_file"]),
        partial_parsing=False,
    ).parse_domain()
    return build_ps_domain(pddl_domain)


def _instance(ps_domain, block_names: Sequence[str]) -> PSInstance:
    return PSInstance(ps_domain, [(name, "block") for name in block_names])


def _proposition(instance: PSInstance, name: str, args: Sequence[str]):
    return instance.get_proposition(
        instance.domain.get_predicate(name),
        tuple(instance.get_object(a) for a in args),
    )


def _flat_obs_m(ps_domain):
    """Uninformative model prior: 0.5 everywhere carries zero objective weight."""
    pre, add, dele = {}, {}, {}
    for schema in ps_domain.action_schemas:
        for predicate in ps_domain.predicates:
            for binding in ps_domain.predicate_arguments[(schema, predicate)]:
                pre[(schema, predicate, binding)] = 0.5
                add[(schema, predicate, binding)] = 0.5
                dele[(schema, predicate, binding)] = 0.5
    return type("ObsM", (), {"pre": pre, "add": add, "dele": dele})()


def _traces(
    ps_domain,
    instance: PSInstance,
    phantom_prob: Optional[float] = None,
    phantoms: FrozenSet[str] = frozenset(),
) -> Traces:
    """The clean trace, optionally overriding what ``phantoms`` are observed at.

    ``phantom_prob=None`` leaves every proposition at its closed-world value,
    which is the Phase-1½ setup.
    """
    states = [
        {_proposition(instance, entry[0], entry[1:]) for entry in state}
        for state in _STATES
    ]
    actions = [
        instance.get_action(
            instance.domain.get_action_schema(step[0]),
            tuple(instance.get_object(a) for a in step[1:]),
        )
        for step in _ACTIONS
    ]
    assert all(action is not None for action in actions), "trace is ungroundable"

    def observed(p, positives) -> float:
        if phantom_prob is not None and str(p) in phantoms:
            return phantom_prob
        return 1 - _EPS if p in positives else _EPS

    obs_p = {
        t: [ObservationP(p, observed(p, positives)) for p in instance.propositions]
        for t, positives in enumerate(states, start=1)
    }
    trace = ObservationT(
        len(actions),
        sorted(states[0], key=str),
        obs_p,
        sorted(states[-1], key=str),
        obs_a={},
    )
    trace.instance = instance
    trace.actions = {t: a for t, a in enumerate(actions, start=1)}
    return Traces(instance=None, obs_m=_flat_obs_m(ps_domain), obs_t=[trace])


def _solve(
    ps_domain,
    instance: PSInstance,
    phantom_prob: Optional[float] = None,
    phantoms: FrozenSet[str] = frozenset(),
    **solve_kwargs,
) -> CPSATObservedActions:
    encoder = CPSATObservedActions(
        ps_domain,
        _traces(ps_domain, instance, phantom_prob, phantoms),
        {"state", "model"},
    )
    assert encoder.solve(time_limit=60, **solve_kwargs), "the clean trace must be solvable"
    return encoder


def _lifted_model(encoder: CPSATObservedActions) -> Dict[Tuple[str, str, str, Any], int]:
    """The action model keyed by *names*, so two Domains would compare equal too."""
    solution = encoder.action_model_sol()
    return {
        (kind, schema.name, predicate.name, binding): value
        for kind, channel in (
            ("pre", solution.pre), ("add", solution.add), ("dele", solution.dele)
        )
        for (schema, predicate, binding), value in channel.items()
    }


# -------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def ps_domain():
    return _ps_domain()


@pytest.fixture(scope="module")
def instances(ps_domain) -> Tuple[PSInstance, PSInstance]:
    """(per-problem grounding, union grounding)."""
    return (
        _instance(ps_domain, PROBLEM_BLOCKS),
        _instance(ps_domain, (*PROBLEM_BLOCKS, PHANTOM_BLOCK)),
    )


@pytest.fixture(scope="module")
def solved(ps_domain, instances) -> Tuple[CPSATObservedActions, CPSATObservedActions]:
    narrow, wide = instances
    return _solve(ps_domain, narrow), _solve(ps_domain, wide)


def _phantom_propositions(instances) -> Set[str]:
    narrow, wide = instances
    own = {str(p) for p in narrow.propositions}
    return {str(p) for p in wide.propositions} - own


# -------------------------------------------------------------- structure


def test_widening_adds_only_propositions_over_the_phantom_object(instances) -> None:
    """25 -> 36: the eleven extras all mention the block the trace never uses."""
    narrow, wide = instances
    assert len(narrow.propositions) == 25
    assert len(wide.propositions) == 36

    phantoms = _phantom_propositions(instances)
    assert len(phantoms) == 11
    assert all(PHANTOM_BLOCK in name.split("_")[1:] for name in phantoms), sorted(phantoms)


def test_the_lifted_vocabulary_does_not_depend_on_the_grounding(ps_domain) -> None:
    """The decoupling claim, provable without a solver.

    ``predicate_arguments`` is built from the ``Domain``, not the ``Instance``,
    so the 78 lifted variables the action model is read off cannot see how many
    objects were grounded. Everything below is about whether the *solution*
    over those variables moves.
    """
    lifted = [
        (schema.name, predicate.name, binding)
        for schema in ps_domain.action_schemas
        for predicate in ps_domain.predicates
        for binding in ps_domain.predicate_arguments[(schema, predicate)]
    ]
    assert len(lifted) == 26
    assert len(set(lifted)) == len(lifted)


def test_the_two_encodings_are_genuinely_different_problems(solved) -> None:
    """Non-vacuity: the union solve carries a ``hol`` variable per phantom per state."""
    narrow_encoder, wide_encoder = solved
    assert len(narrow_encoder.hol) == 25 * len(_STATES)
    assert len(wide_encoder.hol) == 36 * len(_STATES)


# -------------------------------------------------------------- the gate


def test_the_recovered_action_model_is_identical(solved) -> None:
    """The Phase-1½ gate itself."""
    narrow_model = _lifted_model(solved[0])
    wide_model = _lifted_model(solved[1])

    assert set(narrow_model) == set(wide_model)
    differing = {key for key in narrow_model if narrow_model[key] != wide_model[key]}
    assert not differing, f"{len(differing)} of {len(narrow_model)} lifted variables moved: {sorted(differing)[:10]}"


def test_phantom_propositions_are_false_at_every_step(solved, instances) -> None:
    """The inertness prediction: init pins them false and no frame axiom lifts them."""
    phantoms = _phantom_propositions(instances)
    repaired = solved[1].repaired_states(1)
    floated = [
        (t, str(p)) for t, state in enumerate(repaired) for p in state if str(p) in phantoms
    ]
    assert not floated, f"phantom propositions became true: {floated}"


def test_the_repaired_live_states_are_identical(solved, instances) -> None:
    """The state channel does not move either, not just the model channel."""
    narrow_repaired = solved[0].repaired_states(1)
    wide_repaired = solved[1].repaired_states(1)
    phantoms = _phantom_propositions(instances)

    assert len(narrow_repaired) == len(wide_repaired) == len(_STATES)
    for narrow_state, wide_state in zip(narrow_repaired, wide_repaired):
        assert {str(p) for p in narrow_state} == {
            str(p) for p in wide_state
        } - phantoms


class TestPhantomObservationValue:
    """Phase 2.5: the same question, for any value a CV head could emit.

    A head sized on the corpus union has no way to know that block ``e`` is
    absent from *this* problem, so it emits some value for ``(clear e)``. The
    gate above only ever showed it ``eps``. Sweeping the phantom rows over the
    whole clamped range covers every head, trained or not, rather than one
    head's outputs.

    Non-vacuity: the three sweeping tests below all pass even when the sweep
    never reaches the solver, since they assert that nothing moves.
    ``test_a_confidently_observed_phantom_is_still_paid_for_in_disagreements``
    is what shows it does reach it.
    """

    #: The range `cv_predictions_to_trace` clamps a head's output into.
    VALUES = (_EPS, 0.1, 0.25, 0.5, 0.75, 0.9, 1 - _EPS)

    @staticmethod
    def _swept(ps_domain, instances, value: float) -> CPSATObservedActions:
        _narrow, wide = instances
        return _solve(
            ps_domain, wide, value, frozenset(_phantom_propositions(instances))
        )

    @pytest.mark.parametrize("value", VALUES)
    def test_the_recovered_action_model_never_moves(
        self, ps_domain, instances, solved, value: float
    ) -> None:
        """The gate itself, now for the whole range."""
        baseline = _lifted_model(solved[0])
        model = _lifted_model(self._swept(ps_domain, instances, value))
        differing = {key for key in baseline if baseline[key] != model[key]}
        assert not differing, (
            f"at phantom value {value}, {len(differing)} of {len(baseline)} "
            f"lifted variables moved: {sorted(differing)[:10]}"
        )

    @pytest.mark.parametrize("value", VALUES)
    def test_no_phantom_floats_true_however_confidently_it_is_observed(
        self, ps_domain, instances, value: float
    ) -> None:
        """The hard init and the frame axioms outrank the observation channel.

        No observed action binds ``e``, so nothing can lift a phantom off the
        false the closed-world init pins it to — an observation asserting one at
        ``1 - eps`` is simply overruled.
        """
        phantoms = _phantom_propositions(instances)
        repaired = self._swept(ps_domain, instances, value).repaired_states(1)
        floated = [
            (t, str(p))
            for t, state in enumerate(repaired)
            for p in state
            if str(p) in phantoms
        ]
        assert not floated, f"at phantom value {value}, these became true: {floated}"

    @pytest.mark.parametrize("value", VALUES)
    def test_the_objective_value_does_not_move_either(
        self, ps_domain, instances, solved, value: float
    ) -> None:
        """Not merely bounded — identical.

        Each phantom contributes ``_w(prob, scale) * hol``, and ``hol`` is
        forced to 0, so the term is worth 0 whatever the coefficient. §4.2a
        predicted the phantoms would cost "a constant objective offset"; the
        offset is zero.
        """
        baseline = solved[0].solve_stats["objective_value"]
        swept = self._swept(ps_domain, instances, value).solve_stats["objective_value"]
        assert swept == baseline

    def test_a_confidently_observed_phantom_is_still_paid_for_in_disagreements(
        self, ps_domain, instances
    ) -> None:
        """Where the cost does land, since it is not the objective.

        A phantom asserted true and forced false is one observation bit the
        solution contradicts. Nothing prices it, but anything *counting* flips
        against the observations — a repair magnitude, a state agreement rate —
        sees 11 phantoms x 5 states of them that the per-problem grounding does
        not have.
        """
        phantoms = _phantom_propositions(instances)

        def phantom_disagreements(value: float) -> int:
            encoder = self._swept(ps_domain, instances, value)
            repaired = encoder.repaired_states(1)
            observed = encoder.traces.obs_t[0].obs_p
            return sum(
                1
                for t, ops in observed.items()
                for op in ops
                if str(op.proposition) in phantoms
                and (op.prob > 0.5) != (op.proposition in repaired[t - 1])
            )

        assert phantom_disagreements(_EPS) == 0
        assert phantom_disagreements(1 - _EPS) == len(phantoms) * len(_STATES) == 55


def test_the_recovered_model_does_not_depend_on_the_solver_seed(
    ps_domain, instances, solved
) -> None:
    """Guards the gate: equality above is not an artefact of tie-breaking.

    ``solve`` pins ``random_seed`` so two solves of the same encoding break ties
    identically — which would make the gate pass vacuously if the optimum were
    not otherwise determined. It is: re-solving under other seeds returns the
    same 78 values.
    """
    baseline = _lifted_model(solved[0])
    narrow, _ = instances
    for seed in (7, 1234):
        assert _lifted_model(_solve(ps_domain, narrow, random_seed=seed)) == baseline
