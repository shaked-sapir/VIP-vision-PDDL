"""Ground-truth-free scoring of a learned action model against the observations it came from.

This module implements the ``Evaluate`` contract of the CDPS+MILP loop
(``docs/cdps-milp-loop-plan.md`` §2.1). It answers a single question:

    *How well does candidate model ``M`` reproduce the trajectories we actually observed?*

The score ``V`` is deliberately **ground-truth free**: the reference is the frozen
original (noisy, partially observed) trajectories, never the simulator's ground-truth
domain and never planner-generated states. That is what makes it usable as the loop's
selection signal at runtime, where no ground truth exists.

Two complementary numbers are produced:

``rollout``
    Execute the recorded action sequence from ``s0`` under ``M``, chaining the model's
    own output forward. Drift accumulates, so this is the sensitive measure.

``one_step``
    Apply each action to its *observed* previous state independently. No drift, so this
    localises which transitions the model gets wrong.

Both use **apply-anyway** semantics: when an action's preconditions do not hold we record
an inapplicability event and apply the effects regardless. Stopping the rollout instead
would make ``V`` saturate on the first failure and lose all power to rank candidates --
and since PI-SAM is a *safe* learner it over-approximates preconditions, so spurious
inapplicability is the expected failure mode of *every* candidate, not a disqualifier.

    V = w_effect * effect_mismatches + w_inapplicable * inapplicability_events
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from pddl_plus_parser.models import (
    Domain,
    GroundedPredicate,
    Observation,
    ObservedComponent,
    Operator,
    State,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EvaluationWeights",
    "TraceEvaluation",
    "EvaluationResult",
    "observations_reconstruction_score",
]


# --------------------------------------------------------------------------------------
# Configuration & result containers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationWeights:
    """Relative cost of each kind of disagreement between a model and the observations.

    Defaults are 1.0/1.0 per the plan: an unexplained effect and an unexplained
    inapplicability are treated as equally bad until we have evidence otherwise.
    """

    effect_mismatch: float = 1.0
    inapplicability: float = 1.0


@dataclass
class TraceEvaluation:
    """Per-trajectory breakdown of the evaluation."""

    observation_index: int
    num_transitions: int = 0

    # --- rollout (chained) ---
    effect_mismatches: int = 0
    inapplicability_events: int = 0
    mismatched_transitions: int = 0
    graded_slots: int = 0

    # --- one-step (independent) ---
    one_step_effect_mismatches: int = 0
    one_step_inapplicability_events: int = 0
    one_step_mismatched_transitions: int = 0
    one_step_graded_slots: int = 0
    one_step_skipped_transitions: int = 0

    # --- hygiene counters ---
    init_masked_slots: int = 0
    unknown_actions: int = 0

    weights: EvaluationWeights = field(default_factory=EvaluationWeights)

    @property
    def v_raw(self) -> float:
        """The unnormalised score for this trace."""
        return (
            self.weights.effect_mismatch * self.effect_mismatches
            + self.weights.inapplicability * self.inapplicability_events
        )

    @property
    def v_per_transition(self) -> float:
        """``v_raw`` divided by trace length, so long traces do not dominate."""
        if self.num_transitions == 0:
            return 0.0
        return self.v_raw / self.num_transitions

    @property
    def success_rate(self) -> float:
        """Fraction of transitions the model reproduced exactly (rollout)."""
        if self.num_transitions == 0:
            return 1.0
        return 1.0 - (self.mismatched_transitions / self.num_transitions)

    @property
    def one_step_success_rate(self) -> float:
        """Fraction of *gradeable* transitions the model reproduced exactly (one-step)."""
        graded = self.num_transitions - self.one_step_skipped_transitions
        if graded <= 0:
            return 1.0
        return 1.0 - (self.one_step_mismatched_transitions / graded)


@dataclass
class EvaluationResult:
    """Aggregate evaluation of one model over one set of observations."""

    per_trace: List[TraceEvaluation] = field(default_factory=list)
    weights: EvaluationWeights = field(default_factory=EvaluationWeights)

    def _sum(self, attribute: str) -> int:
        return sum(getattr(trace, attribute) for trace in self.per_trace)

    @property
    def num_transitions(self) -> int:
        return self._sum("num_transitions")

    @property
    def effect_mismatches(self) -> int:
        return self._sum("effect_mismatches")

    @property
    def inapplicability_events(self) -> int:
        return self._sum("inapplicability_events")

    @property
    def mismatched_transitions(self) -> int:
        return self._sum("mismatched_transitions")

    @property
    def graded_slots(self) -> int:
        return self._sum("graded_slots")

    @property
    def init_masked_slots(self) -> int:
        return self._sum("init_masked_slots")

    @property
    def unknown_actions(self) -> int:
        return self._sum("unknown_actions")

    @property
    def one_step_effect_mismatches(self) -> int:
        return self._sum("one_step_effect_mismatches")

    @property
    def one_step_inapplicability_events(self) -> int:
        return self._sum("one_step_inapplicability_events")

    @property
    def one_step_skipped_transitions(self) -> int:
        return self._sum("one_step_skipped_transitions")

    @property
    def v_raw(self) -> float:
        """The loop's selection signal. Lower is better; 0 means the model explains
        every observed transition exactly."""
        return (
            self.weights.effect_mismatch * self.effect_mismatches
            + self.weights.inapplicability * self.inapplicability_events
        )

    @property
    def v_per_transition(self) -> float:
        """Length-normalised ``v_raw``, for comparing across folds/domains."""
        if self.num_transitions == 0:
            return 0.0
        return self.v_raw / self.num_transitions

    @property
    def success_rate(self) -> float:
        if self.num_transitions == 0:
            return 1.0
        return 1.0 - (self.mismatched_transitions / self.num_transitions)

    @property
    def one_step_success_rate(self) -> float:
        graded = self.num_transitions - self.one_step_skipped_transitions
        if graded <= 0:
            return 1.0
        mismatched = self._sum("one_step_mismatched_transitions")
        return 1.0 - (mismatched / graded)

    def to_dict(self) -> Dict[str, object]:
        """Flatten to a JSON-serialisable dict for logging into fold results."""
        return {
            "v_raw": self.v_raw,
            "v_per_transition": self.v_per_transition,
            "effect_mismatches": self.effect_mismatches,
            "inapplicability_events": self.inapplicability_events,
            "mismatched_transitions": self.mismatched_transitions,
            "num_transitions": self.num_transitions,
            "graded_slots": self.graded_slots,
            "success_rate": self.success_rate,
            "one_step_success_rate": self.one_step_success_rate,
            "one_step_effect_mismatches": self.one_step_effect_mismatches,
            "one_step_inapplicability_events": self.one_step_inapplicability_events,
            "one_step_skipped_transitions": self.one_step_skipped_transitions,
            "init_masked_slots": self.init_masked_slots,
            "unknown_actions": self.unknown_actions,
            "weights": {
                "effect_mismatch": self.weights.effect_mismatch,
                "inapplicability": self.weights.inapplicability,
            },
            "per_trace": [
                {
                    "observation_index": trace.observation_index,
                    "num_transitions": trace.num_transitions,
                    "v_raw": trace.v_raw,
                    "v_per_transition": trace.v_per_transition,
                    "effect_mismatches": trace.effect_mismatches,
                    "inapplicability_events": trace.inapplicability_events,
                    "success_rate": trace.success_rate,
                    "one_step_success_rate": trace.one_step_success_rate,
                    "one_step_skipped_transitions": trace.one_step_skipped_transitions,
                    "init_masked_slots": trace.init_masked_slots,
                }
                for trace in self.per_trace
            ],
        }


# --------------------------------------------------------------------------------------
# State helpers
# --------------------------------------------------------------------------------------
#
# NOTE ON STATE REPRESENTATION -- this is the single most important detail here.
#
# ``Operator.apply`` and ``GroundedPrecondition.is_applicable`` both assume a
# **positive-only, closed-world** state: delete effects *remove* a predicate from the
# set, and a negative precondition ``(not p)`` holds iff the *positive* form ``p`` is
# absent. Worse, the applicability test is a substring check against
# ``state.serialize()``, so an explicit ``(not (on a b))`` literal in the state would
# make the positive precondition ``(on a b)`` spuriously satisfied.
#
# Our *observation* states are the opposite: fully grounded, carrying an explicit
# literal (positive or negative) for every grounding plus an ``is_masked`` flag.
#
# So we never hand an observation state to the operator. We project it down to a
# positive-only state first, and we grade by comparing observed literals against the
# model state's positive fluent set.


def _fluent_key(predicate: GroundedPredicate) -> str:
    """The polarity-independent identity of a grounded fluent, e.g. ``(on a b)``."""
    objects = " ".join(predicate.grounded_objects)
    return f"({predicate.name} {objects})"


def _project_to_positive_state(state: State, is_init: bool = False) -> Tuple[State, int]:
    """Project an observation state onto the positive-only form the operator expects.

    A masked slot is treated as **False** (i.e. dropped), never as its retained
    polarity. ``mask_state`` sets ``is_masked=True`` but leaves ``is_positive`` at the
    true value, so reading a masked predicate's polarity would silently leak ground
    truth into a metric that must be ground-truth free.

    :param state: an observation state, possibly fully grounded and possibly masked.
    :param is_init: whether the projected state should be flagged as an initial state.
    :return: the positive-only state, and how many masked slots were dropped.
    """
    projected: Dict[str, Set[GroundedPredicate]] = {}
    masked_slots = 0
    for lifted_representation, predicates in state.state_predicates.items():
        kept = set()
        for predicate in predicates:
            if predicate.is_masked:
                masked_slots += 1
                continue
            if predicate.is_positive:
                kept.add(predicate.copy())
        if kept:
            projected[lifted_representation] = kept

    fluents = {name: fluent.copy() for name, fluent in state.state_fluents.items()}
    return State(predicates=projected, fluents=fluents, is_init=is_init), masked_slots


def _positive_fluent_keys(state: State) -> Set[str]:
    """The set of fluents that hold in a positive-only model state."""
    return {
        _fluent_key(predicate)
        for predicates in state.state_predicates.values()
        for predicate in predicates
        if predicate.is_positive
    }


def _grade_state(
    model_state: State,
    observed_state: State,
    gradeable_keys: Optional[Set[str]] = None,
) -> Tuple[int, int]:
    """Count how many observed fluent slots the model got wrong.

    Only **unmasked** observed literals are graded -- masked slots carry no information
    we are allowed to use. Under the closed-world reading of ``model_state``, a fluent
    is predicted true iff it is present as a positive literal.

    :param model_state: the positive-only state the model produced.
    :param observed_state: the reference observation state.
    :param gradeable_keys: if given, restrict grading to these fluent keys.
    :return: ``(mismatches, graded_slots)``.
    """
    predicted_true = _positive_fluent_keys(model_state)
    mismatches = 0
    graded = 0
    for predicates in observed_state.state_predicates.values():
        for predicate in predicates:
            if predicate.is_masked:
                continue
            key = _fluent_key(predicate)
            if gradeable_keys is not None and key not in gradeable_keys:
                continue
            graded += 1
            if (key in predicted_true) != predicate.is_positive:
                mismatches += 1

    return mismatches, graded


# --------------------------------------------------------------------------------------
# Operator helpers
# --------------------------------------------------------------------------------------


def _build_operator(
    domain: Domain, component: ObservedComponent, observation: Observation
) -> Optional[Operator]:
    """Build a grounded operator for a component's action, or ``None`` if the model
    does not define that action at all."""
    call = component.grounded_action_call
    action = domain.actions.get(call.name)
    if action is None:
        logger.debug("Model does not define action %s - counting as inapplicable.", call.name)
        return None

    operator = Operator(
        action=action,
        domain=domain,
        grounded_action_call=list(call.parameters),
        problem_objects=observation.grounded_objects,
    )
    operator.ground()
    return operator


def _precondition_fluent_keys(operator: Operator) -> Set[str]:
    """The grounded fluents the operator's preconditions read."""
    keys = set()
    for _, condition in operator.grounded_preconditions:
        if isinstance(condition, GroundedPredicate):
            keys.add(_fluent_key(condition))

    return keys


def _effect_fluent_keys(operator: Operator) -> Set[str]:
    """The grounded fluents the operator's effects write."""
    keys = set()
    for effect in operator.grounded_effects:
        for predicate in effect.grounded_discrete_effects:
            keys.add(_fluent_key(predicate))

    return keys


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


def _evaluate_rollout(
    domain: Domain,
    observation: Observation,
    operators: Sequence[Optional[Operator]],
    trace: TraceEvaluation,
) -> None:
    """Chain the model forward from ``s0`` and grade every state it produces."""
    if not observation.components:
        return

    current_state, trace.init_masked_slots = _project_to_positive_state(
        observation.components[0].previous_state, is_init=True
    )
    if trace.init_masked_slots:
        logger.debug(
            "Trace %d: %d masked slots in s0 were treated as false.",
            trace.observation_index,
            trace.init_masked_slots,
        )

    for component, operator in zip(observation.components, operators):
        inapplicable = False
        if operator is None:
            # The model cannot express this action; the state simply does not advance.
            inapplicable = True
        else:
            if not operator.is_applicable(current_state):
                inapplicable = True
            current_state = operator.apply(current_state, allow_inapplicable_actions=True)

        mismatches, graded = _grade_state(current_state, component.next_state)
        trace.effect_mismatches += mismatches
        trace.graded_slots += graded
        if inapplicable:
            trace.inapplicability_events += 1
        if mismatches or inapplicable:
            trace.mismatched_transitions += 1


def _evaluate_one_step(
    domain: Domain,
    observation: Observation,
    operators: Sequence[Optional[Operator]],
    trace: TraceEvaluation,
) -> None:
    """Grade each transition independently from its own observed previous state.

    This is the only place where execution meets masking, so two guards apply:

    * If any fluent the preconditions read is masked in the previous state, the
      applicability verdict would be an artefact of the mask -- skip the transition.
    * A fluent the action does not write keeps its previous value, so if that previous
      value was masked the predicted value is an artefact too -- exclude that slot.
    """
    for component, operator in zip(observation.components, operators):
        if operator is None:
            trace.one_step_inapplicability_events += 1
            trace.one_step_mismatched_transitions += 1
            continue

        previous_state, _ = _project_to_positive_state(component.previous_state)
        masked_keys = {
            _fluent_key(predicate)
            for predicates in component.previous_state.state_predicates.values()
            for predicate in predicates
            if predicate.is_masked
        }

        if masked_keys & _precondition_fluent_keys(operator):
            trace.one_step_skipped_transitions += 1
            continue

        written_keys = _effect_fluent_keys(operator)
        gradeable_keys = None
        if masked_keys:
            # Everything except "untouched fluent whose previous value was masked".
            leaked_keys = masked_keys - written_keys
            gradeable_keys = {
                _fluent_key(predicate)
                for predicates in component.next_state.state_predicates.values()
                for predicate in predicates
                if _fluent_key(predicate) not in leaked_keys
            }

        inapplicable = not operator.is_applicable(previous_state)
        next_state = operator.apply(previous_state, allow_inapplicable_actions=True)
        mismatches, graded = _grade_state(next_state, component.next_state, gradeable_keys)

        trace.one_step_effect_mismatches += mismatches
        trace.one_step_graded_slots += graded
        if inapplicable:
            trace.one_step_inapplicability_events += 1
        if mismatches or inapplicable:
            trace.one_step_mismatched_transitions += 1


def observations_reconstruction_score(
    domain: Domain,
    observations: Sequence[Observation],
    weights: Optional[EvaluationWeights] = None,
) -> EvaluationResult:
    """Score a learned action model against the observations it is meant to explain.

    Ground-truth-free by construction: the only reference is ``observations``. Do not
    confuse this with ``benchmark.experiment_running_helpers.evaluation.evaluate_model``,
    which scores against the ground-truth domain and is for offline reporting only --
    using that one as a selection signal would silently invalidate the experiment.

    :param domain: the candidate action model. Only its actions are used; the
        observations supply the objects.
    :param observations: the frozen reference trajectories. These must be the
        **original** observations, not repaired ones -- scoring a model against the very
        repairs that produced it would be circular.
    :param weights: relative cost of effect mismatches vs. inapplicability events.
    :return: the aggregate result, with a per-trace breakdown.
    """
    weights = weights or EvaluationWeights()
    result = EvaluationResult(weights=weights)

    for observation_index, observation in enumerate(observations):
        trace = TraceEvaluation(
            observation_index=observation_index,
            num_transitions=len(observation.components),
            weights=weights,
        )

        operators = [
            _build_operator(domain, component, observation)
            for component in observation.components
        ]
        trace.unknown_actions = sum(1 for operator in operators if operator is None)

        _evaluate_rollout(domain, observation, operators, trace)
        _evaluate_one_step(domain, observation, operators, trace)
        result.per_trace.append(trace)

    logger.debug(
        "Evaluated model over %d traces: V=%.2f (%d effect mismatches, %d inapplicability events).",
        len(result.per_trace),
        result.v_raw,
        result.effect_mismatches,
        result.inapplicability_events,
    )
    return result
