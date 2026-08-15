"""Turn a learned action model into the MILP's reference-model channel.

The loop (``docs/pisam-milp-loop-plan.md`` line 100) solves round *r* as
``MILP(S_r, prior=M_best, ...)``: the incumbent model biases the repair so that
consecutive rounds do not wander between equally-cheap explanations. The encoder
reads that prior from ``traces.obs_m`` — an :class:`ObservationM` of three dicts
keyed ``(ActionSchema, Predicate, binding)`` — so a ``LearnerDomain`` has to be
projected onto that vocabulary first.

``benchmark/algorithm_adapters/rosame_milp/model_bridge.py`` does the same job for
ROSAME, but ``src`` must not import from ``benchmark``, and the input is a
different object (a network's probabilities vs. a symbolic model), so the two
share no code.

Two properties of the encoder's vocabulary decide everything here:

1. **Bindings are tuples of distinct 1-based action-parameter positions**
   (``planning_structs.domain.build_predicate_arguments`` uses ``permutations``).
   A learned literal that binds one action parameter to two predicate slots —
   ``(on ?x ?x)`` — or that mentions a domain constant has no variable in the
   encoding and therefore cannot be expressed.
2. **``pre`` has no polarity.** There is one boolean per ``(a, p, x)`` meaning "is
   a precondition"; a *negative* precondition is inexpressible.

Neither is silently ignored: both are counted in :class:`PriorProjection.stats`
so a run can show how much of ``M_best`` actually reached the solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

from pddl_plus_parser.models import Predicate

from planning_structs.traces import ObservationM

# Probability of a slot the incumbent model does not declare.
#
# The encoder prices a slot by ``_w(prob) = round((2*prob - 1) * scale)``, so 0.5
# is "no opinion" and 0.0 is an active penalty. We use 0.5 deliberately: PI-SAM
# returns a *partial* model, in which an absent slot means "never justified by the
# data", not "known to be absent". Pricing absence at 0.0 would let the prior
# argue against bits the incumbent was merely silent about. The prior may
# therefore only ever say "these bits worked before" — never "these bits are bad".
NEUTRAL_PROBABILITY = 0.5
PRESENT_PROBABILITY = 1.0


@dataclass
class PriorProjection:
    """``M_best`` expressed in the encoder's vocabulary, plus what did not fit.

    Attributes:
        observation_m: The prior, ready to hand to ``Traces(obs_m=...)``.
        stats: Counts of the literals that could not be represented. All-zero is
            the healthy case; non-zero values are not errors, but they do mean
            the prior is a lossy view of ``M_best``.
    """

    observation_m: ObservationM
    stats: Dict[str, int] = field(default_factory=dict)

    @property
    def is_lossless(self) -> bool:
        """Whether every literal of the source model reached the encoding."""
        return not any(count for key, count in self.stats.items() if key != "slots")


def _position_map(signature: Mapping[str, Any]) -> Dict[str, int]:
    """Action parameter name -> its 1-based position, the encoder's convention."""
    return {name: index for index, name in enumerate(signature, start=1)}


def _binding(predicate: Predicate, positions: Mapping[str, int]) -> Optional[Tuple[int, ...]]:
    """The predicate's argument tuple as action-parameter positions.

    ``None`` when the literal cannot be a key, because it mentions something
    that is not an action parameter — a domain constant.

    The distinctness check is defence only. ``build_predicate_arguments``
    enumerates permutations of *distinct* slots, so a literal repeating one
    parameter — ``(on ?x ?x)`` — would indeed have no key; but it cannot reach
    here, since ``Predicate.signature`` is a dict keyed by argument name and the
    repeat has already collapsed to a single argument upstream.
    """
    try:
        binding = tuple(positions[argument] for argument in predicate.signature)
    except KeyError:
        return None
    return binding if len(set(binding)) == len(binding) else None


def _flatten_preconditions(preconditions) -> Iterator[Predicate]:
    """The plain predicates of a precondition tree.

    ``CompoundPrecondition`` iteration already flattens nested ``Precondition``
    nodes into ``(operator, operand)`` pairs; anything that is not a plain
    ``Predicate`` (a universal quantifier, a numeric expression) has no place in
    this encoding and is dropped by the caller, which counts it.
    """
    for _operator, condition in preconditions:
        yield condition


def learner_domain_to_observation_m(learned_domain, ps_domain) -> PriorProjection:
    """Project a learned model onto the encoder's ``(schema, predicate, binding)`` grid.

    Args:
        learned_domain: PI-SAM's model — anything exposing ``actions`` whose values
            carry ``signature``, ``preconditions`` and ``discrete_effects``.
        ps_domain: The vendored domain the encoder was built from
            (``converter.build_ps_domain``), which owns the key vocabulary.

    Returns:
        A :class:`PriorProjection`. Slots the model does not declare are left at
        :data:`NEUTRAL_PROBABILITY`, so the prior expresses only what the model
        asserts.
    """
    pre: Dict[Any, float] = {}
    add: Dict[Any, float] = {}
    dele: Dict[Any, float] = {}
    for schema in ps_domain.action_schemas:
        for predicate in ps_domain.predicates:
            for binding in ps_domain.predicate_arguments[(schema, predicate)]:
                key = (schema, predicate, binding)
                pre[key] = add[key] = dele[key] = NEUTRAL_PROBABILITY

    stats = {
        "slots": len(pre),
        "unknown_actions": 0,
        "unknown_predicates": 0,
        "negative_preconditions_dropped": 0,
        "unbindable_literals": 0,
        "non_predicate_preconditions": 0,
    }

    for action in learned_domain.actions.values():
        schema = ps_domain.get_action_schema(action.name)
        if schema is None:
            stats["unknown_actions"] += 1
            continue
        positions = _position_map(action.signature)

        def assign(literal: Predicate, table: Dict[Any, float]) -> None:
            predicate = ps_domain.get_predicate(literal.name)
            if predicate is None:
                stats["unknown_predicates"] += 1
                return
            binding = _binding(literal, positions)
            key = (schema, predicate, binding)
            if binding is None or key not in table:
                stats["unbindable_literals"] += 1
                return
            table[key] = PRESENT_PROBABILITY

        for condition in _flatten_preconditions(action.preconditions):
            if not isinstance(condition, Predicate):
                stats["non_predicate_preconditions"] += 1
                continue
            if not condition.is_positive:
                # The encoding has one polarity-free "is a precondition" bit; a
                # negative precondition has nowhere to go. See module docstring.
                stats["negative_preconditions_dropped"] += 1
                continue
            assign(condition, pre)

        for effect in action.discrete_effects:
            assign(effect, add if effect.is_positive else dele)

    return PriorProjection(ObservationM(pre, add, dele), stats)
