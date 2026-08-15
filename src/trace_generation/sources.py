"""Where a trace comes from: walk a problem, or read an existing trajectory.

Every source yields the same :class:`TraceStep` records, so the cutter and the
emitter never branch on the source.

``ProblemWalkSource`` follows the AMLGym procedure -- plan, execute, substitute a
random applicable action with probability ``p_rnd``, replan -- with ``p_rnd``
as the single dial:

    ``p_rnd = 0``    pure planner walk
    ``0 < p_rnd < 1``  mixed walk; no planner is consulted once the goal is hit
    ``p_rnd = 1``    pure random walk; the planner is never invoked and the
                     source problem's goal is irrelevant

``unified_planning`` and ``tarski`` are imported inside the walk, not at module
scope, so importing this package stays cheap for the trajectory source.
"""

from __future__ import annotations

import contextlib
import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from src.trace_generation.trace_step import TraceStep
from src.trace_generation.up_bridge import (
    action_to_eval_string,
    ground_fluents,
    object_types,
    state_to_trace_state,
    typed_objects,
)

logger = logging.getLogger(__name__)

# Same search configuration AMLGym's gen_states_predictability.py uses.
DOWNWARD_SEARCH_CFG = (
    "let(hff,ff(),"
    "let(hcea,cea(),"
    "lazy_greedy([hff,hcea],preferred=[hff,hcea])))"
)

DEFAULT_MAX_PLANNING_TIME = 120
DEFAULT_MAX_REPLANNING_TIME = 60
DEFAULT_MAX_RANDOM_TRIALS = 3
DEFAULT_MAX_STEPS = 1000


class TraceSource(ABC):
    """A trace, plus the metadata the emitter needs to write it out."""

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """The ``(:domain ...)`` name emitted problems must reference."""

    @property
    @abstractmethod
    def objects(self) -> Tuple[str, ...]:
        """The trace's object universe, as eval-schema ``name:type`` strings."""

    @abstractmethod
    def steps(self) -> Iterator[TraceStep]:
        """Yield the trace's steps in order. Pulled lazily by the cutter."""

    @abstractmethod
    def describe(self) -> Dict[str, Any]:
        """The source's parameters, recorded in ``generation_info.json``."""


@dataclass(frozen=True)
class UPWalkStep:
    """One walk transition, still in ``unified_planning`` types."""

    prev_state: Any
    action: Any
    next_state: Any


# ── The walk ─────────────────────────────────────────────────────────────

def walk_problem(
    problem: Any,
    ground_actions: Dict[str, Any],
    *,
    rng: random.Random,
    p_rnd: float,
    max_steps: int,
    preserve_solvability: bool = False,
    stop_at_goal: bool = False,
    max_planning_time: int = DEFAULT_MAX_PLANNING_TIME,
    max_replanning_time: int = DEFAULT_MAX_REPLANNING_TIME,
    max_random_trials: int = DEFAULT_MAX_RANDOM_TRIALS,
) -> Iterator[UPWalkStep]:
    """Walk a parsed UP problem, yielding at most ``max_steps`` transitions.

    Args:
        problem: A ``unified_planning`` ``Problem``.
        ground_actions: Tarski's grounding, as returned by :func:`ground_actions_of`.
        rng: The walk's random source. Independent of the cutter's.
        p_rnd: Probability of substituting a random applicable action for the
            planned one. ``1`` skips planning entirely.
        max_steps: Cap on yielded transitions.
        preserve_solvability: Replan after each random action and keep only
            substitutions that leave the problem solvable. Off by default: a
            corpus walk does not need it and it is expensive.
        stop_at_goal: Stop once the plan runs to completion, instead of
            continuing with random actions up to ``max_steps``.
        max_planning_time: Planner timeout, in seconds.
        max_replanning_time: Solvability-check timeout, in seconds.
        max_random_trials: Random actions tried per substitution when
            ``preserve_solvability`` is set.

    Yields:
        :class:`UPWalkStep` transitions, in order.
    """
    if not 0.0 <= p_rnd <= 1.0:
        raise ValueError(f"p_rnd must be in [0, 1], got {p_rnd}.")
    if max_steps < 0:
        raise ValueError(f"max_steps must be >= 0, got {max_steps}.")
    if max_steps == 0:
        return

    configure_unified_planning()
    from unified_planning.shortcuts import SequentialSimulator

    fluents = ground_fluents(problem)
    resolved = resolve_ground_actions(problem, ground_actions)

    with SequentialSimulator(problem=problem) as simulator:
        state = simulator.get_initial_state()
        if p_rnd >= 1.0:
            yield from _random_walk(simulator, resolved, fluents, state, rng, max_steps)
            return
        yield from _guided_walk(
            simulator, problem, resolved, fluents, state, rng,
            p_rnd=p_rnd, max_steps=max_steps,
            preserve_solvability=preserve_solvability, stop_at_goal=stop_at_goal,
            max_planning_time=max_planning_time,
            max_replanning_time=max_replanning_time,
            max_random_trials=max_random_trials,
        )


def _random_walk(simulator: Any, resolved: Sequence[Tuple[Any, List[Any]]],
                 fluents: Sequence[Any], state: Any, rng: random.Random,
                 max_steps: int) -> Iterator[UPWalkStep]:
    """Apply random state-changing actions until the cap or a dead end."""
    for _ in range(max_steps):
        step = _random_step(simulator, resolved, fluents, state, rng)
        if step is None:
            logger.debug("No applicable state-changing action; walk ends here.")
            return
        yield step
        state = step.next_state


def _guided_walk(
    simulator: Any,
    problem: Any,
    resolved: Sequence[Tuple[Any, List[Any]]],
    fluents: Sequence[Any],
    state: Any,
    rng: random.Random,
    *,
    p_rnd: float,
    max_steps: int,
    preserve_solvability: bool,
    stop_at_goal: bool,
    max_planning_time: int,
    max_replanning_time: int,
    max_random_trials: int,
) -> Iterator[UPWalkStep]:
    """Follow a plan, occasionally substituting a random action."""
    planner_cfg = planner_config(max_planning_time)
    replan_cfg = planner_config(max_replanning_time)

    emitted = 0
    plan = None
    while emitted < max_steps:
        if plan is None:
            plan = _plan_from(problem, state, planner_cfg, max_planning_time)
        if plan is None:
            logger.debug("Planning failed; leaving the guided walk.")
            break

        restarted = False
        for action_instance in plan.actions:
            if emitted >= max_steps:
                return

            if rng.random() < p_rnd:
                step, new_plan = _substitute_random_action(
                    simulator, problem, resolved, fluents, state, rng,
                    preserve_solvability=preserve_solvability,
                    replan_cfg=replan_cfg,
                    max_replanning_time=max_replanning_time,
                    max_random_trials=max_random_trials,
                )
                if step is not None:
                    yield step
                    emitted += 1
                    state = step.next_state
                    plan = new_plan
                    restarted = True
                    break

            next_state = simulator.apply(state, action_instance)
            if next_state is None:
                logger.warning("Planned action %s was not applicable; walk ends here.",
                               action_instance)
                return
            yield UPWalkStep(state, action_instance, next_state)
            emitted += 1
            state = next_state

        if restarted:
            continue

        if stop_at_goal:
            logger.debug("Goal reached; stopping the walk.")
            return
        # Replanning from a goal state returns the empty plan forever.
        yield from _random_walk(simulator, resolved, fluents, state, rng,
                                max_steps - emitted)
        return


def _substitute_random_action(
    simulator: Any,
    problem: Any,
    resolved: Sequence[Tuple[Any, List[Any]]],
    fluents: Sequence[Any],
    state: Any,
    rng: random.Random,
    *,
    preserve_solvability: bool,
    replan_cfg: Dict[str, Any],
    max_replanning_time: int,
    max_random_trials: int,
) -> Tuple[Optional[UPWalkStep], Optional[Any]]:
    """Pick a random applicable action; return it and the plan to follow next.

    A ``None`` plan tells the caller to replan from the resulting state.
    """
    from unified_planning.plans import ActionInstance

    if not preserve_solvability:
        return _random_step(simulator, resolved, fluents, state, rng), None

    for _ in range(max_random_trials):
        applicable = _applicable(simulator, resolved, state)
        if not applicable:
            return None, None
        action, params = rng.choice(applicable)
        instance = ActionInstance(action, params)
        new_plan = _replan(problem, state, instance, replan_cfg, max_replanning_time)
        if new_plan is None:
            continue
        next_state = simulator.apply(state, instance)
        if next_state is not None:
            return UPWalkStep(state, instance, next_state), new_plan

    logger.debug("No feasible random action found; continuing with the plan.")
    return None, None


def _random_step(simulator: Any, resolved: Sequence[Tuple[Any, List[Any]]],
                 fluents: Sequence[Any], state: Any,
                 rng: random.Random) -> Optional[UPWalkStep]:
    """One random applicable action that actually changes the state."""
    from unified_planning.plans import ActionInstance

    candidates = _applicable(simulator, resolved, state)
    rng.shuffle(candidates)
    for action, params in candidates:
        instance = ActionInstance(action, params)
        next_state = simulator.apply(state, instance)
        if next_state is None:
            continue
        if _true_fluents(state, fluents) != _true_fluents(next_state, fluents):
            return UPWalkStep(state, instance, next_state)
    return None


def _applicable(simulator: Any, resolved: Sequence[Tuple[Any, List[Any]]],
                state: Any) -> List[Tuple[Any, List[Any]]]:
    """The subset of ground actions applicable in ``state``."""
    # simulator._is_applicable is private UP API; it is what AMLGym uses and it
    # is coupled to unified_planning 1.3.
    return [(action, params) for action, params in resolved
            if simulator._is_applicable(state, action, params)]


def _true_fluents(state: Any, fluents: Sequence[Any]) -> frozenset:
    """The set of ground fluents holding in ``state``."""
    return frozenset(f for f in fluents if state.get_value(f).is_true())


def _plan_from(problem: Any, state: Any, planner_cfg: Dict[str, Any],
               timeout: int) -> Optional[Any]:
    """Solve the problem from ``state``; ``None`` when no plan was found."""
    from unified_planning.shortcuts import OneshotPlanner

    rooted = problem.clone()
    for fluent in rooted.initial_values:
        rooted.set_initial_value(fluent, state.get_value(fluent))

    with contextlib.redirect_stdout(open(os.devnull, "w")):
        with OneshotPlanner(problem_kind=rooted.kind, **planner_cfg) as planner:
            return planner.solve(rooted, timeout=timeout).plan


def _replan(problem: Any, state: Any, instance: Any, planner_cfg: Dict[str, Any],
            timeout: int) -> Optional[Any]:
    """The plan that remains after applying ``instance``; ``None`` if unsolvable."""
    from unified_planning.shortcuts import SequentialSimulator

    rooted = problem.clone()
    for fluent in rooted.initial_values:
        rooted.set_initial_value(fluent, state.get_value(fluent))

    with SequentialSimulator(problem=rooted) as sim:
        next_state = sim.apply(state, instance)
    if next_state is None:
        return None

    return _plan_from(problem, next_state, planner_cfg, timeout)


# ── unified_planning / tarski entry points ───────────────────────────────

def configure_unified_planning() -> None:
    """Silence engine credits and stop successor states from chaining ancestors.

    ``MAX_ANCESTORS = None`` makes every successor condense into an ancestor-free
    state. Such a state still holds only its non-default assignments, so it must
    be read with ``get_value`` rather than by inspecting its own mapping.
    """
    import unified_planning.shortcuts
    from unified_planning.model import UPState

    unified_planning.shortcuts.get_environment().credits_stream = None
    UPState.MAX_ANCESTORS = None


def planner_config(timeout: int) -> Dict[str, Any]:
    """The Fast Downward configuration used for every plan and replan."""
    return {
        "name": "fast-downward",
        "params": {
            "fast_downward_search_config": DOWNWARD_SEARCH_CFG,
            "fast_downward_search_time_limit": f"{timeout}s",
        },
    }


def parse_problem(domain_file: Path, problem_file: Path) -> Any:
    """Parse a domain and problem into a ``unified_planning`` ``Problem``."""
    from unified_planning.io import PDDLReader

    configure_unified_planning()
    return PDDLReader().parse_problem(str(domain_file), str(problem_file))


def ground_actions_of(domain_file: Path, problem_file: Path) -> Dict[str, Any]:
    """Ground every action of a problem with tarski's LP grounder."""
    from tarski.grounding import LPGroundingStrategy
    from tarski.io import PDDLReader as TarskiPDDLReader

    reader = TarskiPDDLReader(raise_on_error=True)
    reader.parse_domain(str(domain_file))
    reader.parse_instance(str(problem_file))
    return LPGroundingStrategy(reader.problem).ground_actions()


def resolve_ground_actions(problem: Any,
                           ground_actions: Dict[str, Any]) -> List[Tuple[Any, List[Any]]]:
    """Turn tarski's grounding into UP ``(action, parameters)`` pairs, sorted.

    Sorted so the applicable-action list an RNG draws from does not depend on
    dictionary iteration order.
    """
    resolved: List[Tuple[Any, List[Any]]] = []
    for action_name, parameter_tuples in ground_actions.items():
        action = problem.action(action_name.lower())
        for objects in parameter_tuples:
            resolved.append((action, [problem.object(o.lower()) for o in objects]))
    return sorted(resolved, key=lambda pair: f"{pair[0]} - {pair[1]}")


# ── Sources ──────────────────────────────────────────────────────────────

class ProblemWalkSource(TraceSource):
    """A trace produced by walking a PDDL problem with a planner and/or randomly."""

    def __init__(
        self,
        problem_file: Path,
        domain_file: Path,
        *,
        backend: str = "native",
        p_rnd: float = 1.0,
        seed: Optional[int] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        preserve_solvability: bool = False,
        stop_at_goal: bool = False,
        max_planning_time: int = DEFAULT_MAX_PLANNING_TIME,
        max_replanning_time: int = DEFAULT_MAX_REPLANNING_TIME,
        max_random_trials: int = DEFAULT_MAX_RANDOM_TRIALS,
    ) -> None:
        """Configure the walk. Nothing is parsed until :meth:`steps` is called.

        Args:
            problem_file: The PDDL problem to walk.
            domain_file: Its domain.
            backend: ``"native"``. ``"pddlgym"`` is the renderer-only backend and
                is not implemented here.
            p_rnd: Probability of a random action. ``1`` skips the planner, so a
                problem with an unreachable or trivial goal is still walkable.
            seed: Seed for the walk RNG. The cutter is seeded separately.
            max_steps: Cap on walked transitions.
            preserve_solvability: See :func:`walk_problem`.
            stop_at_goal: See :func:`walk_problem`.
            max_planning_time: Planner timeout, in seconds.
            max_replanning_time: Solvability-check timeout, in seconds.
            max_random_trials: See :func:`walk_problem`.
        """
        if backend != "native":
            raise NotImplementedError(
                f"backend={backend!r} is not available; only 'native' is implemented. "
                "The pddlgym backend renders images and stays in "
                "src/trajectory_handlers/pddlgym_problem_generator.py.")
        if not 0.0 <= p_rnd <= 1.0:
            raise ValueError(f"p_rnd must be in [0, 1], got {p_rnd}.")

        self.problem_file = Path(problem_file)
        self.domain_file = Path(domain_file)
        self.backend = backend
        self.p_rnd = p_rnd
        self.seed = seed
        self.max_steps = max_steps
        self.preserve_solvability = preserve_solvability
        self.stop_at_goal = stop_at_goal
        self.max_planning_time = max_planning_time
        self.max_replanning_time = max_replanning_time
        self.max_random_trials = max_random_trials

        self._problem: Optional[Any] = None
        self._types: Optional[Dict[str, str]] = None
        self._objects: Optional[Tuple[str, ...]] = None

    @property
    def domain_name(self) -> str:
        """The domain name declared in the domain file."""
        from pddl_plus_parser.lisp_parsers import DomainParser

        return DomainParser(self.domain_file, partial_parsing=True).parse_domain().name

    @property
    def objects(self) -> Tuple[str, ...]:
        """The problem's object universe; constant for the whole walk."""
        self._load()
        return self._objects  # type: ignore[return-value]

    def steps(self) -> Iterator[TraceStep]:
        """Walk the problem, converting each transition into the eval schema."""
        self._load()
        problem = self._problem
        types, objects = self._types, self._objects
        fluents = ground_fluents(problem)

        walk = walk_problem(
            problem,
            ground_actions_of(self.domain_file, self.problem_file),
            rng=random.Random(self.seed),
            p_rnd=self.p_rnd,
            max_steps=self.max_steps,
            preserve_solvability=self.preserve_solvability,
            stop_at_goal=self.stop_at_goal,
            max_planning_time=self.max_planning_time,
            max_replanning_time=self.max_replanning_time,
            max_random_trials=self.max_random_trials,
        )
        for step in walk:
            yield TraceStep(
                prev_state=state_to_trace_state(step.prev_state, fluents, types, objects),
                action=action_to_eval_string(step.action, types),
                next_state=state_to_trace_state(step.next_state, fluents, types, objects),
            )

    def describe(self) -> Dict[str, Any]:
        """The walk's parameters, for ``generation_info.json``."""
        return {
            "source_kind": "problem",
            "source_file": str(self.problem_file),
            "domain_file": str(self.domain_file),
            "backend": self.backend,
            "p_rnd": self.p_rnd,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "preserve_solvability": self.preserve_solvability,
            "stop_at_goal": self.stop_at_goal,
        }

    def _load(self) -> None:
        """Parse the problem once, caching its types and object universe."""
        if self._problem is not None:
            return
        self._problem = parse_problem(self.domain_file, self.problem_file)
        self._types = object_types(self._problem)
        self._objects = typed_objects(self._problem)
