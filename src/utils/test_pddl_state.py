"""Tests for grounding utilities in :mod:`src.utils.pddl_state`.

Focus: type-tag normalisation. ``GroundedPredicate`` hashes on ``str(self)``,
which embeds the parameter type, so a state and the CWA-completion groundings
must agree on how that type is spelled or the completion silently adds a
negative literal beside a present positive.

Depot is the only domain in ``src/domains/`` that can expose this: ``(clear ?x)``
is declared untyped, so its parameter type is the non-leaf ``object`` while its
objects are ``package`` / ``crane`` / ``pile`` / ``truck`` / ``depot``.
"""

import unittest
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import GroundedPredicate, Observation, State

from src.utils.pddl_state import (
    get_state_grounded_predicates,
    ground_observation_completely,
    normalize_predicate_types_in_state,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

DEPOT_DOMAIN = REPO_ROOT / "src/domains/depot/depot.pddl"
DEPOT_PROBLEM = REPO_ROOT / "src/domains/depot/problems/problem6/problem6.pddl"
DEPOT_TRAJECTORY = REPO_ROOT / "src/domains/depot/problems/problem6/problem6.trajectory"

GRIPPER_DOMAIN = REPO_ROOT / "src/domains/gripper/gripper.pddl"
GRIPPER_PROBLEM = REPO_ROOT / "src/domains/gripper/problems/problem6/problem6.pddl"
GRIPPER_TRAJECTORY = REPO_ROOT / "src/domains/gripper/problems/problem6/problem6.trajectory"


def _fluent_polarities(state: State) -> Dict[Tuple[str, Tuple[str, ...]], Set[bool]]:
    """Map each (predicate name, grounded args) to the polarities present in state."""
    polarities = defaultdict(set)
    for predicate in get_state_grounded_predicates(state):
        key = (predicate.name, tuple(predicate.object_mapping.values()))
        polarities[key].add(predicate.is_positive)
    return polarities


def _contradictory_fluents(observation: Observation) -> List[Tuple[int, str, Tuple[str, ...]]]:
    """Every (state index, name, args) asserted in both polarities in one state."""
    states = [observation.components[0].previous_state]
    states += [component.next_state for component in observation.components]

    contradictions = []
    for index, state in enumerate(states):
        for (name, args), polarities in _fluent_polarities(state).items():
            if len(polarities) > 1:
                contradictions.append((index, name, args))
    return contradictions


def _typed_strings(observation: Observation) -> List[Set[str]]:
    """Per-state set of fully typed predicate strings, e.g. '(clear p2 - package)'."""
    states = [observation.components[0].previous_state]
    states += [component.next_state for component in observation.components]
    return [{str(p) for p in get_state_grounded_predicates(state)} for state in states]


class TestTypeTagNormalization(unittest.TestCase):
    """The CWA completion must not depend on which parser convention ran."""

    @classmethod
    def setUpClass(cls):
        cls.depot_domain = DomainParser(DEPOT_DOMAIN, partial_parsing=True).parse_domain()
        cls.depot_problem = ProblemParser(DEPOT_PROBLEM, cls.depot_domain).parse_problem()

    def _depot_grounded(self, with_problem: bool) -> Observation:
        parser = (
            TrajectoryParser(self.depot_domain, self.depot_problem)
            if with_problem
            else TrajectoryParser(partial_domain=self.depot_domain)
        )
        observation = parser.parse_trajectory(DEPOT_TRAJECTORY)
        return ground_observation_completely(self.depot_domain, observation)

    def test_no_contradictions_without_problem_file(self):
        """TrajectoryParser(domain) — the convention the CDPS/MILP path uses."""
        contradictions = _contradictory_fluents(self._depot_grounded(with_problem=False))
        self.assertEqual([], contradictions)

    def test_no_contradictions_with_problem_file(self):
        """TrajectoryParser(domain, problem) — the convention the ROSAME runners use."""
        contradictions = _contradictory_fluents(self._depot_grounded(with_problem=True))
        self.assertEqual([], contradictions)

    def test_both_parser_conventions_agree(self):
        """Both conventions must ground to byte-identical typed states.

        Stronger than the two tests above: it also rules out a fix that merely
        stops the contradiction while leaving the two spellings in circulation.
        """
        self.assertEqual(
            _typed_strings(self._depot_grounded(with_problem=False)),
            _typed_strings(self._depot_grounded(with_problem=True)),
        )

    def test_untyped_parameter_is_tagged_with_the_concrete_type(self):
        """depot's '(clear ?x)' must ground to '- package', not the declared '- object'."""
        grounded = self._depot_grounded(with_problem=False)
        initial_state = grounded.components[0].previous_state
        clear_predicates = {
            str(p)
            for p in get_state_grounded_predicates(initial_state)
            if p.name == "clear" and "p1" in p.object_mapping.values()
        }
        self.assertTrue(clear_predicates, "expected a grounding of (clear p1)")
        for predicate_string in clear_predicates:
            self.assertIn("- package", predicate_string)
            self.assertNotIn("- object", predicate_string)


class TestNormalizationIsSurgical(unittest.TestCase):
    """Re-tagging must touch only parameters declared with a non-leaf type."""

    def test_gripper_states_are_unchanged(self):
        """A flat-hierarchy domain — declared type is already the concrete type."""
        domain = DomainParser(GRIPPER_DOMAIN, partial_parsing=True).parse_domain()
        problem = ProblemParser(GRIPPER_PROBLEM, domain).parse_problem()
        observation = TrajectoryParser(domain, problem).parse_trajectory(GRIPPER_TRAJECTORY)

        for component in observation.components:
            for state in (component.previous_state, component.next_state):
                normalized = normalize_predicate_types_in_state(
                    state, observation.grounded_objects)
                self.assertEqual(
                    {str(p) for p in get_state_grounded_predicates(state)},
                    {str(p) for p in get_state_grounded_predicates(normalized)},
                )

    def test_depot_leaf_typed_predicates_are_unchanged(self):
        """Within depot, only '(clear ?x)' may be re-tagged; the rest must not move."""
        domain = DomainParser(DEPOT_DOMAIN, partial_parsing=True).parse_domain()
        observation = TrajectoryParser(
            partial_domain=domain).parse_trajectory(DEPOT_TRAJECTORY)

        def non_clear(state: State) -> Set[str]:
            return {
                str(p) for p in get_state_grounded_predicates(state) if p.name != "clear"
            }

        for component in observation.components:
            for state in (component.previous_state, component.next_state):
                normalized = normalize_predicate_types_in_state(
                    state, observation.grounded_objects)
                self.assertEqual(non_clear(state), non_clear(normalized))


class TestNormalizationPreservesPredicateFlags(unittest.TestCase):
    """Re-tagging rebuilds predicates, so it must carry every flag across."""

    def setUp(self):
        self.domain = DomainParser(DEPOT_DOMAIN, partial_parsing=True).parse_domain()
        self.problem = ProblemParser(DEPOT_PROBLEM, self.domain).parse_problem()
        self.objects = self.problem.objects

    def _normalize_one(self, predicate: GroundedPredicate) -> GroundedPredicate:
        state = State(predicates={"(clear ?x)": {predicate}}, fluents={})
        normalized = normalize_predicate_types_in_state(state, self.objects)
        return next(iter(get_state_grounded_predicates(normalized)))

    def test_polarity_and_mask_survive(self):
        original = GroundedPredicate(
            name="clear",
            signature={"?x": self.domain.types["object"]},
            object_mapping={"?x": "p1"},
            is_positive=False,
            is_masked=True,
        )
        normalized = self._normalize_one(original)

        self.assertEqual("package", str(normalized.signature["?x"]))
        self.assertEqual(original.object_mapping, normalized.object_mapping)
        self.assertFalse(normalized.is_positive)
        self.assertTrue(normalized.is_masked)

    def test_input_state_is_not_mutated(self):
        original = GroundedPredicate(
            name="clear",
            signature={"?x": self.domain.types["object"]},
            object_mapping={"?x": "p1"},
        )
        state = State(predicates={"(clear ?x)": {original}}, fluents={})
        normalize_predicate_types_in_state(state, self.objects)

        self.assertEqual("object", str(original.signature["?x"]))
        self.assertEqual({original}, state.state_predicates["(clear ?x)"])


if __name__ == "__main__":
    unittest.main()
