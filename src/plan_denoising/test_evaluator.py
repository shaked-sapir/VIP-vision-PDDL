"""Unit tests for the ground-truth-free model evaluator.

The properties under test are the ones the PI-SAM+MILP loop actually relies on:

  1. A perfect model on a clean trace scores V = 0.
  2. V counts injected noise exactly (the miniature form of the P3 exit-A check).
  3. V is monotone in model damage -- a worse model never scores better.
  4. Masked slots are never graded and never leak their retained ground-truth polarity.
  5. Inapplicability is *recorded*, not fatal: the rollout keeps going (apply-anyway).
  6. The one-step metric localises damage where the rollout compounds it.

All fixtures are synthetic blocks trajectories written to a temp dir and loaded through
the same path the benchmark uses, so nothing here depends on external data files.
"""

import copy
import logging
import tempfile
import unittest
from pathlib import Path
from typing import List

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Observation

from src.plan_denoising.evaluator import EvaluationWeights, observations_reconstruction_score
from src.utils.masking import load_masked_observation
from src.utils.pddl_state import flip_fluent_in_state

DOMAIN_PDDL = """
(define (domain blocks)
    (:requirements :strips :typing)
    (:types block)
    (:predicates
        (on ?x - block ?y - block)
        (ontable ?x - block)
        (clear ?x - block)
        (handempty)
        (holding ?x - block)
    )
    (:action pick_up
        :parameters (?x - block)
        :precondition (and (clear ?x) (ontable ?x) (handempty))
        :effect (and (not (ontable ?x)) (not (clear ?x)) (not (handempty)) (holding ?x))
    )
    (:action put_down
        :parameters (?x - block)
        :precondition (and (holding ?x))
        :effect (and (not (holding ?x)) (clear ?x) (handempty) (ontable ?x))
    )
    (:action stack
        :parameters (?x - block ?y - block)
        :precondition (and (holding ?x) (clear ?y))
        :effect (and (not (holding ?x)) (not (clear ?y)) (clear ?x) (handempty) (on ?x ?y))
    )
    (:action unstack
        :parameters (?x - block ?y - block)
        :precondition (and (on ?x ?y) (clear ?x) (handempty))
        :effect (and (holding ?x) (clear ?y) (not (clear ?x)) (not (handempty)) (not (on ?x ?y)))
    )
)
"""

CLEAN_STATES = [
    "(on c b) (on b a) (ontable a) (clear c) (handempty)",
    "(holding c) (clear b) (on b a) (ontable a)",
    "(ontable c) (clear c) (handempty) (clear b) (on b a) (ontable a)",
    "(holding b) (clear a) (ontable a) (ontable c) (clear c)",
    "(ontable b) (clear b) (handempty) (clear a) (ontable a) (ontable c) (clear c)",
]
ACTIONS = ["(unstack c b)", "(put_down c)", "(unstack b a)", "(put_down b)"]


def _write_trajectory(dir_path: Path, states: List[str], actions: List[str], name: str) -> Path:
    """Write a ``.trajectory`` file plus an all-empty ``.masking_info`` beside it."""
    lines = ["(", f"(:init {states[0]})"]
    for action, state in zip(actions, states[1:]):
        lines.append(f"(operator: {action})")
        lines.append(f"(:state {state})")
    lines.append(")")
    trajectory_path = dir_path / f"{name}.trajectory"
    trajectory_path.write_text("\n".join(lines) + "\n")
    (dir_path / f"{name}.masking_info").write_text("\n" * len(states))
    return trajectory_path


def _mask_fluent(state, fluent_key: str) -> None:
    """Mark a grounded fluent as masked, keeping its (true) polarity -- exactly what
    ``mask_state`` does, so the tests exercise the real leak hazard."""
    for predicates in state.state_predicates.values():
        for predicate in predicates:
            objects = " ".join(predicate.grounded_objects)
            if f"({predicate.name} {objects})" == fluent_key:
                predicate.is_masked = True
                return

    raise AssertionError(f"fluent {fluent_key} not found in state")


class _EvaluatorTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        logging.disable(logging.WARNING)
        cls._tmp = tempfile.TemporaryDirectory(prefix="evaluator_test_")
        cls.tmp_dir = Path(cls._tmp.name)
        domain_file = cls.tmp_dir / "blocks.pddl"
        domain_file.write_text(DOMAIN_PDDL)
        # partial_parsing=False keeps preconditions and effects -- this is the *model*.
        cls.domain = DomainParser(domain_file, partial_parsing=False).parse_domain()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()
        logging.disable(logging.NOTSET)

    def _observation(self, name: str) -> Observation:
        trajectory = _write_trajectory(self.tmp_dir, CLEAN_STATES, ACTIONS, name)
        return load_masked_observation(
            trajectory, trajectory.with_suffix(".masking_info"), self.domain
        )

    def _damaged_domain(self, action_name: str, effect_representation: str) -> Domain:
        """Return a copy of the model with one discrete effect deleted."""
        damaged = copy.deepcopy(self.domain)
        action = damaged.actions[action_name]
        action.discrete_effects = {
            effect
            for effect in action.discrete_effects
            if effect.untyped_representation != effect_representation
        }
        return damaged


class TestPerfectModel(_EvaluatorTestBase):
    """The ground-truth model on a clean trace must be indistinguishable from perfect."""

    def test_clean_trace_scores_zero(self):
        result = observations_reconstruction_score(self.domain, [self._observation("clean")])
        self.assertEqual(result.v_raw, 0.0)
        self.assertEqual(result.effect_mismatches, 0)
        self.assertEqual(result.inapplicability_events, 0)
        self.assertEqual(result.success_rate, 1.0)
        self.assertEqual(result.one_step_success_rate, 1.0)

    def test_grading_is_two_sided(self):
        """Observations are fully grounded, so negative slots are graded too -- without
        them the metric could never punish a model for *adding* a fluent."""
        result = observations_reconstruction_score(self.domain, [self._observation("clean_slots")])
        # 4 transitions x 19 groundings (6 `on`, 3 `ontable`, 3 `clear`, 1 `handempty`,
        # 3 `holding` ... over 3 blocks) -- the exact count matters less than "all of them".
        self.assertEqual(result.graded_slots, 4 * 19)
        self.assertEqual(result.num_transitions, 4)

    def test_no_observations_is_not_an_error(self):
        result = observations_reconstruction_score(self.domain, [])
        self.assertEqual(result.v_raw, 0.0)
        self.assertEqual(result.num_transitions, 0)
        self.assertEqual(result.success_rate, 1.0)


class TestNoiseCounting(_EvaluatorTestBase):
    """V must equal the amount of injected noise. This is P3 exit-check A in miniature.

    Because the rollout is driven by the *model*, and the model here is ground truth,
    every state the rollout produces is correct. So each flipped observed fluent
    produces exactly one mismatch -- no more, no less.
    """

    def test_single_flip_costs_exactly_one(self):
        observation = self._observation("one_flip")
        flip_fluent_in_state(observation.components[1].next_state, "(clear b)")
        result = observations_reconstruction_score(self.domain, [observation])
        self.assertEqual(result.effect_mismatches, 1)
        self.assertEqual(result.inapplicability_events, 0)
        self.assertEqual(result.v_raw, 1.0)

    def test_v_counts_every_flip(self):
        observation = self._observation("three_flips")
        flip_fluent_in_state(observation.components[0].next_state, "(holding c)")
        flip_fluent_in_state(observation.components[1].next_state, "(clear b)")
        flip_fluent_in_state(observation.components[3].next_state, "(ontable b)")
        result = observations_reconstruction_score(self.domain, [observation])
        self.assertEqual(result.effect_mismatches, 3)
        self.assertEqual(result.v_raw, 3.0)

    def test_flips_are_attributed_to_the_right_trace(self):
        clean = self._observation("multi_clean")
        noisy = self._observation("multi_noisy")
        flip_fluent_in_state(noisy.components[2].next_state, "(clear a)")
        result = observations_reconstruction_score(self.domain, [clean, noisy])
        self.assertEqual(result.per_trace[0].v_raw, 0.0)
        self.assertEqual(result.per_trace[1].v_raw, 1.0)
        self.assertEqual(result.v_raw, 1.0)

    def test_per_transition_normalisation(self):
        observation = self._observation("normalised")
        flip_fluent_in_state(observation.components[1].next_state, "(clear b)")
        result = observations_reconstruction_score(self.domain, [observation])
        self.assertEqual(result.v_per_transition, 1.0 / 4)


class TestDamagedModel(_EvaluatorTestBase):
    """A model that mispredicts must score strictly worse than one that does not."""

    def test_missing_add_effect_is_penalised(self):
        observation = self._observation("damaged_add")
        damaged = self._damaged_domain("unstack", "(holding ?x)")
        baseline = observations_reconstruction_score(self.domain, [observation])
        result = observations_reconstruction_score(damaged, [observation])
        self.assertEqual(baseline.v_raw, 0.0)
        self.assertGreater(result.v_raw, 0.0)

    def test_a_missing_delete_effect_drifts(self):
        """Dropping `(not (on ?x ?y))` from `unstack` leaves `(on c b)` true forever --
        nothing downstream ever deletes it. The chained rollout keeps paying for that
        at every later state, while the one-step view only sees it twice."""
        observation = self._observation("drift")
        damaged = self._damaged_domain("unstack", "(not (on ?x ?y))")
        result = observations_reconstruction_score(damaged, [observation])
        self.assertGreater(result.effect_mismatches, result.one_step_effect_mismatches)

    def test_a_missing_add_effect_surfaces_as_inapplicability(self):
        """Dropping `(holding ?x)` from `unstack` makes the following `put_down`
        inapplicable -- but apply-anyway still applies its effects, which happen to
        restore the state. So this damage shows up in the inapplicability channel
        rather than as compounding effect mismatches, which is precisely why V needs
        both terms."""
        observation = self._observation("inapplicable_drift")
        damaged = self._damaged_domain("unstack", "(holding ?x)")
        result = observations_reconstruction_score(damaged, [observation])
        self.assertEqual(result.effect_mismatches, result.one_step_effect_mismatches)
        self.assertGreater(result.inapplicability_events, 0)
        self.assertGreater(result.v_raw, result.effect_mismatches)

    def test_unknown_action_is_counted_not_crashed(self):
        crippled = copy.deepcopy(self.domain)
        del crippled.actions["put_down"]
        result = observations_reconstruction_score(crippled, [self._observation("unknown_action")])
        # `put_down` is used twice in the trajectory.
        self.assertEqual(result.unknown_actions, 2)
        self.assertGreaterEqual(result.inapplicability_events, 2)


class TestApplyAnyway(_EvaluatorTestBase):
    """Inapplicability is recorded and the rollout continues; it never aborts."""

    def test_rollout_grades_every_transition_despite_failures(self):
        crippled = copy.deepcopy(self.domain)
        # Make `unstack` unsatisfiable from the very first step.
        crippled.actions["unstack"].discrete_effects = set()
        result = observations_reconstruction_score(crippled, [self._observation("apply_anyway")])
        self.assertEqual(result.num_transitions, 4)
        # All four states were still graded -- the rollout did not stop at step 0.
        self.assertEqual(result.graded_slots, 4 * 19)
        self.assertGreater(result.inapplicability_events, 0)

    def test_weights_are_applied(self):
        observation = self._observation("weighted")
        crippled = copy.deepcopy(self.domain)
        crippled.actions["unstack"].discrete_effects = set()
        unweighted = observations_reconstruction_score(crippled, [observation])
        weighted = observations_reconstruction_score(
            crippled,
            [observation],
            weights=EvaluationWeights(effect_mismatch=1.0, inapplicability=10.0),
        )
        expected = unweighted.effect_mismatches + 10.0 * unweighted.inapplicability_events
        self.assertEqual(weighted.v_raw, expected)
        self.assertGreater(weighted.v_raw, unweighted.v_raw)


class TestMasking(_EvaluatorTestBase):
    """Masked slots must be invisible to the metric -- both as evidence and as truth."""

    def test_masked_slot_in_a_next_state_is_not_graded(self):
        observation = self._observation("masked_next")
        flip_fluent_in_state(observation.components[1].next_state, "(clear b)")
        noisy = observations_reconstruction_score(self.domain, [observation])
        self.assertEqual(noisy.v_raw, 1.0)

        _mask_fluent(observation.components[1].next_state, "(clear b)")
        masked = observations_reconstruction_score(self.domain, [observation])
        self.assertEqual(masked.v_raw, 0.0)
        self.assertEqual(masked.graded_slots, noisy.graded_slots - 1)

    def test_masked_init_slot_is_treated_as_false_not_as_its_true_value(self):
        """``mask_state`` leaves ``is_positive`` at the true value. If the rollout read
        that, ground truth would leak into a metric that must be ground-truth free."""
        observation = self._observation("masked_init")
        _mask_fluent(observation.components[0].previous_state, "(handempty )")
        result = observations_reconstruction_score(self.domain, [observation])
        self.assertEqual(result.init_masked_slots, 1)
        # (handempty) was a true precondition of `unstack`; dropping it must be visible.
        self.assertGreater(result.inapplicability_events, 0)

    def test_one_step_skips_transitions_whose_preconditions_are_masked(self):
        observation = self._observation("masked_precondition")
        _mask_fluent(observation.components[0].previous_state, "(on c b)")
        result = observations_reconstruction_score(self.domain, [observation])
        self.assertEqual(result.one_step_skipped_transitions, 1)
        self.assertEqual(result.one_step_effect_mismatches, 0)

    def test_one_step_does_not_blame_the_model_for_a_masked_untouched_fluent(self):
        """`(on b a)` is untouched by `(unstack c b)`. Masking it in the previous state
        makes the predicted next value an artefact of the mask, so that slot must be
        excluded rather than counted as a mismatch."""
        observation = self._observation("masked_frame")
        _mask_fluent(observation.components[0].previous_state, "(on b a)")
        result = observations_reconstruction_score(self.domain, [observation])
        self.assertEqual(result.one_step_skipped_transitions, 0)
        self.assertEqual(result.one_step_effect_mismatches, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
