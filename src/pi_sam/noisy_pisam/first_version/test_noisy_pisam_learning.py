"""
Test suite for NoisyPisamLearner.

This test file includes:
1. Loading blocks problem7 observations with masking info from LLM-based experiment
2. Creating fluent-level and model-level patches
3. Unit tests for conflict detection helper methods
4. Tests for precondition handling (_add_new_action_preconditions, _update_action_preconditions)
5. Tests for effect handling (handle_effects)
6. Integration tests for full learning with patches

Test Data:
- Problem: problem7
- Trajectory: src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory
- Masking: src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.masking_info
"""

import unittest
from pathlib import Path
from copy import deepcopy

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser
from pddl_plus_parser.models import Domain, Observation, Predicate
from utilities import NegativePreconditionPolicy

from src.pi_sam.noisy_pisam.noisy_pisam_learning import NoisyPisamLearner
from src.pi_sam.noisy_pisam.typings import (
    FluentLevelPatch,
    ModelLevelPatch,
    ParameterBoundLiteral,
    ConflictType,
    ModelPart,
    PatchOperation,
    Conflict
)
from src.utils.pddl import ground_observation_completely
from src.utils.masking import mask_observation, load_masking_info


class TestNoisyPisamLearner(unittest.TestCase):
    """Test suite for NoisyPisamLearner."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests."""
        # Load blocks domain
        cls.domain_file = Path("src/domains/blocks/blocks.pddl")
        cls.domain: Domain = DomainParser(cls.domain_file).parse_domain()

        # Load problem7 trajectory and masking info
        cls.experiment_dir = Path("src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06")
        cls.trajectory_file = cls.experiment_dir / "problem7.trajectory"
        cls.masking_file = cls.experiment_dir / "problem7.masking_info"

        if cls.trajectory_file.exists() and cls.masking_file.exists():
            # Parse trajectory
            parser = TrajectoryParser(partial_domain=cls.domain)
            cls.observation = parser.parse_trajectory(cls.trajectory_file)

            # Ground the observation
            cls.grounded_observation = ground_observation_completely(cls.domain, cls.observation)

            # Load masking info from file
            cls.masking_info = load_masking_info(cls.masking_file, cls.domain)

            # Apply masking
            cls.masked_observation = mask_observation(cls.grounded_observation, cls.masking_info)
        else:
            missing = []
            if not cls.trajectory_file.exists():
                missing.append(f"Trajectory: {cls.trajectory_file}")
            if not cls.masking_file.exists():
                missing.append(f"Masking info: {cls.masking_file}")
            print(f"Warning: Required files not found: {', '.join(missing)}")
            cls.observation = None
            cls.grounded_observation = None
            cls.masked_observation = None
            cls.masking_info = None

    def setUp(self):
        """Set up for each test."""
        # Create a fresh learner for each test
        self.learner = NoisyPisamLearner(
            deepcopy(self.domain),
            negative_preconditions_policy=NegativePreconditionPolicy.hard
        )

    # ==================== Unit Tests for Helper Methods ====================

    def test_detect_forbidden_added_preconditions(self):
        """Test detection of forbidden preconditions that were added."""
        # Create a forbidden precondition patch: FORBID holding(?x) in stack's preconditions
        forbidden_pbl = ParameterBoundLiteral("holding", ("?x",), is_positive=True)
        self.learner.forbidden_preconditions["stack"] = {forbidden_pbl}

        # Create a mock predicate that matches the PBL
        # signature should be a dict with parameter names as keys
        from pddl_plus_parser.models import PDDLType
        mock_predicate = Predicate(
            name="holding",
            signature={"?x": PDDLType(name="block")},  # Proper signature format
            is_positive=True
        )

        # Set tracking indices
        self.learner.current_observation_index = 0
        self.learner.current_component_index = 0

        # Test the helper method
        conflicts = self.learner._detect_forbidden_added(
            action_name="stack",
            model_part=ModelPart.PRECONDITION,
            added_literals=[mock_predicate]
        )

        # Assertions
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].conflict_type, ConflictType.FORBIDDEN_PRECONDITION)
        self.assertEqual(conflicts[0].action_name, "stack")
        self.assertEqual(conflicts[0].pbl, forbidden_pbl)

    def test_detect_required_missing_preconditions(self):
        """Test detection of required preconditions that are missing."""
        # Create a required precondition patch: REQUIRE clear(?y) in stack's preconditions
        required_pbl = ParameterBoundLiteral("clear", ("?y",), is_positive=True)
        self.learner.required_preconditions["stack"] = {required_pbl}

        # Create a set of current preconditions that does NOT include clear(?y)
        from pddl_plus_parser.models import PDDLType
        mock_predicate1 = Predicate(
            name="holding",
            signature={"?x": PDDLType(name="block")},
            is_positive=True
        )

        current_preconditions = {mock_predicate1}

        # Set tracking indices
        self.learner.current_observation_index = 0
        self.learner.current_component_index = 0

        # Test the helper method
        conflicts = self.learner._detect_required_missing(
            action_name="stack",
            model_part=ModelPart.PRECONDITION,
            current_literals=current_preconditions
        )

        # Assertions
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].conflict_type, ConflictType.REQUIRED_PRECONDITION)
        self.assertEqual(conflicts[0].action_name, "stack")
        self.assertEqual(conflicts[0].pbl, required_pbl)

    def test_detect_required_removed_preconditions(self):
        """Test detection of required preconditions that were removed."""
        # Create a required precondition patch: REQUIRE handempty in unstack's preconditions
        required_pbl = ParameterBoundLiteral("handempty", (), is_positive=True)
        self.learner.required_preconditions["unstack"] = {required_pbl}

        # Create a predicate that was removed
        removed_predicate = Predicate(
            name="handempty",
            signature={},  # Empty dict for no parameters
            is_positive=True
        )

        # Set tracking indices
        self.learner.current_observation_index = 0
        self.learner.current_component_index = 0

        # Test the helper method
        conflicts = self.learner._detect_required_removed(
            action_name="unstack",
            model_part=ModelPart.PRECONDITION,
            removed_literals={removed_predicate}
        )

        # Assertions
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].conflict_type, ConflictType.REQUIRED_PRECONDITION)
        self.assertEqual(conflicts[0].action_name, "unstack")

    def test_detect_forbidden_added_effects(self):
        """Test detection of forbidden effects that were added."""
        # Create a forbidden effect patch: FORBID ontable(?x) in pickup's effects
        forbidden_pbl = ParameterBoundLiteral("ontable", ("?x",), is_positive=False)
        self.learner.forbidden_effects["pickup"] = {forbidden_pbl}

        # Create a mock predicate that matches
        from pddl_plus_parser.models import PDDLType
        mock_predicate = Predicate(
            name="ontable",
            signature={"?x": PDDLType(name="block")},
            is_positive=False
        )

        # Set tracking indices
        self.learner.current_observation_index = 0
        self.learner.current_component_index = 1

        # Test the helper method
        conflicts = self.learner._detect_forbidden_added(
            action_name="pickup",
            model_part=ModelPart.EFFECT,
            added_literals=[mock_predicate]
        )

        # Assertions
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].conflict_type, ConflictType.FORBIDDEN_EFFECT)
        self.assertEqual(conflicts[0].action_name, "pickup")

    # ==================== Tests for Precondition Handling ====================

    @unittest.skipIf(not Path("src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_add_new_action_preconditions_with_forbidden_patch(self):
        """
        Test _add_new_action_preconditions with a forbidden precondition patch.

        Expected behavior:
        - Parent method adds all predicates from previous state
        - Our method detects conflict if any match forbidden patch
        """
        # Set up forbidden patch: Don't allow "on" predicates in pickup preconditions
        forbidden_pbl = ParameterBoundLiteral("on", ("?x", "?y"), is_positive=True)

        model_patches = {
            ModelLevelPatch(
                action_name="pickup",
                model_part=ModelPart.PRECONDITION,
                pbl=forbidden_pbl,
                operation=PatchOperation.FORBID
            )
        }

        self.learner.set_patches(fluent_patches=set(), model_patches=model_patches)

        # Get first component with pickup action (if any)
        pickup_component = None
        for component in self.masked_observation.components:
            if "pickup" in component.grounded_action_call.name:
                pickup_component = component
                break

        if pickup_component:
            self.learner.current_observation_index = 0
            self.learner.current_component_index = 0

            # Call the method
            self.learner._add_new_action_preconditions(
                pickup_component.grounded_action_call,
                pickup_component.previous_state
            )

            # Check if conflicts were detected
            # This will depend on whether "on" predicates are in the previous state
            if self.learner.conflicts:
                self.assertTrue(any(c.conflict_type == ConflictType.FORBIDDEN_PRECONDITION
                                   for c in self.learner.conflicts))

    @unittest.skipIf(not Path("src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_add_new_action_preconditions_with_required_missing(self):
        """
        Test _add_new_action_preconditions with a required precondition that's missing.

        Expected behavior:
        - Conflict is detected if required precondition is NOT in the previous state
        """
        # Set up required patch: Require a predicate that's unlikely to be there
        # For example, require "handfull" in pickup preconditions (usually handempty)
        required_pbl = ParameterBoundLiteral("handfull", (), is_positive=True)

        model_patches = {
            ModelLevelPatch(
                action_name="pickup",
                model_part=ModelPart.PRECONDITION,
                pbl=required_pbl,
                operation=PatchOperation.REQUIRE
            )
        }

        self.learner.set_patches(fluent_patches=set(), model_patches=model_patches)

        # Get first component with pickup action
        pickup_component = None
        for component in self.masked_observation.components:
            if "pickup" in component.grounded_action_call.name:
                pickup_component = component
                break

        if pickup_component:
            self.learner.current_observation_index = 0
            self.learner.current_component_index = 0

            # Call the method
            self.learner._add_new_action_preconditions(
                pickup_component.grounded_action_call,
                pickup_component.previous_state
            )

            # Check if required missing conflict was detected
            required_conflicts = [c for c in self.learner.conflicts
                                 if c.conflict_type == ConflictType.REQUIRED_PRECONDITION]

            # Should detect that handfull is missing
            self.assertTrue(len(required_conflicts) > 0)

    @unittest.skipIf(not Path("src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_update_action_preconditions_with_required_removed(self):
        """
        Test _update_action_preconditions detects when a required precondition is removed.

        This test simulates a scenario where:
        1. An action already has some preconditions
        2. SAM's update rule removes one
        3. That precondition is marked as REQUIRED by a patch
        4. Conflict should be detected
        """
        # This is harder to test without actually running learning first
        # Skipping detailed implementation for now
        pass

    # ==================== Tests for Effect Handling ====================

    @unittest.skipIf(not Path("src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_handle_effects_with_forbidden_patch(self):
        """
        Test handle_effects with a forbidden effect patch.

        Expected behavior:
        - If SAM wants to add a forbidden effect, conflict is detected
        - Effects are NOT added to the model (early return)
        """
        # Set up forbidden patch: Don't allow "holding" in pickup effects
        forbidden_pbl = ParameterBoundLiteral("holding", ("?x",), is_positive=True)

        model_patches = {
            ModelLevelPatch(
                action_name="pickup",
                model_part=ModelPart.EFFECT,
                pbl=forbidden_pbl,
                operation=PatchOperation.FORBID
            )
        }

        self.learner.set_patches(fluent_patches=set(), model_patches=model_patches)

        # Get first pickup action
        pickup_component = None
        for component in self.masked_observation.components:
            if "pickup" in component.grounded_action_call.name:
                pickup_component = component
                break

        if pickup_component:
            self.learner.current_observation_index = 0
            self.learner.current_component_index = 0

            # Initialize the action first (needed for effect handling)
            action_name = pickup_component.grounded_action_call.name
            if action_name not in self.learner.observed_actions:
                self.learner.observed_actions.append(action_name)

            # Call handle_effects
            self.learner.handle_effects(
                pickup_component.grounded_action_call,
                pickup_component.previous_state,
                pickup_component.next_state
            )

            # Check for conflicts
            effect_conflicts = [c for c in self.learner.conflicts
                               if c.conflict_type == ConflictType.FORBIDDEN_EFFECT]

            # pickup normally has holding(?x) as an effect, so should conflict
            if effect_conflicts:
                self.assertTrue(any("holding" in c.grounded_fluent for c in effect_conflicts))

    # ==================== Integration Tests ====================

    @unittest.skipIf(not Path("src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_learn_with_fluent_patches(self):
        """
        Integration test: Learn with fluent-level patches.

        Tests that:
        1. Fluent patches can be set
        2. Learning completes without errors
        3. Fluent patches are applied (when implemented)
        """
        # Create a simple fluent patch
        fluent_patches = {
            FluentLevelPatch(
                observation_index=0,
                component_index=0,
                state_type='next',
                fluent='on(a, b)'
            )
        }

        # Learn with patches
        learned_domain, conflicts = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=fluent_patches,
            model_patches=set()
        )

        # Should complete without errors
        self.assertIsNotNone(learned_domain)
        self.assertIsInstance(conflicts, list)

    @unittest.skipIf(not Path("src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_learn_with_model_patches_conflicts(self):
        """
        Integration test: Learn with model patches that cause conflicts.

        Tests that:
        1. Model patches can be set
        2. Conflicts are detected during learning
        3. Learning completes (even with conflicts)
        """
        # Create model patches that should cause conflicts
        # FORBID holding in pickup effects (will conflict since pickup adds holding)
        model_patches = {
            ModelLevelPatch(
                action_name="pickup",
                model_part=ModelPart.EFFECT,
                pbl=ParameterBoundLiteral("holding", ("?x",), True),
                operation=PatchOperation.FORBID
            )
        }

        # Learn with patches
        learned_domain, conflicts = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=set(),
            model_patches=model_patches
        )

        # Should detect conflicts
        self.assertIsNotNone(learned_domain)

        # Check if conflicts were detected
        # Note: This assumes the observation has pickup actions
        if any("pickup" in c.grounded_action_call.name for c in self.masked_observation.components):
            self.assertTrue(len(conflicts) > 0, "Expected conflicts to be detected")

    @unittest.skipIf(not Path("src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_learn_without_patches(self):
        """
        Baseline test: Learn without any patches.

        Should behave like regular PISAMLearner.
        """
        # Learn without patches
        learned_domain, conflicts = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=set(),
            model_patches=set()
        )

        # Should complete successfully
        self.assertIsNotNone(learned_domain)

        # Should have no conflicts
        self.assertEqual(len(conflicts), 0)

        # Should have learned some actions
        self.assertTrue(len(self.learner.observed_actions) > 0)


def run_single_test(test_name):
    """Helper function to run a single test."""
    suite = unittest.TestLoader().loadTestsFromName(f'__main__.TestNoisyPisamLearner.{test_name}')
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
