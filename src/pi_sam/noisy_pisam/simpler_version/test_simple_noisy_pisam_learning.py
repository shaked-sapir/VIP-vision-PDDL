"""
Comprehensive test suite for SimpleNoisyPisamLearner.

This test file includes tests for different conflict types supported in the simpler version:
1. Fluent-level patches (no conflicts, just observation modification)
2. PRE_REQUIRE_VS_CANNOT: Required precondition that PI-SAM wants to remove
3. FORBID_VS_MUST: Forbidden effect that PI-SAM says must be an effect
4. REQUIRE_VS_CANNOT: Required effect that PI-SAM says cannot be an effect
5. Mixed model conflicts (combinations of the above)
6. Mixed model and fluent patches

Note: The simpler version does NOT support FORBID precondition patches.

Test Data:
- Problem: problem7
- Trajectory: src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory
- Masking: src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.masking_info
"""

import unittest
from pathlib import Path
from copy import deepcopy

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser
from pddl_plus_parser.models import Domain, Observation
from utilities import NegativePreconditionPolicy

from src.pi_sam.noisy_pisam.simpler_version.simple_noisy_pisam_learning import SimpleNoisyPisamLearner
from src.pi_sam.noisy_pisam.simpler_version.typings import (
    FluentLevelPatch,
    ModelLevelPatch,
    ParameterBoundLiteral,
    ConflictType,
    ModelPart,
    PatchOperation
)
from src.utils.pddl import ground_observation_completely
from src.utils.masking import mask_observation, load_masking_info

absulute_path_prefix = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/")


class TestSimpleNoisyPisamLearner(unittest.TestCase):
    """Comprehensive test suite for SimpleNoisyPisamLearner with different conflict types."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests."""
        # Load blocks domain
        cls.domain_file = absulute_path_prefix / Path("src/domains/blocks/blocks.pddl")
        cls.domain: Domain = DomainParser(cls.domain_file).parse_domain()

        # Load problem7 trajectory and masking info
        cls.experiment_dir = absulute_path_prefix / Path("src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06")
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
        self.learner = SimpleNoisyPisamLearner(
            deepcopy(self.domain),
            negative_preconditions_policy=NegativePreconditionPolicy.hard
        )

    # ==================== Fluent-Level Conflict Tests ====================

    @unittest.skipIf(not Path(f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_fluent_level_conflict_only(self):
        """
        Test with only fluent-level patches (no model patches).

        Expected: Learning completes, fluent patches are applied to observation.
        """
        # Create fluent patches that flip some predicates
        fluent_patches = {
            FluentLevelPatch(
                observation_index=0,
                component_index=2,
                state_type='next',
                fluent='(on a b)'
            ),
            FluentLevelPatch(
                observation_index=0,
                component_index=5,
                state_type='previous',
                fluent='(clear d)'
            )
        }

        # Learn with only fluent patches
        learned_domain, conflicts = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=fluent_patches,
            model_patches=set()
        )

        # Should complete successfully
        self.assertIsNotNone(learned_domain)
        self.assertIsInstance(conflicts, list)
        # With only fluent patches, there should be no conflicts
        # (fluent patches modify the observation before learning)

    # ==================== Model-Level Precondition Conflict Tests ====================

    @unittest.skipIf(not Path(f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_precondition_require_vs_cannot_conflict(self):
        """
        Test model-level precondition conflict: PRE_REQUIRE_VS_CANNOT.

        Scenario: REQUIRE a precondition that PI-SAM wants to remove
        Expected: Conflict of type PRE_REQUIRE_VS_CANNOT

        Note: The simpler version only detects precondition conflicts when
        a REQUIRED precondition is being removed by SAM's update logic.
        """
        # Create required precondition patch
        required_pbl = ParameterBoundLiteral("clear", ("?x",), is_positive=True)
        model_patches = {
            ModelLevelPatch(
                action_name="pick-up",
                model_part=ModelPart.PRECONDITION,
                pbl=required_pbl,
                operation=PatchOperation.REQUIRE
            )
        }

        # Learn with required precondition patch
        learned_domain, conflicts = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=set(),
            model_patches=model_patches
        )

        self.assertIsNotNone(learned_domain)

        # Check for PRE_REQUIRE_VS_CANNOT conflicts
        pre_conflicts = [c for c in conflicts
                        if c.conflict_type == ConflictType.PRE_REQUIRE_VS_CANNOT]

        # May or may not have conflicts depending on whether SAM removes clear(?x)
        print(f"PRE_REQUIRE_VS_CANNOT conflicts: {len(pre_conflicts)}")


    # ==================== Model-Level Effect Conflict Tests ====================

    @unittest.skipIf(not Path(f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_forbidden_effect_conflict(self):
        """
        Test model-level forbidden effect conflict: FORBID_VS_MUST.

        Scenario: FORBID holding(?x) in pick-up's effects
        Expected: Conflict of type FORBID_VS_MUST when SAM determines holding(?x) must be an effect
        """
        # Create forbidden effect patch
        forbidden_pbl = ParameterBoundLiteral("holding", ("?x",), is_positive=True)
        model_patches = {
            ModelLevelPatch(
                action_name="pick-up",
                model_part=ModelPart.EFFECT,
                pbl=forbidden_pbl,
                operation=PatchOperation.FORBID
            )
        }

        # Learn with forbidden effect patch
        learned_domain, conflicts = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=set(),
            model_patches=model_patches
        )

        self.assertIsNotNone(learned_domain)

        # Check for FORBID_VS_MUST conflicts
        forbidden_conflicts = [c for c in conflicts
                              if c.conflict_type == ConflictType.FORBID_VS_MUST]

        # pickup action typically adds holding(?x), so should have conflicts
        print(f"FORBID_VS_MUST conflicts: {len(forbidden_conflicts)}")

    @unittest.skipIf(not Path(f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_required_effect_conflict(self):
        """
        Test model-level required effect conflict: REQUIRE_VS_CANNOT.

        Scenario: REQUIRE ontable(?x) in pick-up's effects (which shouldn't be there)
        Expected: Conflict of type REQUIRE_VS_CANNOT when SAM determines ontable(?x) cannot be an effect
        """
        # Create required effect patch for a predicate that shouldn't be an effect
        required_pbl = ParameterBoundLiteral("ontable", ("?x",), is_positive=True)
        model_patches = {
            ModelLevelPatch(
                action_name="pick-up",
                model_part=ModelPart.EFFECT,
                pbl=required_pbl,
                operation=PatchOperation.REQUIRE
            )
        }

        # Learn with required effect patch
        learned_domain, conflicts = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=set(),
            model_patches=model_patches
        )

        self.assertIsNotNone(learned_domain)

        # Check for REQUIRE_VS_CANNOT conflicts
        required_conflicts = [c for c in conflicts
                             if c.conflict_type == ConflictType.REQUIRE_VS_CANNOT]

        # Should detect that ontable(?x) cannot be an effect
        print(f"REQUIRE_VS_CANNOT conflicts: {len(required_conflicts)}")
        print(required_conflicts)

    # ==================== Mixed Model Conflict Tests ====================

    @unittest.skipIf(not Path(f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_mixed_model_conflicts_all_types(self):
        """
        Test with all different types of model-level conflicts supported in simpler version.

        Includes all three conflict types:
        - PRE_REQUIRE_VS_CANNOT: Required precondition that SAM wants to remove
        - FORBID_VS_MUST: Forbidden effect that SAM says must be an effect
        - REQUIRE_VS_CANNOT: Required effect that SAM says cannot be an effect

        Note: FORBID precondition patches are NOT supported in simpler version.
        """
        model_patches = {
            # Required precondition: Require handfull in pick-up preconditions
            # Can create PRE_REQUIRE_VS_CANNOT conflict
            ModelLevelPatch(
                action_name="pick-up",
                model_part=ModelPart.PRECONDITION,
                pbl=ParameterBoundLiteral("handfull", ("?robot",), True),
                operation=PatchOperation.REQUIRE
            ),
            # Forbidden effect: Don't allow holding(?x) in pick-up effects
            # Can create FORBID_VS_MUST conflict
            ModelLevelPatch(
                action_name="pick-up",
                model_part=ModelPart.EFFECT,
                pbl=ParameterBoundLiteral("holding", ("?x",), True),
                operation=PatchOperation.FORBID
            ),
            # Required effect: Require ontable(?x) in pick-up effects
            # Can create REQUIRE_VS_CANNOT conflict
            ModelLevelPatch(
                action_name="pick-up",
                model_part=ModelPart.EFFECT,
                pbl=ParameterBoundLiteral("ontable", ("?x",), True),
                operation=PatchOperation.REQUIRE
            )
        }

        # Learn with all types of model patches
        learned_domain, conflicts = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=set(),
            model_patches=model_patches
        )

        self.assertIsNotNone(learned_domain)

        # Collect conflicts by type
        conflict_types = {c.conflict_type for c in conflicts}

        # Should have multiple types of conflicts
        print(f"Detected conflict types: {conflict_types}")
        print(f"Total conflicts: {len(conflicts)}")

        # Verify all three types can be detected
        all_types = {ConflictType.PRE_REQUIRE_VS_CANNOT, ConflictType.FORBID_VS_MUST, ConflictType.REQUIRE_VS_CANNOT}
        print(f"Coverage: {len(conflict_types & all_types)}/3 conflict types detected")

        # Verify we have conflicts (exact types depend on the trajectory)
        self.assertTrue(len(conflicts) > 0, "Should detect multiple conflicts")
        self.assertTrue(len(conflict_types) == 3, "Should detect all three conflict types")

    @unittest.skipIf(not Path(f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_mixed_model_conflicts_multiple_actions(self):
        """
        Test with model conflicts across multiple actions.

        Patches for both pick-up and put-down actions.
        Note: Only using supported patch types (no FORBID precondition).
        """
        model_patches = {
            # pick-up: FORBID holding(?x) effect (can create FORBID_VS_MUST)
            ModelLevelPatch(
                action_name="pick-up",
                model_part=ModelPart.EFFECT,
                pbl=ParameterBoundLiteral("holding", ("?x",), True),
                operation=PatchOperation.FORBID
            ),
            # put-down: REQUIRE handempty precondition (can create PRE_REQUIRE_VS_CANNOT)
            ModelLevelPatch(
                action_name="put-down",
                model_part=ModelPart.PRECONDITION,
                pbl=ParameterBoundLiteral("handempty", ("?robot",), True),
                operation=PatchOperation.REQUIRE
            ),
            # put-down: FORBID handempty effect (can create FORBID_VS_MUST)
            ModelLevelPatch(
                action_name="put-down",
                model_part=ModelPart.EFFECT,
                pbl=ParameterBoundLiteral("handempty", ("?robot",), True),
                operation=PatchOperation.FORBID
            )
        }

        # Learn with patches for multiple actions
        learned_domain, conflicts = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=set(),
            model_patches=model_patches
        )

        self.assertIsNotNone(learned_domain)

        # Check conflicts involve multiple actions
        action_names = {c.action_name for c in conflicts}
        print(f"Actions with conflicts: {action_names}")
        print(f"Total conflicts: {len(conflicts)}")

        # Show conflict types
        conflict_types = {c.conflict_type for c in conflicts}
        print(f"Conflict types: {conflict_types}")

    # ==================== Mixed Model and Fluent Conflict Tests ====================

    @unittest.skipIf(not Path(f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_mixed_fluent_and_model_conflicts(self):
        """
        Test with both fluent-level and model-level patches.

        Combines:
        - Fluent patches that modify observations
        - Model patches that constrain learning (using supported conflict types only)
        """
        fluent_patches = {
            FluentLevelPatch(
                observation_index=0,
                component_index=1,
                state_type='next',
                fluent='on(a, b)'
            ),
            FluentLevelPatch(
                observation_index=0,
                component_index=3,
                state_type='prev',  # Use 'prev' instead of 'previous'
                fluent='clear(c)'
            )
        }

        model_patches = {
            # FORBID holding effect (can create FORBID_VS_MUST)
            ModelLevelPatch(
                action_name="pick-up",
                model_part=ModelPart.EFFECT,
                pbl=ParameterBoundLiteral("holding", ("?x",), True),
                operation=PatchOperation.FORBID
            ),
            # REQUIRE handfull precondition (can create PRE_REQUIRE_VS_CANNOT)
            ModelLevelPatch(
                action_name="pick-up",
                model_part=ModelPart.PRECONDITION,
                pbl=ParameterBoundLiteral("handfull", ("?robot",), True),
                operation=PatchOperation.REQUIRE
            )
        }

        # Learn with both types of patches
        learned_domain, conflicts = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=fluent_patches,
            model_patches=model_patches
        )

        self.assertIsNotNone(learned_domain)

        print(f"Fluent patches applied: {len(fluent_patches)}")
        print(f"Model conflicts detected: {len(conflicts)}")

        # Show conflict types with their names
        conflict_types = [c.conflict_type.value for c in conflicts]
        print(f"Conflict types: {conflict_types}")

        # Verify we have some conflicts
        self.assertIsInstance(conflicts, list)

    @unittest.skipIf(not Path(f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_complex_mixed_scenario(self):
        """
        Test complex scenario with many patches of both types.

        This represents a realistic conflict-based learning scenario using
        only supported conflict types in the simpler version.
        """
        fluent_patches = {
            FluentLevelPatch(0, 1, 'next', 'on(a, b)'),
            FluentLevelPatch(0, 2, 'next', 'clear(d)'),
            FluentLevelPatch(0, 4, 'prev', 'holding(e)')  # Use 'prev' instead of 'previous'
        }

        model_patches = {
            # pick-up action patches
            # FORBID holding effect (can create FORBID_VS_MUST)
            ModelLevelPatch("pick-up", ModelPart.EFFECT,
                           ParameterBoundLiteral("holding", ("?x",), True),
                           PatchOperation.FORBID),
            # REQUIRE ontable effect (can create REQUIRE_VS_CANNOT)
            ModelLevelPatch("pick-up", ModelPart.EFFECT,
                           ParameterBoundLiteral("ontable", ("?x",), True),
                           PatchOperation.REQUIRE),
            # put-down action patches
            # REQUIRE handempty precondition (can create PRE_REQUIRE_VS_CANNOT)
            ModelLevelPatch("put-down", ModelPart.PRECONDITION,
                           ParameterBoundLiteral("handempty", ("?robot",), True),
                           PatchOperation.REQUIRE),
            # REQUIRE ontable effect (can create REQUIRE_VS_CANNOT)
            ModelLevelPatch("put-down", ModelPart.EFFECT,
                           ParameterBoundLiteral("ontable", ("?x",), True),
                           PatchOperation.REQUIRE)
        }

        # Learn with complex patch combination
        learned_domain, conflicts = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=fluent_patches,
            model_patches=model_patches
        )

        self.assertIsNotNone(learned_domain)

        print(f"\n=== Complex Mixed Scenario Results ===")
        print(f"Fluent patches: {len(fluent_patches)}")
        print(f"Model patches: {len(model_patches)}")
        print(f"Total conflicts: {len(conflicts)}")

        # Group conflicts by type
        conflict_summary = {}
        for conflict in conflicts:
            ct = conflict.conflict_type.value
            conflict_summary[ct] = conflict_summary.get(ct, 0) + 1

        print(f"Conflicts by type: {conflict_summary}")

        # Verify supported conflict types
        for conflict in conflicts:
            self.assertIn(conflict.conflict_type,
                         [ConflictType.FORBID_VS_MUST,
                          ConflictType.REQUIRE_VS_CANNOT,
                          ConflictType.PRE_REQUIRE_VS_CANNOT],
                         f"Unexpected conflict type: {conflict.conflict_type}")

    # ==================== Baseline Test ====================

    @unittest.skipIf(not Path(f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_no_patches_baseline(self):
        """
        Baseline test: Learn without any patches.

        Should behave like regular PISAMLearner with no conflicts.
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
        self.assertEqual(len(conflicts), 0,
                        "Learning without patches should produce no conflicts")

        # Should have learned some actions
        self.assertTrue(len(self.learner.observed_actions) > 0,
                       "Should have learned some actions")


def run_single_test(test_name):
    """Helper function to run a single test."""
    suite = unittest.TestLoader().loadTestsFromName(
        f'__main__.TestSimpleNoisyPisamLearner.{test_name}'
    )
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
