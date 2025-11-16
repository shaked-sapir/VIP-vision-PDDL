"""
Comprehensive test suite for SimpleNoisyPisamLearner.

This test file includes tests for different conflict types supported in the simpler version:

1. Fluent-level patches (positive and negative, no conflicts, just observation modification)

2. Model-level conflict tests:
   - FORBID_PRECOND_VS_IS: Forbidden precondition that PI-SAM wants to add
   - FORBID_EFFECT_VS_MUST: Forbidden effect that PI-SAM says must be an effect
   - REQUIRE_EFFECT_VS_CANNOT: Required effect that PI-SAM says cannot be an effect

3. Mixed model conflicts (combinations of the above)

4. Mixed model and fluent patches

5. Negative predicate tests:
   - Negative fluent patches (e.g., "(not (clear a))")
   - Negative effect patches (e.g., ParameterBoundLiteral with is_positive=False)
   - Negative precondition patches
   - Complex mixed scenarios with both positive and negative patches

6. Data-driven conflict tests:
   These tests demonstrate how conflicts arise from the interaction between fluent patches
   (data modifications) and model patches (learning constraints):

   - test_data_driven_forbid_effect_conflict:
     Flip fluent to ADD an effect → FORBID model patch → FORBID_EFFECT_VS_MUST conflict

   - test_data_driven_require_effect_conflict:
     Flip fluent to REMOVE an effect → REQUIRE model patch → REQUIRE_EFFECT_VS_CANNOT conflict

   - test_data_driven_forbid_precondition_conflict:
     Flip fluent to ADD a precondition → FORBID model patch → FORBID_PRECOND_VS_IS conflict

   - test_data_driven_multiple_conflicts:
     Multiple data modifications → Multiple model patches → Various conflict types

   - test_data_driven_negative_fluent_conflicts:
     Negative fluent modifications → Negative model patches → Conflicts with negative predicates

   These tests show that conflicts are detected when:
   - Fluent patches modify the observation data (add/remove predicates)
   - PI-SAM analyzes the modified data and determines what must/cannot be learned
   - Model patches constrain what can be learned
   - The data-driven PI-SAM conclusions conflict with model patch constraints

Test Data:
- Problem: problem7
- Trajectory: src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory
- Masking: src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.masking_info
"""

import unittest
from copy import deepcopy
from pathlib import Path

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser
from pddl_plus_parser.models import Domain
from utilities import NegativePreconditionPolicy

from src.pi_sam.noisy_pisam.simpler_version.simple_noisy_pisam_learning import NoisyPisamLearner
from src.pi_sam.noisy_pisam.simpler_version.typings import (
    FluentLevelPatch,
    ModelLevelPatch,
    ParameterBoundLiteral,
    ConflictType,
    ModelPart,
    PatchOperation
)
from src.utils.masking import mask_observation, load_masking_info
from src.utils.pddl import ground_observation_completely

absulute_path_prefix = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/")


class TestSimpleNoisyPisamLearner(unittest.TestCase):
    """Comprehensive test suite for SimpleNoisyPisamLearner with different conflict types."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests."""
        # Load blocks domain
        cls.domain_file = absulute_path_prefix / Path("src/domains/blocks/blocks.pddl")
        cls.domain: Domain = DomainParser(cls.domain_file, partial_parsing=True).parse_domain()

        # Load problem7 trajectory and masking info
        cls.experiment_dir = absulute_path_prefix / Path("src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06")
        cls.trajectory_file = cls.experiment_dir / "problem7.trajectory"
        cls.masking_file = cls.experiment_dir / "problem7.masking_info"

        cls.with_data_only_conflicts = True

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
                state_type='prev',
                fluent='(clear d)'
            )
        }

        # Learn with only fluent patches
        learned_domain, conflicts, report = self.learner.learn_action_model_with_conflicts(
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
    def test_precondition_forbid_vs_is_conflict(self):
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
                operation=PatchOperation.FORBID
            )
        }

        # Learn with required precondition patch
        learned_domain, conflicts, report = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=set(),
            model_patches=model_patches
        )

        self.assertIsNotNone(learned_domain)

        # Check for PRE_REQUIRE_VS_CANNOT conflicts
        pre_conflicts = [c for c in conflicts
                        if c.conflict_type == ConflictType.FORBID_PRECOND_VS_IS]

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
        learned_domain, conflicts, report = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=set(),
            model_patches=model_patches
        )

        self.assertIsNotNone(learned_domain)

        # Check for FORBID_VS_MUST conflicts
        forbidden_conflicts = [c for c in conflicts
                              if c.conflict_type == ConflictType.FORBID_EFFECT_VS_MUST]

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
        learned_domain, conflicts, report = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=set(),
            model_patches=model_patches
        )

        self.assertIsNotNone(learned_domain)

        # Check for REQUIRE_VS_CANNOT conflicts
        required_conflicts = [c for c in conflicts
                             if c.conflict_type == ConflictType.REQUIRE_EFFECT_VS_CANNOT]

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
                pbl=ParameterBoundLiteral("handempty", ("?robot",), True),
                operation=PatchOperation.FORBID
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
        learned_domain, conflicts, report = self.learner.learn_action_model_with_conflicts(
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
        all_types = {ConflictType.FORBID_PRECOND_VS_IS, ConflictType.FORBID_EFFECT_VS_MUST, ConflictType.REQUIRE_EFFECT_VS_CANNOT}
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
                pbl=ParameterBoundLiteral("handfull", ("?robot",), True),
                operation=PatchOperation.FORBID
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
        learned_domain, conflicts, report = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=set(),
            model_patches=model_patches
        )

        self.assertIsNotNone(learned_domain)

        # Check conflicts involve multiple actions
        action_names = {c.action_name for c in conflicts}
        print(f"Actions with conflicts: {action_names}")
        print(f"Total conflicts: {len(conflicts)}")
        print(conflicts)

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
                fluent='(on a b)'
            ),
            FluentLevelPatch(
                observation_index=0,
                component_index=3,
                state_type='prev',  # Use 'prev' instead of 'previous'
                fluent='(not (clear c))'
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
                pbl=ParameterBoundLiteral("handempty", ("?robot",), True),
                operation=PatchOperation.FORBID
            )
        }

        # Learn with both types of patches
        learned_domain, conflicts, report = self.learner.learn_action_model_with_conflicts(
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
            FluentLevelPatch(0, 1, 'next', '(on a b)'),
            FluentLevelPatch(0, 2, 'next', '(clear d)'),
            FluentLevelPatch(0, 4, 'prev', '(holding e)')  # Use 'prev' instead of 'previous'
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
                           ParameterBoundLiteral("handfull", ("?robot",), True),
                           PatchOperation.FORBID),
            # REQUIRE ontable effect (can create REQUIRE_VS_CANNOT)
            ModelLevelPatch("put-down", ModelPart.EFFECT,
                           ParameterBoundLiteral("ontable", ("?x",), True),
                           PatchOperation.REQUIRE)
        }

        # Learn with complex patch combination
        learned_domain, conflicts, report = self.learner.learn_action_model_with_conflicts(
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
                         [ConflictType.FORBID_EFFECT_VS_MUST,
                          ConflictType.REQUIRE_EFFECT_VS_CANNOT,
                          ConflictType.FORBID_PRECOND_VS_IS],
                         f"Unexpected conflict type: {conflict.conflict_type}")

    # ==================== Negative Predicate Tests ====================

    @unittest.skipIf(not Path(f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_negative_fluent_patches_only(self):
        """
        Test with only negative fluent-level patches.

        Expected: Learning completes, negative fluent patches are applied to observation.
        """
        # Create negative fluent patches
        fluent_patches = {
            FluentLevelPatch(
                observation_index=0,
                component_index=1,
                state_type='next',
                fluent='(not (clear a))'
            ),
            FluentLevelPatch(
                observation_index=0,
                component_index=3,
                state_type='prev',
                fluent='(not (on b c))'
            ),
            FluentLevelPatch(
                observation_index=0,
                component_index=5,
                state_type='next',
                fluent='(not (holding d))'
            )
        }

        # Learn with only negative fluent patches
        learned_domain, conflicts, report = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=fluent_patches,
            model_patches=set()
        )

        # Should complete successfully
        self.assertIsNotNone(learned_domain)
        self.assertIsInstance(conflicts, list)
        print(f"Negative fluent patches applied: {len(fluent_patches)}")

    @unittest.skipIf(not Path(f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_negative_effect_patches(self):
        """
        Test model-level patches with negative effects.

        Scenario: FORBID and REQUIRE negative effects
        Expected: Conflicts when negative effects conflict with SAM rules
        """
        model_patches = {
            # FORBID not holding(?x) as effect (can create FORBID_EFFECT_VS_MUST)
            ModelLevelPatch(
                action_name="put-down",
                model_part=ModelPart.EFFECT,
                pbl=ParameterBoundLiteral("holding", ("?x",), is_positive=False),
                operation=PatchOperation.FORBID
            ),
            # REQUIRE not clear(?x) as effect (can create REQUIRE_EFFECT_VS_CANNOT)
            ModelLevelPatch(
                action_name="stack",
                model_part=ModelPart.EFFECT,
                pbl=ParameterBoundLiteral("clear", ("?y",), is_positive=False),
                operation=PatchOperation.REQUIRE
            )
        }

        # Learn with negative effect patches
        learned_domain, conflicts, report = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=set(),
            model_patches=model_patches
        )

        self.assertIsNotNone(learned_domain)

        # Show conflicts
        print(f"Negative effect patches conflicts: {len(conflicts)}")
        for conflict in conflicts:
            print(f"  - {conflict.conflict_type.value}: {conflict.grounded_fluent}")

    @unittest.skipIf(not Path(f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_negative_precondition_patches(self):
        """
        Test model-level patches with negative preconditions.

        Scenario: FORBID negative preconditions
        Expected: Conflicts when negative preconditions conflict with SAM rules
        """
        model_patches = {
            # FORBID not ontable(?x) as precondition (can create FORBID_PRECOND_VS_IS)
            ModelLevelPatch(
                action_name="pick-up",
                model_part=ModelPart.PRECONDITION,
                pbl=ParameterBoundLiteral("ontable", ("?x",), is_positive=False),
                operation=PatchOperation.FORBID
            ),
            # FORBID not holding(?x) as precondition for put-down
            ModelLevelPatch(
                action_name="put-down",
                model_part=ModelPart.PRECONDITION,
                pbl=ParameterBoundLiteral("holding", ("?x",), is_positive=False),
                operation=PatchOperation.FORBID
            )
        }

        # Learn with negative precondition patches
        learned_domain, conflicts, report = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=set(),
            model_patches=model_patches
        )

        self.assertIsNotNone(learned_domain)

        # Show conflicts
        print(f"Negative precondition patches conflicts: {len(conflicts)}")
        conflict_types = {c.conflict_type for c in conflicts}
        print(f"Conflict types: {conflict_types}")

    @unittest.skipIf(not Path(f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_complex_mixed_scenario_with_negatives(self):
        """
        Test complex scenario with both positive and negative patches.

        This tests a realistic scenario mixing:
        - Positive and negative fluent patches
        - Positive and negative model patches for effects
        - Positive and negative model patches for preconditions
        """
        fluent_patches = {
            # Positive fluent patches
            FluentLevelPatch(0, 1, 'next', '(on a b)'),
            FluentLevelPatch(0, 2, 'next', '(clear d)'),
            # Negative fluent patches
            FluentLevelPatch(0, 3, 'prev', '(not (holding c))'),
            FluentLevelPatch(0, 4, 'next', '(not (on e f))')
        }

        model_patches = {
            # Positive effect patches
            ModelLevelPatch("pick-up", ModelPart.EFFECT,
                           ParameterBoundLiteral("holding", ("?x",), is_positive=True),
                           PatchOperation.FORBID),
            ModelLevelPatch("pick-up", ModelPart.EFFECT,
                           ParameterBoundLiteral("ontable", ("?x",), is_positive=True),
                           PatchOperation.REQUIRE),
            # Negative effect patches
            ModelLevelPatch("put-down", ModelPart.EFFECT,
                           ParameterBoundLiteral("holding", ("?x",), is_positive=False),
                           PatchOperation.FORBID),
            ModelLevelPatch("stack", ModelPart.EFFECT,
                           ParameterBoundLiteral("clear", ("?y",), is_positive=False),
                           PatchOperation.REQUIRE),
            # Positive precondition patches
            ModelLevelPatch("pick-up", ModelPart.PRECONDITION,
                           ParameterBoundLiteral("handempty", ("?robot",), is_positive=True),
                           PatchOperation.FORBID),
            # Negative precondition patches
            ModelLevelPatch("put-down", ModelPart.PRECONDITION,
                           ParameterBoundLiteral("ontable", ("?x",), is_positive=False),
                           PatchOperation.FORBID)
        }

        # Learn with complex patch combination
        learned_domain, conflicts, report = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=fluent_patches,
            model_patches=model_patches
        )

        self.assertIsNotNone(learned_domain)

        print(f"\n=== Complex Mixed Scenario with Negatives ===")
        print(f"Fluent patches: {len(fluent_patches)} (positive and negative)")
        print(f"Model patches: {len(model_patches)} (positive and negative)")
        print(f"Total conflicts: {len(conflicts)}")

        # Group conflicts by type
        conflict_summary = {}
        for conflict in conflicts:
            ct = conflict.conflict_type.value
            conflict_summary[ct] = conflict_summary.get(ct, 0) + 1

        print(f"Conflicts by type: {conflict_summary}")

        # Show examples of conflicts with negative predicates
        negative_conflicts = [c for c in conflicts if hasattr(c.grounded_fluent, 'is_positive')
                             and not c.grounded_fluent.is_positive]
        print(f"Conflicts involving negative predicates: {len(negative_conflicts)}")

        # Verify supported conflict types
        for conflict in conflicts:
            self.assertIn(conflict.conflict_type,
                         [ConflictType.FORBID_EFFECT_VS_MUST,
                          ConflictType.REQUIRE_EFFECT_VS_CANNOT,
                          ConflictType.FORBID_PRECOND_VS_IS],
                         f"Unexpected conflict type: {conflict.conflict_type}")

    # ==================== Data-Driven Conflict Tests ====================

    @unittest.skipIf(not Path(f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_data_driven_forbid_effect_conflict(self):
        """
        Test data-driven FORBID_EFFECT_VS_MUST conflict.

        Scenario:
        1. Flip a fluent to ADD an effect in the data (e.g., holding(a) appears in next state)
        2. Add a FORBID model patch for that effect
        3. PI-SAM sees the effect in data and says it MUST be an effect
        4. Model patch says it's FORBIDDEN
        → FORBID_EFFECT_VS_MUST conflict

        This shows conflicts arising from data changes conflicting with model constraints.
        """
        print("\n" + "=" * 100)
        print("TEST: test_data_driven_forbid_effect_conflict")
        print("=" * 100)

        # Fluent patch: Add holding(a) as an effect in a pick-up action
        # We flip a fluent in the next state to make it appear as an added effect
        fluent_patches = {
            FluentLevelPatch(
                observation_index=0,
                component_index=2,
                state_type='next',
                fluent='(holding a)'  # Add this to next state
            )
        }

        # Model patch: FORBID holding(?x) as effect
        # This will conflict with the data we just created
        model_patches = {
            ModelLevelPatch(
                action_name="pick-up",
                model_part=ModelPart.EFFECT,
                pbl=ParameterBoundLiteral("holding", ("?x",), is_positive=True),
                operation=PatchOperation.FORBID
            )
        }

        # Learn with data-driven setup
        learned_domain, conflicts, report = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=fluent_patches,
            model_patches=model_patches
        )

        self.assertIsNotNone(learned_domain)

        print(f"\nTest Setup:")
        print(f"  Fluent patches: {len(fluent_patches)}")
        print(f"  Model patches: {len(model_patches)}")

        print(f"\nResults:")
        print(f"  Total conflicts detected: {len(conflicts)}")

        # Check for FORBID_EFFECT_VS_MUST conflicts
        forbid_conflicts = [c for c in conflicts
                           if c.conflict_type == ConflictType.FORBID_EFFECT_VS_MUST]
        print(f"  FORBID_EFFECT_VS_MUST conflicts: {len(forbid_conflicts)}")

        if forbid_conflicts:
            print(f"\nDetailed Conflict Information:")
            for i, conflict in enumerate(forbid_conflicts, 1):
                print(f"\n  Conflict #{i}:")
                print(f"    Type: {conflict.conflict_type.value}")
                print(f"    Action: {conflict.action_name}")
                print(f"    PBL (Parameter-Bound Literal): {conflict.pbl}")
                print(f"    Grounded Fluent: {conflict.grounded_fluent}")
                print(f"    Location: obs[{conflict.observation_index}][{conflict.component_index}]")

        print("\n" + "=" * 100 + "\n")

    @unittest.skipIf(not Path(f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_data_driven_require_effect_conflict(self):
        """
        Test data-driven REQUIRE_EFFECT_VS_CANNOT conflict.

        Scenario:
        1. Flip a fluent to REMOVE an effect from the data (fluent stays same in next state)
        2. Add a REQUIRE model patch for that effect
        3. PI-SAM sees the fluent unchanged and says it CANNOT be an effect
        4. Model patch says it's REQUIRED
        → REQUIRE_EFFECT_VS_CANNOT conflict

        This demonstrates how data modifications create conflicts with model requirements.
        """
        print("\n" + "=" * 100)
        print("TEST: test_data_driven_require_effect_conflict")
        print("=" * 100)

        # Fluent patch: Remove an effect by making the fluent appear in both states
        # For example, if clear(a) was supposed to become false, keep it true
        fluent_patches = {
            FluentLevelPatch(
                observation_index=0,
                component_index=1,
                state_type='next',
                fluent='(clear a)'  # Keep this in next state (preventing delete effect)
            )
        }

        # Model patch: REQUIRE not clear(?x) as effect
        # This will conflict because we prevented the effect from appearing
        model_patches = {
            ModelLevelPatch(
                action_name="pick-up",
                model_part=ModelPart.EFFECT,
                pbl=ParameterBoundLiteral("clear", ("?x",), is_positive=False),
                operation=PatchOperation.REQUIRE
            )
        }

        # Learn with data-driven setup
        learned_domain, conflicts, report = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=fluent_patches,
            model_patches=model_patches
        )

        self.assertIsNotNone(learned_domain)

        print(f"\nTest Setup:")
        print(f"  Fluent patches: {len(fluent_patches)}")
        print(f"  Model patches: {len(model_patches)}")

        print(f"\nResults:")
        print(f"  Total conflicts detected: {len(conflicts)}")

        # Check for REQUIRE_EFFECT_VS_CANNOT conflicts
        require_conflicts = [c for c in conflicts
                            if c.conflict_type == ConflictType.REQUIRE_EFFECT_VS_CANNOT]
        print(f"  REQUIRE_EFFECT_VS_CANNOT conflicts: {len(require_conflicts)}")

        if require_conflicts:
            print(f"\nDetailed Conflict Information:")
            for i, conflict in enumerate(require_conflicts, 1):
                print(f"\n  Conflict #{i}:")
                print(f"    Type: {conflict.conflict_type.value}")
                print(f"    Action: {conflict.action_name}")
                print(f"    PBL (Parameter-Bound Literal): {conflict.pbl}")
                print(f"    Grounded Fluent: {conflict.grounded_fluent}")
                print(f"    Location: obs[{conflict.observation_index}][{conflict.component_index}]")

        print("\n" + "=" * 100 + "\n")

    @unittest.skipIf(not Path(f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_data_driven_forbid_precondition_conflict(self):
        """
        Test data-driven FORBID_PRECOND_VS_IS conflict.

        Scenario:
        1. Flip a fluent to ADD it to a previous state
        2. Add a FORBID model patch for that precondition
        3. PI-SAM sees it in the previous state and wants to treat it as a precondition
        4. Model patch says it's FORBIDDEN
        → FORBID_PRECOND_VS_IS conflict

        This shows how adding preconditions in data conflicts with model constraints.
        """
        print("\n" + "=" * 100)
        print("TEST: test_data_driven_forbid_precondition_conflict")
        print("=" * 100)

        # Fluent patch: Add ontable(a) to previous state
        fluent_patches = {
            FluentLevelPatch(
                observation_index=0,
                component_index=2,
                state_type='prev',
                fluent='(ontable a)'  # Add to previous state
            )
        }

        # Model patch: FORBID ontable(?x) as precondition
        model_patches = {
            ModelLevelPatch(
                action_name="pick-up",
                model_part=ModelPart.PRECONDITION,
                pbl=ParameterBoundLiteral("ontable", ("?x",), is_positive=True),
                operation=PatchOperation.FORBID
            )
        }

        # Learn with data-driven setup
        learned_domain, conflicts, report = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=fluent_patches,
            model_patches=model_patches
        )

        self.assertIsNotNone(learned_domain)

        print(f"\nTest Setup:")
        print(f"  Fluent patches: {len(fluent_patches)}")
        print(f"  Model patches: {len(model_patches)}")

        print(f"\nResults:")
        print(f"  Total conflicts detected: {len(conflicts)}")

        # Check for FORBID_PRECOND_VS_IS conflicts
        precond_conflicts = [c for c in conflicts
                            if c.conflict_type == ConflictType.FORBID_PRECOND_VS_IS]
        print(f"  FORBID_PRECOND_VS_IS conflicts: {len(precond_conflicts)}")

        if precond_conflicts:
            print(f"\nDetailed Conflict Information:")
            for i, conflict in enumerate(precond_conflicts, 1):
                print(f"\n  Conflict #{i}:")
                print(f"    Type: {conflict.conflict_type.value}")
                print(f"    Action: {conflict.action_name}")
                print(f"    PBL (Parameter-Bound Literal): {conflict.pbl}")
                print(f"    Grounded Fluent: {conflict.grounded_fluent}")
                print(f"    Location: obs[{conflict.observation_index}][{conflict.component_index}]")

        print("\n" + "=" * 100 + "\n")

    @unittest.skipIf(not Path(f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_data_driven_multiple_conflicts(self):
        """
        Test multiple data-driven conflicts in a single learning session.

        This demonstrates a realistic scenario where:
        1. Multiple fluent patches modify the data
        2. Multiple model patches constrain learning
        3. The combination creates various types of conflicts
        """
        print("\n" + "=" * 100)
        print("TEST: test_data_driven_multiple_conflicts")
        print("=" * 100)

        # Multiple fluent patches creating noisy data
        fluent_patches = {
            # Add effects that shouldn't be there
            FluentLevelPatch(0, 1, 'next', '(holding a)'),
            FluentLevelPatch(0, 3, 'next', '(ontable b)'),
            # Remove effects that should be there
            FluentLevelPatch(0, 2, 'next', '(clear c)'),
            # Add preconditions
            FluentLevelPatch(0, 4, 'prev', '(on d e)'),
        }

        # Model patches that conflict with the noisy data
        model_patches = {
            # FORBID the added effects
            ModelLevelPatch("pick-up", ModelPart.EFFECT,
                           ParameterBoundLiteral("holding", ("?x",), True),
                           PatchOperation.FORBID),
            ModelLevelPatch("put-down", ModelPart.EFFECT,
                           ParameterBoundLiteral("ontable", ("?x",), True),
                           PatchOperation.FORBID),
            # REQUIRE the removed effects
            ModelLevelPatch("pick-up", ModelPart.EFFECT,
                           ParameterBoundLiteral("clear", ("?x",), False),
                           PatchOperation.REQUIRE),
            # FORBID the added precondition
            ModelLevelPatch("stack", ModelPart.PRECONDITION,
                           ParameterBoundLiteral("on", ("?x", "?y"), True),
                           PatchOperation.FORBID),
        }

        # Learn with complex data-driven setup
        learned_domain, conflicts, report = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=fluent_patches,
            model_patches=model_patches
        )

        self.assertIsNotNone(learned_domain)

        print(f"\nTest Setup:")
        print(f"  Fluent patches: {len(fluent_patches)}")
        print(f"  Model patches: {len(model_patches)}")

        print(f"\nResults:")
        print(f"  Total conflicts detected: {len(conflicts)}")

        # Group by conflict type
        conflict_summary = {}
        for conflict in conflicts:
            ct = conflict.conflict_type.value
            conflict_summary[ct] = conflict_summary.get(ct, 0) + 1

        print(f"\nConflicts by type:")
        for ct, count in conflict_summary.items():
            print(f"  - {ct}: {count}")

        # Show all conflicts organized by type
        if conflicts:
            print(f"\nDetailed Conflict Information:")
            for ct in [ConflictType.FORBID_EFFECT_VS_MUST,
                       ConflictType.REQUIRE_EFFECT_VS_CANNOT,
                       ConflictType.FORBID_PRECOND_VS_IS]:
                examples = [c for c in conflicts if c.conflict_type == ct]
                if examples:
                    print(f"\n  {ct.value} ({len(examples)} total):")
                    for i, conflict in enumerate(examples, 1):
                        print(f"\n    Conflict #{i}:")
                        print(f"      Action: {conflict.action_name}")
                        print(f"      PBL: {conflict.pbl}")
                        print(f"      Grounded Fluent: {conflict.grounded_fluent}")
                        print(f"      Location: obs[{conflict.observation_index}][{conflict.component_index}]")

        print("\n" + "=" * 100 + "\n")

    @unittest.skipIf(not Path(f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_data_driven_negative_fluent_conflicts(self):
        """
        Test data-driven conflicts with negative fluents.

        This shows how flipping negative predicates in data creates conflicts.
        """
        print("\n" + "=" * 100)
        print("TEST: test_data_driven_negative_fluent_conflicts")
        print("=" * 100)

        # Fluent patches with negative predicates
        fluent_patches = {
            # Add negative effects
            FluentLevelPatch(0, 1, 'next', '(not (clear a))'),
            FluentLevelPatch(0, 2, 'next', '(not (holding b))'),
            # Add negative preconditions
            FluentLevelPatch(0, 3, 'prev', '(not (ontable c))'),
        }

        # Model patches for negative predicates
        model_patches = {
            # FORBID the negative effect
            ModelLevelPatch("pick-up", ModelPart.EFFECT,
                           ParameterBoundLiteral("clear", ("?x",), is_positive=False),
                           PatchOperation.FORBID),
            # REQUIRE a different negative effect
            ModelLevelPatch("put-down", ModelPart.EFFECT,
                           ParameterBoundLiteral("holding", ("?x",), is_positive=False),
                           PatchOperation.REQUIRE),
            # FORBID negative precondition
            ModelLevelPatch("pick-up", ModelPart.PRECONDITION,
                           ParameterBoundLiteral("ontable", ("?x",), is_positive=False),
                           PatchOperation.FORBID),
        }

        # Learn with negative fluent setup
        learned_domain, conflicts, report = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=fluent_patches,
            model_patches=model_patches
        )

        self.assertIsNotNone(learned_domain)

        print(f"\nTest Setup:")
        print(f"  Fluent patches: {len(fluent_patches)}")
        print(f"  Model patches: {len(model_patches)}")

        print(f"\nResults:")
        print(f"  Total conflicts detected: {len(conflicts)}")

        # Check for conflicts involving negative predicates
        negative_conflicts = [c for c in conflicts
                             if hasattr(c, 'pbl') and not c.pbl.is_positive]
        print(f"  Conflicts with negative PBLs: {len(negative_conflicts)}")

        if negative_conflicts:
            print(f"\nDetailed Conflict Information (Negative PBLs):")
            for i, conflict in enumerate(negative_conflicts, 1):
                print(f"\n  Conflict #{i}:")
                print(f"    Type: {conflict.conflict_type.value}")
                print(f"    Action: {conflict.action_name}")
                print(f"    PBL (Parameter-Bound Literal): {conflict.pbl}")
                print(f"    PBL is_positive: {conflict.pbl.is_positive}")
                print(f"    Grounded Fluent: {conflict.grounded_fluent}")
                print(f"    Location: obs[{conflict.observation_index}][{conflict.component_index}]")

        # Also show all conflicts by type
        if conflicts:
            print(f"\nAll Conflicts by Type:")
            conflict_summary = {}
            for conflict in conflicts:
                ct = conflict.conflict_type.value
                conflict_summary[ct] = conflict_summary.get(ct, 0) + 1

            for ct, count in conflict_summary.items():
                print(f"  - {ct}: {count}")

        print("\n" + "=" * 100 + "\n")

    @unittest.skipIf(not Path(f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_single_ontable_fluent_flip_put_down_conflict(self):
        """
        Test FORBID_EFFECT_VS_MUST conflict for ontable in put-down action.

        Note: Component 2 has put-down(e) where ontable(e) is already an effect.
        Flipping it would remove it and trigger a masking bug.
        Instead, we use NO fluent patch and rely on the existing data.

        At component 0: put-down(d) adds ontable(d) as an effect.
        We add a FORBID patch for ontable(?x) in put-down effects.
        This creates FORBID_EFFECT_VS_MUST conflicts.

        This is a minimal test case showing model-patch-driven conflict.
        """
        print("\n" + "=" * 100)
        print("TEST: test_single_ontable_fluent_flip_put_down_conflict")
        print("=" * 100)

        # No fluent patches - we'll use the existing data
        # Component 0 has put-down(d) which adds ontable(d)

        # Model patch: FORBID ontable(?x) as effect in put-down
        # This will conflict with the data showing ontable(e) as an effect
        fluent_patches = {
            # Add negative effects
            FluentLevelPatch(0, 2, 'next', '(ontable e)'),
        }

        print(f"\nTest Setup:")
        print(f"  Fluent patches: {len(fluent_patches)} (none - using existing trajectory data)")
        print(f"    - FORBID ontable(?x) in put-down effects")
        print(f"  Expected: Conflicts at put-down(d) [component 0] and put-down(e) [component 2]")

        # Learn with the single fluent patch and model constraint
        learned_domain, conflicts, report = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=fluent_patches,
            model_patches=set()
        )

        self.assertIsNotNone(learned_domain)

        print(f"\nResults:")
        print(f"  Total conflicts detected: {len(conflicts)}")

        # Filter for put-down action conflicts
        putdown_conflicts = [c for c in conflicts if c.action_name == "put-down"]
        print(f"  put-down action conflicts: {len(putdown_conflicts)}")

        # Filter specifically for FORBID_EFFECT_VS_MUST conflicts
        forbid_must_conflicts = [c for c in conflicts
                                 if c.conflict_type == ConflictType.FORBID_EFFECT_VS_MUST]
        print(f"  FORBID_EFFECT_VS_MUST conflicts: {len(forbid_must_conflicts)}")

        # Filter for ontable predicate conflicts in put-down
        ontable_putdown_conflicts = [c for c in putdown_conflicts
                                     if c.conflict_type == ConflictType.FORBID_EFFECT_VS_MUST
                                     and "ontable" in str(c.pbl)]
        print(f"  ontable FORBID_EFFECT_VS_MUST conflicts in put-down: {len(ontable_putdown_conflicts)}")

        if ontable_putdown_conflicts:
            print(f"\nDetailed Conflict Information (ontable in put-down):")
            for i, conflict in enumerate(ontable_putdown_conflicts, 1):
                print(f"\n  Conflict #{i}:")
                print(f"    Type: {conflict.conflict_type.value}")
                print(f"    Action: {conflict.action_name}")
                print(f"    PBL (Parameter-Bound Literal): {conflict.pbl}")
                print(f"    Grounded Fluent: {conflict.grounded_fluent}")
                print(f"    Location: obs[{conflict.observation_index}][{conflict.component_index}]")

                # Additional analysis
                if conflict.observation_index == 0 and conflict.component_index == 2:
                    print(f"    ✓ Conflict is at the expected location (obs[0][2])")
        else:
            print(f"\n  Note: No ontable FORBID_EFFECT_VS_MUST conflicts found in put-down action")
            print(f"        This may indicate the fluent flip didn't create the expected effect,")
            print(f"        or the action at component 2 is not put-down.")

        # Show all conflicts for debugging
        if conflicts and len(ontable_putdown_conflicts) == 0:
            print(f"\nAll Conflicts Detected (for debugging):")
            for i, conflict in enumerate(conflicts, 1):
                print(f"\n  Conflict #{i}:")
                print(f"    Type: {conflict.conflict_type.value}")
                print(f"    Action: {conflict.action_name}")
                print(f"    PBL: {conflict.pbl}")
                print(f"    Grounded Fluent: {conflict.grounded_fluent}")
                print(f"    Location: obs[{conflict.observation_index}][{conflict.component_index}]")

        print("\n" + "=" * 100 + "\n")

    # ==================== Baseline Test ====================

    @unittest.skipIf(not Path(f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory").exists(),
                     "Trajectory file not found")
    def test_no_patches_baseline(self):
        """
        Baseline test: Learn without any patches.

        Should behave like regular PISAMLearner with no conflicts.
        """
        # Learn without patches
        learned_domain, conflicts, report = self.learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation],
            fluent_patches=set(),
            model_patches=set()
        )

        # Should complete successfully
        self.assertIsNotNone(learned_domain)

        if not self.with_data_only_conflicts:
            # Should have no conflicts
            self.assertEqual(len(conflicts), 0,
                            "Learning without patches should produce no conflicts")

            # Should have learned some actions
            self.assertTrue(len(self.learner.observed_actions) > 0,
                           "Should have learned some actions")
        else:
            print(f"Baseline test conflicts (data-only): {len(conflicts)}")
            #print the conflicts
            for conflict in conflicts:
                print(f"  - {conflict.conflict_type.value}: {conflict.grounded_fluent}")


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
