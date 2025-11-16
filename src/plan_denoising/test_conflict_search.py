"""
Test suite for ConflictDrivenPatchSearch.

This test file demonstrates the complete workflow:
1. Load problem7 with trajectory and masking
2. Solve with regular PISAM and show report
3. Solve with SimpleNoisyPisamLearner (no patches) and show report
4. Verify that regular PISAM and SimpleNoisyPisamLearner produce equal results
5. Run ConflictDrivenPatchSearch and show report
6. Generate conflict-tree traversal visualization file

Test Data:
- Problem: problem7
- Trajectory: src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory
- Masking: src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.masking_info
"""

import unittest
from pathlib import Path
from copy import deepcopy
from typing import List, Dict, Set
from datetime import datetime

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser
from pddl_plus_parser.models import Domain, Observation
from utilities import NegativePreconditionPolicy

from src.pi_sam.pi_sam_learning import PISAMLearner
from src.pi_sam.noisy_pisam.simpler_version.simple_noisy_pisam_learning import NoisyPisamLearner
from src.pi_sam.noisy_pisam.simpler_version.typings import (
    FluentLevelPatch,
    ModelLevelPatch,
    Conflict,
)
from src.plan_denoising.conflict_search import ConflictDrivenPatchSearch, Key
from src.utils.pddl import ground_observation_completely
from src.utils.masking import mask_observation, load_masking_info

absulute_path_prefix = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/")


class TestConflictDrivenPatchSearch(unittest.TestCase):
    """Test suite for ConflictDrivenPatchSearch."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests."""
        # Load blocks domain
        cls.domain_file = absulute_path_prefix / Path("src/domains/blocks/blocks.pddl")
        cls.domain: Domain = DomainParser(cls.domain_file).parse_domain()

        # Load problem7 trajectory and masking info
        cls.experiment_dir = absulute_path_prefix / Path(
            "src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06"
        )
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

    @unittest.skipIf(
        not Path(
            f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory"
        ).exists(),
        "Trajectory file not found",
    )
    def test_1_regular_pisam(self):
        """
        Test 1: Solve with regular PISAM and show report.
        """
        print("\n" + "=" * 80)
        print("TEST 1: Regular PISAM Learning")
        print("=" * 80)

        # Create regular PISAM learner
        learner = PISAMLearner(
            deepcopy(self.domain), negative_preconditions_policy=NegativePreconditionPolicy.hard
        )

        # Learn action model
        learned_domain, report = learner.learn_action_model(observations=[self.masked_observation])

        print(f"\n{'─' * 80}")
        print("PISAM Learning Report:")
        print(f"{'─' * 80}")
        for key, value in report.items():
            print(f"  {key}: {value}")

        print(f"\n{'─' * 80}")
        print("Learned Actions:")
        print(f"{'─' * 80}")
        for action_name, action in learned_domain.actions.items():
            print(f"\n  Action: {action_name}")
            print(f"    Preconditions: {len(action.preconditions.root.operands)} literals")
            for prec in action.preconditions.root.operands:
                print(prec)
            print("-------------")
            print(f"    Effects: {len(action.discrete_effects)} effects")
            for eff in action.discrete_effects:
                print(eff)

        self.assertIsNotNone(learned_domain)
        self.assertTrue(len(learner.observed_actions) > 0)

        # Store for later comparison
        self.__class__.pisam_domain = learned_domain
        self.__class__.pisam_report = report

    @unittest.skipIf(
        not Path(
            f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory"
        ).exists(),
        "Trajectory file not found",
    )
    def test_2_simple_noisy_pisam_no_patches(self):
        """
        Test 2: Solve with SimpleNoisyPisamLearner (no patches) and show report.
        """
        print("\n" + "=" * 80)
        print("TEST 2: SimpleNoisyPisamLearner (No Patches)")
        print("=" * 80)

        # Create SimpleNoisyPisamLearner
        learner = NoisyPisamLearner(
            deepcopy(self.domain), negative_preconditions_policy=NegativePreconditionPolicy.hard
        )

        # Learn without patches
        learned_domain, conflicts, report = learner.learn_action_model_with_conflicts(
            observations=[self.masked_observation], fluent_patches=set(), model_patches=set()
        )

        print(f"\n{'─' * 80}")
        print("SimpleNoisyPisamLearner Report:")
        print(f"{'─' * 80}")
        for key, value in report.items():
            print(f"  {key}: {value}")

        print(f"\n{'─' * 80}")
        print(f"Conflicts Detected: {len(conflicts)}")
        print(f"{'─' * 80}")

        print(f"\n{'─' * 80}")
        print("Learned Actions:")
        print(f"{'─' * 80}")
        for action_name, action in learned_domain.actions.items():
            print(f"\n  Action: {action_name}")
            print(f"    Preconditions: {len(action.preconditions.root.operands)} literals")
            print(f"    Effects: {len(action.discrete_effects)} effects")

        self.assertIsNotNone(learned_domain)
        self.assertEqual(len(conflicts), 0, "Learning without patches should produce no conflicts")

        # Store for later comparison
        self.__class__.simple_noisy_domain = learned_domain
        self.__class__.simple_noisy_report = report

    @unittest.skipIf(
        not Path(
            f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory"
        ).exists(),
        "Trajectory file not found",
    )
    def test_3_compare_pisam_and_simple_noisy(self):
        """
        Test 3: Verify that regular PISAM and SimpleNoisyPisamLearner produce equal results.
        """
        print("\n" + "=" * 80)
        print("TEST 3: Comparing PISAM and SimpleNoisyPisamLearner")
        print("=" * 80)

        # Get domains from previous tests
        pisam_domain = getattr(self.__class__, "pisam_domain", None)
        simple_noisy_domain = getattr(self.__class__, "simple_noisy_domain", None)

        if pisam_domain is None or simple_noisy_domain is None:
            self.skipTest("Previous tests did not run successfully")

        print(f"\n{'─' * 80}")
        print("Comparison Results:")
        print(f"{'─' * 80}")

        # Compare number of actions
        pisam_actions = set(pisam_domain.actions.keys())
        simple_noisy_actions = set(simple_noisy_domain.actions.keys())

        print(f"\n  Actions in PISAM: {len(pisam_actions)}")
        print(f"  Actions in SimpleNoisy: {len(simple_noisy_actions)}")

        if pisam_actions == simple_noisy_actions:
            print(f"  ✓ Action sets are EQUAL")
        else:
            print(f"  ✗ Action sets DIFFER")
            print(f"    Only in PISAM: {pisam_actions - simple_noisy_actions}")
            print(f"    Only in SimpleNoisy: {simple_noisy_actions - pisam_actions}")

        # Compare each action's structure
        print(f"\n{'─' * 80}")
        print("Action-by-Action Comparison:")
        print(f"{'─' * 80}")

        for action_name in pisam_actions & simple_noisy_actions:
            pisam_action = pisam_domain.actions[action_name]
            simple_action = simple_noisy_domain.actions[action_name]

            pisam_pre_count = len(pisam_action.preconditions.root.operands)
            simple_pre_count = len(simple_action.preconditions.root.operands)

            pisam_eff_count = len(pisam_action.discrete_effects)
            simple_eff_count = len(simple_action.discrete_effects)

            match = pisam_pre_count == simple_pre_count and pisam_eff_count == simple_eff_count
            status = "✓" if match else "✗"

            print(f"\n  {status} {action_name}:")
            print(f"      Preconditions: PISAM={pisam_pre_count}, SimpleNoisy={simple_pre_count}")
            print(f"      Effects: PISAM={pisam_eff_count}, SimpleNoisy={simple_eff_count}")

        # Assert equality
        self.assertEqual(
            pisam_actions,
            simple_noisy_actions,
            "PISAM and SimpleNoisyPisamLearner should learn the same actions",
        )

    @unittest.skipIf(
        not Path(
            f"{absulute_path_prefix}/src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/problem7.trajectory"
        ).exists(),
        "Trajectory file not found",
    )
    def test_4_conflict_search_with_tree_visualization(self):
        """
        Test 4: Run ConflictDrivenPatchSearch and generate conflict-tree traversal file.
        """
        print("\n" + "=" * 80)
        print("TEST 4: ConflictDrivenPatchSearch with Tree Visualization")
        print("=" * 80)

        # Track search progress for visualization
        tree_log = ConflictTreeLogger()

        # Create search instance with logger
        search = ConflictDrivenPatchSearch(
            partial_domain_template=deepcopy(self.domain),
            negative_preconditions_policy=NegativePreconditionPolicy.hard,
            seed=42,
            logger=tree_log,
        )

        print(f"\n{'─' * 80}")
        print("Running Conflict-Driven Patch Search...")
        print(f"{'─' * 80}")

        # Run search with no max_nodes limit
        learned_domain, conflicts, model_constraints, fluent_patches, report = search.run(
            observations=[self.masked_observation], max_nodes=None
        )

        print(f"\n{'─' * 80}")
        print("Search Results:")
        print(f"{'─' * 80}")
        print(f"  Nodes Expanded: {tree_log.nodes_expanded}")
        print(f"  Solution Found: {len(conflicts) == 0}")
        print(f"  Final Conflicts: {len(conflicts)}")
        print(f"  Model Constraints: {len(model_constraints)}")
        print(f"  Fluent Patches: {len(fluent_patches)}")

        if len(conflicts) == 0:
            print(f"\n  ✓ Conflict-free model found!")
        else:
            print(f"\n  ✗ No conflict-free model found within search limits")

        # Show model constraints
        print(f"\n{'─' * 80}")
        print("Model Constraints Applied:")
        print(f"{'─' * 80}")
        for (action_name, part, pbl), op in model_constraints.items():
            print(f"  {op.value.upper()} {pbl} in {part.value} of {action_name}")

        # Show fluent patches
        print(f"\n{'─' * 80}")
        print("Fluent Patches Applied:")
        print(f"{'─' * 80}")
        for patch in fluent_patches:
            print(
                f"  Flip '{patch.fluent}' at obs[{patch.observation_index}][{patch.component_index}].{patch.state_type}"
            )

        # Show final conflicts (if any)
        if conflicts:
            print(f"\n{'─' * 80}")
            print("Remaining Conflicts:")
            print(f"{'─' * 80}")
            for i, conflict in enumerate(conflicts, 1):
                print(f"  {i}. {conflict.conflict_type.value}:")
                print(f"     Action: {conflict.action_name}")
                print(f"     PBL: {conflict.pbl}")
                print(f"     Grounded: {conflict.grounded_fluent}")
                print(f"     Location: obs[{conflict.observation_index}][{conflict.component_index}]")

        # Generate tree visualization file
        output_file = self.experiment_dir / "conflict_search_tree.txt"
        tree_log.save_to_file(output_file)
        print(f"\n{'─' * 80}")
        print(f"Conflict Tree saved to: {output_file}")
        print(f"{'─' * 80}")

        # Show learned actions
        print(f"\n{'─' * 80}")
        print("Learned Actions:")
        print(f"{'─' * 80}")
        for action_name, action in learned_domain.actions.items():
            print(f"\n  Action: {action_name}")
            print(f"    Preconditions: {len(action.preconditions.root.operands)} literals")
            print(f"    Effects: {len(action.discrete_effects)} effects")

        self.assertIsNotNone(learned_domain)


class ConflictTreeLogger:
    """Logger to track conflict-driven search tree traversal."""

    def __init__(self):
        self.nodes_expanded = 0
        self.tree_structure: List[Dict] = []
        self.start_time = datetime.now()

    def log_node(
        self,
        node_id: int,
        depth: int,
        cost: int,
        model_constraints: Dict[Key, any],
        fluent_patches: Set[FluentLevelPatch],
        conflicts: List[Conflict],
        is_solution: bool,
    ):
        """Log expansion of a search node."""
        self.nodes_expanded += 1
        self.tree_structure.append(
            {
                "node_id": node_id,
                "depth": depth,
                "cost": cost,
                "model_constraints": len(model_constraints),
                "fluent_patches": len(fluent_patches),
                "conflicts": len(conflicts),
                "is_solution": is_solution,
                "constraints_detail": dict(model_constraints),
                "fluent_detail": set(fluent_patches),
                "conflicts_detail": conflicts,
            }
        )

    def save_to_file(self, filepath: Path):
        """Save tree traversal to a human-readable file."""
        with open(filepath, "w") as f:
            f.write("=" * 100 + "\n")
            f.write("CONFLICT-DRIVEN PATCH SEARCH - TREE TRAVERSAL\n")
            f.write("=" * 100 + "\n")
            f.write(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Nodes Expanded: {self.nodes_expanded}\n")
            f.write("=" * 100 + "\n\n")

            for i, node_info in enumerate(self.tree_structure):
                indent = "  " * node_info["depth"]

                f.write(f"{'─' * 100}\n")
                f.write(f"{indent}Node #{node_info['node_id']} (Depth={node_info['depth']})\n")
                f.write(f"{'─' * 100}\n")
                f.write(f"{indent}Cost: {node_info['cost']}\n")
                f.write(
                    f"{indent}Patches: {node_info['model_constraints']} model + {node_info['fluent_patches']} fluent\n"
                )
                f.write(f"{indent}Conflicts: {node_info['conflicts']}\n")
                f.write(
                    f"{indent}Status: {'✓ SOLUTION' if node_info['is_solution'] else '✗ Has Conflicts'}\n"
                )

                # Show model constraints
                if node_info["constraints_detail"]:
                    f.write(f"\n{indent}Model Constraints:\n")
                    for (action_name, part, pbl), op in node_info["constraints_detail"].items():
                        f.write(f"{indent}  - {op.value.upper()} {pbl} in {part.value} of {action_name}\n")

                # Show fluent patches
                if node_info["fluent_detail"]:
                    f.write(f"\n{indent}Fluent Patches:\n")
                    for patch in node_info["fluent_detail"]:
                        f.write(
                            f"{indent}  - Flip '{patch.fluent}' at obs[{patch.observation_index}][{patch.component_index}].{patch.state_type}\n"
                        )

                # Show conflicts
                if node_info["conflicts_detail"]:
                    f.write(f"\n{indent}Conflicts:\n")
                    for j, conflict in enumerate(node_info["conflicts_detail"], 1):
                        f.write(f"{indent}  {j}. {conflict.conflict_type.value}\n")
                        f.write(f"{indent}     Action: {conflict.action_name}\n")
                        f.write(f"{indent}     PBL: {conflict.pbl}\n")
                        f.write(f"{indent}     Grounded: {conflict.grounded_fluent}\n")
                        f.write(
                            f"{indent}     Location: obs[{conflict.observation_index}][{conflict.component_index}]\n"
                        )

                if node_info["is_solution"]:
                    f.write(f"\n{indent}{'★' * 50}\n")
                    f.write(f"{indent}SOLUTION FOUND!\n")
                    f.write(f"{indent}{'★' * 50}\n")

                f.write("\n")

            f.write("=" * 100 + "\n")
            f.write("END OF TREE TRAVERSAL\n")
            f.write("=" * 100 + "\n")


def run_single_test(test_name):
    """Helper function to run a single test."""
    suite = unittest.TestLoader().loadTestsFromName(
        f"__main__.TestConflictDrivenPatchSearch.{test_name}"
    )
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    # Run all tests in order
    unittest.main(verbosity=2)
