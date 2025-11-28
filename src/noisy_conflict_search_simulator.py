"""
Noisy Conflict Search Simulator for VIP-vision-PDDL.

This simulator extends the regular simulator to use conflict-driven patch search
during cross-validation instead of regular PI-SAM learning. It detects and resolves
conflicts in noisy observations by finding minimal patches (data flips and model constraints).
"""

import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from copy import deepcopy

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Observation
from sam_learning.core import LearnerDomain
from utilities import NegativePreconditionPolicy

from src.pi_sam.pisam_experiment_runner import OfflinePiSamExperimentRunner
from src.plan_denoising.conflict_search import ConflictDrivenPatchSearch
from src.simulator import Simulator


class NoisyConflictSearchExperimentRunner(OfflinePiSamExperimentRunner):
    """
    Experiment runner that uses conflict-driven patch search instead of regular PI-SAM.

    This runner:
    - Inherits trajectory loading and masking from OfflinePiSamExperimentRunner
    - Overrides the learning algorithm to use ConflictDrivenPatchSearch
    - Enriches the learning report with patch diff information (added/removed/changed patches)
    """

    def __init__(
        self,
        working_directory_path: Path,
        domain_file_name: str,
        problem_prefix: str = "problem",
        n_split: int = 5,
        fluent_patch_cost: int = 1,
        model_patch_cost: int = 1,
        max_search_nodes: int = None,
        seed: int = 42
    ):
        """
        Initialize the noisy conflict search experiment runner.

        :param working_directory_path: Directory containing experiment files.
        :param domain_file_name: Name of the domain file.
        :param problem_prefix: Prefix for problem files.
        :param n_split: Number of cross-validation folds.
        :param fluent_patch_cost: Cost of adding a fluent patch.
        :param model_patch_cost: Cost of adding a model constraint.
        :param max_search_nodes: Maximum nodes to explore in search (None = unlimited).
        :param seed: Random seed for reproducibility.
        """
        super().__init__(
            working_directory_path=working_directory_path,
            domain_file_name=domain_file_name,
            problem_prefix=problem_prefix,
            n_split=n_split
        )

        self.fluent_patch_cost = fluent_patch_cost
        self.model_patch_cost = model_patch_cost
        self.max_search_nodes = max_search_nodes
        self.seed = seed

        self.logger.info(f"Initialized NoisyConflictSearchExperimentRunner")
        self.logger.info(f"  Fluent patch cost: {fluent_patch_cost}")
        self.logger.info(f"  Model patch cost: {model_patch_cost}")
        self.logger.info(f"  Max search nodes: {max_search_nodes if max_search_nodes else 'unlimited'}")
        self.logger.info(f"  Random seed: {seed}")

    def _apply_learning_algorithm(
        self, partial_domain: Domain, allowed_observations: List[Observation], test_set_dir_path: Path
    ) -> Tuple[LearnerDomain, Dict[str, str]]:
        """
        Learns the action model using conflict-driven patch search.

        This method replaces PI-SAM learning with conflict-driven search that:
        1. Detects conflicts between observations and learned model
        2. Explores patch space (data flips + model constraints)
        3. Finds minimal set of patches for conflict-free learning

        The learning report is enriched with patch diff information showing
        what corrections were needed.

        :param partial_domain: The partial domain without actions' preconditions and effects.
        :param allowed_observations: The allowed observations (already grounded and masked).
        :param test_set_dir_path: The path to the directory containing test problems.
        :return: The learned action model and enriched learning report.
        """
        self.logger.info(f"Starting conflict-driven patch search with {len(allowed_observations)} observations")

        # Create conflict-driven search instance
        search = ConflictDrivenPatchSearch(
            partial_domain_template=deepcopy(partial_domain),
            negative_preconditions_policy=self.negative_precondition_policy,
            fluent_patch_cost=self.fluent_patch_cost,
            model_patch_cost=self.model_patch_cost,
            seed=self.seed,
            logger=None  # No tree logging during cross-validation for performance
        )

        # Run search on all observations
        learned_domain, conflicts, model_constraints, fluent_patches, cost, report = search.run(
            observations=allowed_observations,
            max_nodes=self.max_search_nodes
        )

        # Extract patch diff from report
        patch_diff = report.get("patch_diff", {})

        # Enrich learning report with patch statistics
        enriched_report = dict(report)

        # Add high-level patch statistics
        enriched_report["solution_found"] = str(len(conflicts) == 0)
        enriched_report["final_conflicts"] = str(len(conflicts))
        enriched_report["solution_cost"] = str(cost)
        enriched_report["total_model_constraints"] = str(len(model_constraints))
        enriched_report["total_fluent_patches"] = str(len(fluent_patches))

        # Add detailed patch diff statistics
        model_added = patch_diff.get("model_patches_added", {})
        model_removed = patch_diff.get("model_patches_removed", {})
        model_changed = patch_diff.get("model_patches_changed", {})
        fluent_added = patch_diff.get("fluent_patches_added", set())
        fluent_removed = patch_diff.get("fluent_patches_removed", set())

        enriched_report["model_patches_added"] = str(len(model_added))
        enriched_report["model_patches_removed"] = str(len(model_removed))
        enriched_report["model_patches_changed"] = str(len(model_changed))
        enriched_report["fluent_patches_added"] = str(len(fluent_added))
        enriched_report["fluent_patches_removed"] = str(len(fluent_removed))

        # Add detailed patch information as string lists for CSV export
        if model_added:
            enriched_report["model_patches_added_detail"] = "; ".join([
                f"{op.value.upper()} {pbl} in {part.value} of {action}"
                for (action, part, pbl), op in model_added.items()
            ])

        if model_removed:
            enriched_report["model_patches_removed_detail"] = "; ".join([
                f"{op.value.upper()} {pbl} in {part.value} of {action}"
                for (action, part, pbl), op in model_removed.items()
            ])

        if fluent_added:
            enriched_report["fluent_patches_added_detail"] = "; ".join([
                f"{patch.fluent} at obs[{patch.observation_index}][{patch.component_index}].{patch.state_type}"
                for patch in sorted(fluent_added, key=lambda p: (p.observation_index, p.component_index))
            ])

        if fluent_removed:
            enriched_report["fluent_patches_removed_detail"] = "; ".join([
                f"{patch.fluent} at obs[{patch.observation_index}][{patch.component_index}].{patch.state_type}"
                for patch in sorted(fluent_removed, key=lambda p: (p.observation_index, p.component_index))
            ])

        # Log results
        self.logger.info(f"Conflict-driven search completed:")
        self.logger.info(f"  Solution found: {len(conflicts) == 0}")
        self.logger.info(f"  Final conflicts: {len(conflicts)}")
        self.logger.info(f"  Solution cost: {cost}")
        self.logger.info(f"  Model constraints: {len(model_constraints)}")
        self.logger.info(f"  Fluent patches: {len(fluent_patches)}")
        self.logger.info(f"  Patches added: {len(model_added)} model + {len(fluent_added)} fluent")
        self.logger.info(f"  Patches removed: {len(model_removed)} model + {len(fluent_removed)} fluent")

        if len(conflicts) > 0:
            self.logger.warning(f"  Warning: {len(conflicts)} conflicts remain unresolved")

        return learned_domain, enriched_report


class NoisyConflictSearchSimulator(Simulator):
    """
    Simulator that uses conflict-driven patch search for cross-validation.

    This extends the base Simulator to use NoisyConflictSearchExperimentRunner
    instead of OfflinePiSamExperimentRunner during cross-validation.
    """

    def __init__(
        self,
        domain_name: str,
        openai_apikey: str,
        pddl_domain_file: Path,
        pddl_problem_dir: Path,
        object_detection_model_name: str,
        object_detection_temperature: float,
        fluent_classification_model_name: str,
        fluent_classification_temperature: float,
        experiment_dir_path: Path = Path("noisy_conflict_experiments"),
        fluent_patch_cost: int = 1,
        model_patch_cost: int = 1,
        max_search_nodes: int = None,
        seed: int = 42
    ):
        """
        Initialize the noisy conflict search simulator.

        :param domain_name: Name of the PDDL gym domain (e.g., 'PDDLEnvBlocks-v0').
        :param openai_apikey: OpenAI API key for LLM-based components.
        :param pddl_domain_file: Path to the PDDL domain file.
        :param pddl_problem_dir: Directory containing PDDL problem files.
        :param object_detection_model_name: LLM model name for object detection.
        :param object_detection_temperature: Temperature for object detection LLM.
        :param fluent_classification_model_name: LLM model name for fluent classification.
        :param fluent_classification_temperature: Temperature for fluent classification LLM.
        :param experiment_dir_path: Directory for experiment outputs.
        :param fluent_patch_cost: Cost of adding a fluent patch in search.
        :param model_patch_cost: Cost of adding a model constraint in search.
        :param max_search_nodes: Maximum nodes to explore in search (None = unlimited).
        :param seed: Random seed for reproducibility.
        """
        super().__init__(
            domain_name=domain_name,
            openai_apikey=openai_apikey,
            pddl_domain_file=pddl_domain_file,
            pddl_problem_dir=pddl_problem_dir,
            object_detection_model_name=object_detection_model_name,
            object_detection_temperature=object_detection_temperature,
            fluent_classification_model_name=fluent_classification_model_name,
            fluent_classification_temperature=fluent_classification_temperature,
            experiment_dir_path=experiment_dir_path
        )

        self.fluent_patch_cost = fluent_patch_cost
        self.model_patch_cost = model_patch_cost
        self.max_search_nodes = max_search_nodes
        self.seed = seed

        print(f"NoisyConflictSearchSimulator initialized")
        print(f"  Fluent patch cost: {fluent_patch_cost}")
        print(f"  Model patch cost: {model_patch_cost}")
        print(f"  Max search nodes: {max_search_nodes if max_search_nodes else 'unlimited'}")
        print(f"  Random seed: {seed}")

    def run_cross_validation_with_conflict_search(
        self,
        problems: List[str],
        num_steps: int = 25,
        experiment_name: str = "noisy_conflict_cv"
    ) -> Path:
        """
        Run complete cross-validation pipeline with conflict-driven search.

        This function:
        1. Generates trajectories for all problems using LLM (sequentially) - same as base
        2. Uses LLM-derived masking info (unknown predicates) - same as base
        3. Runs cross-validation with NoisyConflictSearchExperimentRunner - NEW

        :param problems: List of problem names to process.
        :param num_steps: Number of steps per trajectory.
        :param experiment_name: Name prefix for the experiment directory.
        :return: Path to experiment results directory.
        """
        from time import time

        print(f"="*80)
        print(f"Running Cross-Validation with Conflict-Driven Patch Search")
        print(f"Domain: {self.domain_name}")
        print(f"Problems: {problems}")
        print(f"Steps per trajectory: {num_steps}")
        print(f"="*80)

        # Use the parent's trajectory generation (identical to base Simulator)
        # This returns the working directory with all trajectories and masking info
        from src.simulator import _create_single_problem_trajectory
        from src.utils.time import create_experiment_timestamp

        # Create working directory for this experiment run
        timestamp = create_experiment_timestamp()
        working_dir = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/noisy_conflict_experiments/noisy_conflict_cv__steps=20__model=gpt-4.1__temp=1.0__23-11-2025T01:35:28")
        # working_dir = self.experiment_dir_path / f"{experiment_name}__steps={num_steps}__model={self.fluent_classification_model_name}__temp={self.fluent_classification_temperature}__{timestamp}"
        # working_dir.mkdir(parents=True, exist_ok=True)
        #
        # print(f"\nWorking directory: {working_dir}")
        #
        # # Step 1: Generate trajectories for all problems using LLM (sequentially)
        # print(f"\n{'='*80}")
        # print(f"Step 1: Generating Trajectories with LLM-based detection/classification (SEQUENTIAL)")
        # print(f"{'='*80}\n")
        #
        # successful_problems = []
        # failed_problems = []
        #
        # print(f"Processing {len(problems)} problems sequentially...")
        # trajectory_start_time = time()
        #
        # # Process problems one by one
        # for i, problem_name in enumerate(problems, 1):
        #     print(f"\nProcessing problem {i}/{len(problems)}: {problem_name}")
        #     problem_result, success = _create_single_problem_trajectory(
        #         self.domain_name,
        #         problem_name,
        #         num_steps,
        #         working_dir,
        #         i,
        #         len(problems),
        #         self.openai_apikey,
        #         self.object_detection_model_name,
        #         self.object_detection_temperature,
        #         self.fluent_classification_model_name,
        #         self.fluent_classification_temperature,
        #         self.pddl_domain_file,
        #         self.problem_dir
        #     )
        #
        #     if success:
        #         successful_problems.append(problem_name)
        #     else:
        #         failed_problems.append(problem_name)
        #
        # trajectory_end_time = time()
        #
        # # Report results
        # print(f"\n{'='*80}")
        # print(f"Trajectory Generation Complete")
        # print(f"{'='*80}")
        # print(f"✓ Successful: {len(successful_problems)}/{len(problems)}")
        # if failed_problems:
        #     print(f"✗ Failed: {len(failed_problems)}/{len(problems)}")
        #     print(f"  Failed problems: {', '.join(failed_problems)}")
        # print(f"Total trajectory generation time: {trajectory_end_time - trajectory_start_time:.2f} seconds")
        # print()
        #
        # # Step 2: Copy domain file to working directory
        # print(f"\n{'='*80}")
        # print(f"Step 2: Preparing Experiment Files")
        # print(f"{'='*80}\n")
        #
        # shutil.copy(self.pddl_domain_file, working_dir / self.pddl_domain_file.name)
        # print(f"✓ Copied domain file: {self.pddl_domain_file.name}")
        #
        # # Step 3: Run cross-validation with NoisyConflictSearchExperimentRunner
        # print(f"\n{'='*80}")
        # print(f"Step 3: Running Conflict-Driven Search Cross-Validation")
        # print(f"{'='*80}\n")
        experiment_runner = NoisyConflictSearchExperimentRunner(
            working_directory_path=working_dir,
            domain_file_name=self.pddl_domain_file.name,
            problem_prefix="problem",
            fluent_patch_cost=self.fluent_patch_cost,
            model_patch_cost=self.model_patch_cost,
            max_search_nodes=self.max_search_nodes,
            seed=self.seed
        )

        experiment_runner.run_cross_validation()

        print(f"\n{'='*80}")
        print(f"Conflict-Driven Search Cross-Validation Complete!")
        print(f"{'='*80}")

        # Check for results
        results_file = working_dir / "results_directory/sam_learning_blocks_combined_semantic_performance.csv"
        if results_file.exists():
            print(f"✓ Results saved to: {results_file}")
        else:
            print(f"⚠ Warning: Results file not found at expected location")

        print(f"\nAll experiment files saved to: {working_dir}")

        return working_dir


# Example usage
if __name__ == '__main__':
    from src.utils.config import load_config

    # Load configuration
    config = load_config()

    # Get API key from config
    openai_apikey = config['openai']['api_key']

    domain = 'blocks'
    domain_name = config['domains'][domain]['gym_domain_name']
    pddl_domain_file = Path(config['domains'][domain]['domain_file'])
    pddl_problem_dir = Path(config['domains'][domain]['problems_dir'])

    object_detection_model_name = config['domains'][domain]['object_detection']['model_name']
    object_detection_temperature = config['domains'][domain]['object_detection']['temperature']
    fluent_classification_model_name = config['domains'][domain]['fluent_classification']['model_name']
    fluent_classification_temperature = config['domains'][domain]['fluent_classification']['temperature']

    if openai_apikey == "your-api-key-here":
        raise ValueError(
            "Please set your OpenAI API key in config.yaml\n"
            "Copy config.example.yaml to config.yaml and add your API key."
        )

    # Create noisy conflict search simulator
    simulator = NoisyConflictSearchSimulator(
        domain_name=domain_name,
        openai_apikey=openai_apikey,
        pddl_domain_file=pddl_domain_file,
        pddl_problem_dir=pddl_problem_dir,
        object_detection_model_name=object_detection_model_name,
        object_detection_temperature=object_detection_temperature,
        fluent_classification_model_name=fluent_classification_model_name,
        fluent_classification_temperature=fluent_classification_temperature,
        experiment_dir_path=Path("noisy_conflict_experiments"),
        fluent_patch_cost=1,
        model_patch_cost=1,
        max_search_nodes=None,  # Limit search for efficiency
        seed=42
    )

    # Configuration for cross-validation experiment
    problems_to_test = [
        "problem1",
        "problem3",
        "problem5",
        "problem7",
        "problem9",
    ]

    # Run cross-validation with conflict-driven search
    print("\n" + "="*80)
    print("RUNNING CROSS-VALIDATION WITH CONFLICT-DRIVEN SEARCH")
    print("="*80 + "\n")

    results_dir = simulator.run_cross_validation_with_conflict_search(
        problems=problems_to_test,
        num_steps=20,
        experiment_name="noisy_conflict_cv"
    )

    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION COMPLETE!")
    print(f"Results directory: {results_dir}")
    print(f"{'='*80}\n")