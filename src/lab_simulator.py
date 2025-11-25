"""
Lab Simulator - Compare PI-SAM vs Noisy Conflict Search.

This script runs both standard PI-SAM and conflict-driven search on the same
cross-validation folds and compares their performance.

Each learner produces a complete set of experiment results (per-fold performance,
semantic performance, validation statistics, etc.) in separate directories.

Usage:
    python src/lab_simulator.py
"""

import logging
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Observation

from experiments.experiments_consts import DEFAULT_NUMERIC_TOLERANCE
from sam_learning.core import LearnerDomain
from sam_statistics.learning_statistics_manager import LearningStatisticsManager
from sam_statistics.utils import init_semantic_performance_calculator
from utilities import LearningAlgorithmType, SolverType, NegativePreconditionPolicy
from utilities.k_fold_split import KFoldSplit
from validators import DomainValidator

from src.pi_sam import PISAMLearner
from src.plan_denoising.conflict_search import ConflictDrivenPatchSearch
from src.utils.masking import load_masked_observation


class LabSimulatorRunner:
    """
    Compare PI-SAM vs Noisy Conflict Search on the same cross-validation folds.

    Each learner gets its own complete experiment infrastructure with:
    - Learning statistics manager
    - Semantic performance calculator
    - Domain validator
    - Full result directory structure
    """

    def __init__(
        self,
        working_directory_path: Path,
        domain_file_name: str,
        problem_prefix: str = "problem",
        n_split: int = 5,
        fluent_patch_cost: int = 1,
        model_patch_cost: int = 1,
        max_search_nodes: Optional[int] = None,
        seed: int = 42,
        negative_precondition_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.hard
    ):
        """
        Initialize the lab simulator.

        :param working_directory_path: Path to the experiment directory.
        :param domain_file_name: Name of the domain file.
        :param problem_prefix: Prefix for problem files.
        :param n_split: Number of cross-validation folds.
        :param fluent_patch_cost: Cost for fluent patches in conflict search.
        :param model_patch_cost: Cost for model patches in conflict search.
        :param max_search_nodes: Max nodes for conflict search (None = unlimited).
        :param seed: Random seed.
        :param negative_precondition_policy: Policy for negative preconditions.
        """
        self.logger = logging.getLogger(__name__)
        self.working_directory_path = working_directory_path
        self.domain_file_name = domain_file_name
        self.problem_prefix = problem_prefix
        self.negative_precondition_policy = negative_precondition_policy
        self.n_split = n_split

        # Conflict search parameters
        self.fluent_patch_cost = fluent_patch_cost
        self.model_patch_cost = model_patch_cost
        self.max_search_nodes = max_search_nodes
        self.seed = seed

        self.k_fold = KFoldSplit(
            working_directory_path=working_directory_path,
            domain_file_name=domain_file_name,
            n_split=n_split
        )

        # Create ultimate results directory
        ultimate_results_dir = working_directory_path / "ultimate_results_directory"
        ultimate_results_dir.mkdir(parents=True, exist_ok=True)

        # Create separate result directories for each learner
        self.pisam_results_dir = ultimate_results_dir / "pisam"
        self.noisy_results_dir = ultimate_results_dir / "noisy_conflict_search"

        self.pisam_results_dir.mkdir(exist_ok=True)
        self.noisy_results_dir.mkdir(exist_ok=True)

        # Copy domain file to each results directory (needed by semantic performance calculator)
        domain_source = working_directory_path / domain_file_name
        if domain_source.exists():
            shutil.copy(domain_source, self.pisam_results_dir / domain_file_name)
            shutil.copy(domain_source, self.noisy_results_dir / domain_file_name)

        # Initialize statistics managers for PI-SAM
        self.pisam_stats_manager = LearningStatisticsManager(
            working_directory_path=self.pisam_results_dir,
            domain_path=working_directory_path / domain_file_name,
            learning_algorithm=LearningAlgorithmType.sam_learning
        )

        self.pisam_validator = DomainValidator(
            self.pisam_results_dir,
            LearningAlgorithmType.sam_learning,
            working_directory_path / domain_file_name,
            problem_prefix=problem_prefix
        )

        # Initialize statistics managers for Noisy Conflict Search
        self.noisy_stats_manager = LearningStatisticsManager(
            working_directory_path=self.noisy_results_dir,
            domain_path=working_directory_path / domain_file_name,
            learning_algorithm=LearningAlgorithmType.sam_learning
        )

        self.noisy_validator = DomainValidator(
            self.noisy_results_dir,
            LearningAlgorithmType.sam_learning,
            working_directory_path / domain_file_name,
            problem_prefix=problem_prefix
        )

        # Semantic performance calculators (initialized per fold)
        self.pisam_semantic_calc = None
        self.noisy_semantic_calc = None

        self.logger.info(f"Lab Simulator initialized")
        self.logger.info(f"  Working directory: {working_directory_path}")
        self.logger.info(f"  Learners: PI-SAM, Noisy Conflict Search")
        self.logger.info(f"  Cross-validation folds: {n_split}")
        self.logger.info(f"  PI-SAM results: {self.pisam_results_dir}")
        self.logger.info(f"  Noisy results: {self.noisy_results_dir}")

    def _init_semantic_performance_calculators(self, test_set_dir_path: Path) -> None:
        """Initialize semantic performance calculators for both learners."""
        self.pisam_semantic_calc = init_semantic_performance_calculator(
            self.pisam_results_dir,
            self.domain_file_name,
            learning_algorithm=LearningAlgorithmType.sam_learning,
            problem_prefix=self.problem_prefix,
            test_set_dir_path=test_set_dir_path
        )

        self.noisy_semantic_calc = init_semantic_performance_calculator(
            self.noisy_results_dir,
            self.domain_file_name,
            learning_algorithm=LearningAlgorithmType.sam_learning,
            problem_prefix=self.problem_prefix,
            test_set_dir_path=test_set_dir_path
        )

    def copy_masking_info_to_train_dir(self, train_set_dir_path: Path) -> None:
        """Copy masking info files from working directory to train directory."""
        trajectory_files = sorted(train_set_dir_path.glob("*.trajectory"))

        for traj_file in trajectory_files:
            problem_name = traj_file.stem
            source_masking_file = self.working_directory_path / f"{problem_name}.masking_info"
            dest_masking_file = train_set_dir_path / f"{problem_name}.masking_info"

            if source_masking_file.exists() and not dest_masking_file.exists():
                shutil.copy(source_masking_file, dest_masking_file)

    def load_masked_observations(self, train_set_dir_path: Path, domain: Domain) -> List[Observation]:
        """Load and mask observations from the training directory."""
        masked_observations = []
        trajectory_files = sorted(train_set_dir_path.glob("*.trajectory"))

        for traj_file in trajectory_files:
            problem_name = traj_file.stem
            masking_info_file = train_set_dir_path / f"{problem_name}.masking_info"

            if not masking_info_file.exists():
                self.logger.warning(f"Masking info file not found for {problem_name}, skipping")
                continue

            masked_obs = load_masked_observation(traj_file, masking_info_file, domain)
            masked_observations.append(masked_obs)

        return masked_observations

    def run_pisam_learner(
        self,
        partial_domain: Domain,
        masked_observations: List[Observation]
    ) -> Tuple[LearnerDomain, Dict[str, str]]:
        """
        Run standard PI-SAM learner.

        :param partial_domain: The partial domain template.
        :param masked_observations: The masked observations.
        :return: (learned_domain, learning_report)
        """
        start_time = time.time()

        pisam_learner = PISAMLearner(
            deepcopy(partial_domain),
            negative_preconditions_policy=self.negative_precondition_policy
        )

        learned_domain, report = pisam_learner.learn_action_model(masked_observations)

        learning_time = time.time() - start_time
        report["learning_time"] = str(learning_time)

        return learned_domain, report

    def run_noisy_conflict_search(
        self,
        partial_domain: Domain,
        masked_observations: List[Observation]
    ) -> Tuple[LearnerDomain, Dict[str, str]]:
        """
        Run Conflict-Driven Patch Search.

        :param partial_domain: The partial domain template.
        :param masked_observations: The masked observations.
        :return: (learned_domain, enriched_learning_report)
        """
        start_time = time.time()

        search = ConflictDrivenPatchSearch(
            partial_domain_template=deepcopy(partial_domain),
            negative_preconditions_policy=self.negative_precondition_policy,
            fluent_patch_cost=self.fluent_patch_cost,
            model_patch_cost=self.model_patch_cost,
            seed=self.seed,
            logger=None
        )

        learned_domain, conflicts, model_constraints, fluent_patches, cost, report = search.run(
            observations=masked_observations,
            max_nodes=self.max_search_nodes
        )

        learning_time = time.time() - start_time

        # Enrich the report with patch statistics
        enriched_report = dict(report)
        enriched_report["learning_time"] = str(learning_time)
        enriched_report["solution_found"] = str(len(conflicts) == 0)
        enriched_report["final_conflicts"] = str(len(conflicts))
        enriched_report["solution_cost"] = str(cost)
        enriched_report["total_model_constraints"] = str(len(model_constraints))
        enriched_report["total_fluent_patches"] = str(len(fluent_patches))

        # Extract patch diff details
        patch_diff = report.get("patch_diff", {})

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

        # Add details as semicolon-separated strings
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

        return learned_domain, enriched_report

    def export_learned_domain(
        self,
        learned_domain: LearnerDomain,
        test_set_dir_path: Path,
        results_dir: Path,
        fold_number: int
    ) -> Path:
        """Export learned domain to test directory and backup."""
        # Export to test directory (for validation)
        domain_path = test_set_dir_path / self.domain_file_name
        with open(domain_path, "wt") as domain_file:
            domain_file.write(learned_domain.to_pddl())

        # Also backup to results directory
        domains_backup_dir = results_dir / "domains_backup"
        domains_backup_dir.mkdir(exist_ok=True)

        backup_filename = f"sam_learning_fold_{fold_number}_{learned_domain.name}.pddl"
        backup_path = domains_backup_dir / backup_filename
        with open(backup_path, "wt") as domain_file:
            domain_file.write(learned_domain.to_pddl())

        return domain_path

    def validate_learned_domain(
        self,
        domain_path: Path,
        test_set_dir_path: Path,
        allowed_observations: List[Observation],
        learning_time: float,
        validator: DomainValidator
    ) -> None:
        """Validate the learned domain using the validator."""
        validator.validate_domain(
            tested_domain_file_path=domain_path,
            test_set_directory_path=test_set_dir_path,
            used_observations=allowed_observations,
            tolerance=DEFAULT_NUMERIC_TOLERANCE,
            timeout=60,
            learning_time=learning_time,
            solvers_portfolio=[SolverType.fast_downward]
        )

    def learn_model_offline(
        self,
        fold_num: int,
        train_set_dir_path: Path,
        test_set_dir_path: Path
    ) -> None:
        """
        Run both learners on a single fold with full experiment infrastructure.

        :param fold_num: The fold number.
        :param train_set_dir_path: Path to training data.
        :param test_set_dir_path: Path to test data.
        """
        self.logger.info(f"="*80)
        self.logger.info(f"Processing Fold {fold_num}")
        self.logger.info(f"="*80)

        # Initialize semantic performance calculators for this fold
        self._init_semantic_performance_calculators(test_set_dir_path)

        # Copy masking info to train directory
        self.copy_masking_info_to_train_dir(train_set_dir_path)

        # Parse domain
        partial_domain_path = train_set_dir_path / self.domain_file_name
        partial_domain = DomainParser(domain_path=partial_domain_path, partial_parsing=True).parse_domain()

        # Load masked observations (once, shared by both learners)
        masked_observations = self.load_masked_observations(train_set_dir_path, partial_domain)
        self.logger.info(f"Loaded {len(masked_observations)} masked observations")

        # ========== Run PI-SAM ==========
        self.logger.info(f"\nRunning PI-SAM learner...")
        pisam_domain, pisam_report = self.run_pisam_learner(partial_domain, masked_observations)

        # Add to statistics
        self.pisam_stats_manager.add_to_action_stats(
            masked_observations,
            pisam_domain,
            pisam_report,
            policy=self.negative_precondition_policy
        )

        # Export domain
        pisam_domain_path = self.export_learned_domain(
            pisam_domain,
            test_set_dir_path,
            self.pisam_results_dir,
            fold_num
        )

        # Validate domain
        self.validate_learned_domain(
            pisam_domain_path,
            test_set_dir_path,
            masked_observations,
            float(pisam_report["learning_time"]),
            self.pisam_validator
        )

        # Calculate semantic performance
        self.pisam_semantic_calc.calculate_performance(
            pisam_domain_path,
            len(masked_observations),
            self.negative_precondition_policy
        )

        # Export per-fold statistics
        self.pisam_stats_manager.export_action_learning_statistics(fold_number=fold_num)
        self.pisam_semantic_calc.export_semantic_performance(fold_num + 1)
        self.pisam_validator.write_statistics(fold_num)

        self.logger.info(f"✓ PI-SAM completed in {pisam_report['learning_time']}s")

        # ========== Run Noisy Conflict Search ==========
        self.logger.info(f"\nRunning Noisy Conflict Search learner...")
        noisy_domain, noisy_report = self.run_noisy_conflict_search(partial_domain, masked_observations)

        # Add to statistics
        self.noisy_stats_manager.add_to_action_stats(
            masked_observations,
            noisy_domain,
            noisy_report,
            policy=self.negative_precondition_policy
        )

        # Export domain
        noisy_domain_path = self.export_learned_domain(
            noisy_domain,
            test_set_dir_path,
            self.noisy_results_dir,
            fold_num
        )

        # Validate domain
        self.validate_learned_domain(
            noisy_domain_path,
            test_set_dir_path,
            masked_observations,
            float(noisy_report["learning_time"]),
            self.noisy_validator
        )

        # Calculate semantic performance
        self.noisy_semantic_calc.calculate_performance(
            noisy_domain_path,
            len(masked_observations),
            self.negative_precondition_policy
        )

        # Export per-fold statistics
        self.noisy_stats_manager.export_action_learning_statistics(fold_number=fold_num)
        self.noisy_semantic_calc.export_semantic_performance(fold_num + 1)
        self.noisy_validator.write_statistics(fold_num)

        self.logger.info(f"✓ Noisy Conflict Search completed in {noisy_report['learning_time']}s")
        self.logger.info(f"  Solution found: {noisy_report['solution_found']}")
        self.logger.info(f"  Final conflicts: {noisy_report['final_conflicts']}")
        self.logger.info(f"  Solution cost: {noisy_report['solution_cost']}")

        # Clear statistics for next fold
        self.pisam_validator.clear_statistics()
        self.pisam_stats_manager.clear_statistics()
        self.noisy_validator.clear_statistics()
        self.noisy_stats_manager.clear_statistics()

    def run_cross_validation(self) -> Tuple[Path, Path]:
        """
        Run cross-validation with both learners.

        :return: Tuple of (pisam_results_dir, noisy_results_dir)
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("STARTING MULTI-LEARNER CROSS-VALIDATION")
        self.logger.info("="*80)
        self.logger.info(f"Learners: PI-SAM vs Noisy Conflict Search")
        self.logger.info(f"Working directory: {self.working_directory_path}")
        self.logger.info("")

        # Create results directories
        self.pisam_stats_manager.create_results_directory()
        self.noisy_stats_manager.create_results_directory()

        # Run cross-validation
        for fold_num, (train_dir_path, test_dir_path) in enumerate(self.k_fold.create_k_fold()):
            self.learn_model_offline(fold_num, train_dir_path, test_dir_path)

        # Write complete joint statistics for PI-SAM
        self.logger.info("\n" + "="*80)
        self.logger.info("Finalizing PI-SAM Results")
        self.logger.info("="*80)
        self.pisam_validator.write_complete_joint_statistics()
        self.pisam_semantic_calc.export_combined_semantic_performance()
        self.pisam_stats_manager.write_complete_joint_statistics()

        # Write complete joint statistics for Noisy Conflict Search
        self.logger.info("\n" + "="*80)
        self.logger.info("Finalizing Noisy Conflict Search Results")
        self.logger.info("="*80)
        self.noisy_validator.write_complete_joint_statistics()
        self.noisy_semantic_calc.export_combined_semantic_performance()
        self.noisy_stats_manager.write_complete_joint_statistics()

        self.logger.info("\n" + "="*80)
        self.logger.info("CROSS-VALIDATION COMPLETE!")
        self.logger.info("="*80)
        self.logger.info(f"PI-SAM results: {self.pisam_results_dir}")
        self.logger.info(f"Noisy Conflict Search results: {self.noisy_results_dir}")

        return self.pisam_results_dir, self.noisy_results_dir


def main():
    """Main function to run the lab simulator."""
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Configuration
    working_dir = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/noisy_conflict_experiments/noisy_conflict_cv__steps=20__model=gpt-5-mini__temp=1.0__22-11-2025T21:08:22")
    domain_file = "blocks.pddl"

    print("\n" + "="*80)
    print("LAB SIMULATOR - PI-SAM vs Noisy Conflict Search Comparison")
    print("="*80)
    print(f"Working directory: {working_dir}")
    print(f"Domain file: {domain_file}")
    print("="*80 + "\n")

    # Create and run lab simulator
    lab = LabSimulatorRunner(
        working_directory_path=working_dir,
        domain_file_name=domain_file,
        problem_prefix="problem",
        n_split=5,
        fluent_patch_cost=1,
        model_patch_cost=1,
        max_search_nodes=None,  # Unlimited search
        seed=42
    )

    pisam_dir, noisy_dir = lab.run_cross_validation()

    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print(f"PI-SAM results:\n  {pisam_dir}")
    print(f"Noisy Conflict Search results:\n  {noisy_dir}")
    print("="*80 + "\n")

    # Display combined semantic performance files
    pisam_combined = pisam_dir / "sam_learning_blocks_combined_semantic_performance.csv"
    noisy_combined = noisy_dir / "sam_learning_blocks_combined_semantic_performance.csv"

    if pisam_combined.exists():
        print("\nPI-SAM Combined Semantic Performance:")
        import pandas as pd
        print(pd.read_csv(pisam_combined).to_string())

    if noisy_combined.exists():
        print("\nNoisy Conflict Search Combined Semantic Performance:")
        import pandas as pd
        print(pd.read_csv(noisy_combined).to_string())


if __name__ == "__main__":
    main()
