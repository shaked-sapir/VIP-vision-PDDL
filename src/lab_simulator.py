"""
Lab Simulator - Multi-learner comparison framework.

This module provides infrastructure for running multiple learning algorithms
on the same cross-validation folds and comparing their performance.
"""

import logging
import shutil
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Type

import pandas as pd
from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Observation

from experiments.basic_experiment_runner import OfflineBasicExperimentRunner
from experiments.experiments_consts import DEFAULT_SPLIT
from sam_learning.core import LearnerDomain
from sam_statistics.utils import init_semantic_performance_calculator
from utilities import LearningAlgorithmType, NegativePreconditionPolicy
from utilities.k_fold_split import KFoldSplit
from validators import DomainValidator

from src.pi_sam import PISAMLearner
from src.plan_denoising.conflict_search import ConflictDrivenPatchSearch
from src.utils.masking import load_masked_observation


class BaseLearnerConfig(ABC):
    """Abstract base class for learner configurations."""

    def __init__(self, name: str):
        """
        Initialize learner config.

        :param name: Unique name for this learner (used in result directories).
        """
        self.name = name

    @abstractmethod
    def create_and_learn(
        self,
        partial_domain: Domain,
        observations: List[Observation],
        negative_precondition_policy: NegativePreconditionPolicy,
        logger: Optional[logging.Logger] = None
    ) -> Tuple[LearnerDomain, Dict[str, str]]:
        """
        Create the learner and learn from observations.

        :param partial_domain: The partial domain template.
        :param observations: The masked observations to learn from.
        :param negative_precondition_policy: Policy for handling negative preconditions.
        :param logger: Optional logger.
        :return: Tuple of (learned_domain, learning_report).
        """
        raise NotImplementedError


class PISAMLearnerConfig(BaseLearnerConfig):
    """Configuration for standard PI-SAM learner."""

    def __init__(self, name: str = "pisam"):
        super().__init__(name)

    def create_and_learn(
        self,
        partial_domain: Domain,
        observations: List[Observation],
        negative_precondition_policy: NegativePreconditionPolicy,
        logger: Optional[logging.Logger] = None
    ) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Run PI-SAM learning."""
        pisam_learner = PISAMLearner(
            deepcopy(partial_domain),
            negative_preconditions_policy=negative_precondition_policy
        )
        return pisam_learner.learn_action_model(observations)


class NoisyConflictSearchLearnerConfig(BaseLearnerConfig):
    """Configuration for Conflict-Driven Patch Search learner."""

    def __init__(
        self,
        name: str = "noisy_conflict_search",
        fluent_patch_cost: int = 1,
        model_patch_cost: int = 1,
        max_search_nodes: Optional[int] = None,
        seed: int = 42
    ):
        super().__init__(name)
        self.fluent_patch_cost = fluent_patch_cost
        self.model_patch_cost = model_patch_cost
        self.max_search_nodes = max_search_nodes
        self.seed = seed

    def create_and_learn(
        self,
        partial_domain: Domain,
        observations: List[Observation],
        negative_precondition_policy: NegativePreconditionPolicy,
        logger: Optional[logging.Logger] = None
    ) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Run Conflict-Driven Patch Search."""
        search = ConflictDrivenPatchSearch(
            partial_domain_template=deepcopy(partial_domain),
            negative_preconditions_policy=negative_precondition_policy,
            fluent_patch_cost=self.fluent_patch_cost,
            model_patch_cost=self.model_patch_cost,
            seed=self.seed,
            logger=logger
        )

        learned_domain, conflicts, model_constraints, fluent_patches, cost, report = search.run(
            observations=observations,
            max_nodes=self.max_search_nodes
        )

        # Enrich the report with patch statistics
        enriched_report = dict(report)
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

        # Flatten model patches for counting
        model_added_count = sum(len(v) for v in model_added.values())
        model_removed_count = sum(len(v) for v in model_removed.values())
        model_changed_count = sum(len(v) for v in model_changed.values())

        enriched_report["model_patches_added"] = str(model_added_count)
        enriched_report["model_patches_removed"] = str(model_removed_count)
        enriched_report["model_patches_changed"] = str(model_changed_count)
        enriched_report["fluent_patches_added"] = str(len(fluent_added))
        enriched_report["fluent_patches_removed"] = str(len(fluent_removed))

        # Add details as semicolon-separated strings
        enriched_report["model_patches_added_detail"] = "; ".join(
            f"{action}: {sorted(preds)}" for action, preds in model_added.items() if preds
        )
        enriched_report["model_patches_removed_detail"] = "; ".join(
            f"{action}: {sorted(preds)}" for action, preds in model_removed.items() if preds
        )
        enriched_report["fluent_patches_added_detail"] = "; ".join(str(p) for p in sorted(fluent_added))
        enriched_report["fluent_patches_removed_detail"] = "; ".join(str(p) for p in sorted(fluent_removed))

        return learned_domain, enriched_report


class LabSimulatorRunner:
    """
    Multi-learner experiment runner.

    Runs multiple learning algorithms on the same cross-validation folds
    and combines results for comparison.
    """

    logger: logging.Logger
    working_directory_path: Path
    k_fold: KFoldSplit
    domain_file_name: str
    learner_configs: List[BaseLearnerConfig]
    negative_precondition_policy: NegativePreconditionPolicy

    def __init__(
        self,
        working_directory_path: Path,
        domain_file_name: str,
        learner_configs: List[BaseLearnerConfig],
        problem_prefix: str = "problem",
        n_split: int = DEFAULT_SPLIT,
        negative_precondition_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.no_remove
    ):
        """
        Initialize the lab simulator.

        :param working_directory_path: Path to the experiment directory.
        :param domain_file_name: Name of the domain file.
        :param learner_configs: List of learner configurations to run.
        :param problem_prefix: Prefix for problem files.
        :param n_split: Number of cross-validation folds.
        :param negative_precondition_policy: Policy for negative preconditions.
        """
        self.logger = logging.getLogger(__name__)
        self.working_directory_path = working_directory_path
        self.domain_file_name = domain_file_name
        self.problem_prefix = problem_prefix
        self.learner_configs = learner_configs
        self.negative_precondition_policy = negative_precondition_policy

        self.k_fold = KFoldSplit(
            working_directory_path=working_directory_path,
            domain_file_name=domain_file_name,
            n_split=n_split
        )

        # Create learner-specific result directories
        self.learner_result_dirs: Dict[str, Path] = {}
        for config in learner_configs:
            learner_dir = working_directory_path / "results_directory" / config.name
            learner_dir.mkdir(parents=True, exist_ok=True)
            self.learner_result_dirs[config.name] = learner_dir

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

    def export_learned_domain(
        self, learned_domain: LearnerDomain, output_dir: Path, file_name: str
    ) -> Path:
        """Export learned domain to a file."""
        domain_path = output_dir / file_name
        with open(domain_path, "wt") as domain_file:
            domain_file.write(learned_domain.to_pddl())
        return domain_path

    def run_learner_on_fold(
        self,
        learner_config: BaseLearnerConfig,
        fold_num: int,
        partial_domain: Domain,
        masked_observations: List[Observation],
        test_set_dir_path: Path
    ) -> Dict[str, str]:
        """
        Run a single learner on a single fold.

        :param learner_config: The learner configuration to use.
        :param fold_num: The fold number.
        :param partial_domain: The partial domain template.
        :param masked_observations: The masked observations to learn from.
        :param test_set_dir_path: Path to the test set directory.
        :return: Learning report dictionary.
        """
        self.logger.info(f"Running learner '{learner_config.name}' on fold {fold_num}")

        start_time = time.time()
        learned_domain, learning_report = learner_config.create_and_learn(
            partial_domain=partial_domain,
            observations=masked_observations,
            negative_precondition_policy=self.negative_precondition_policy,
            logger=self.logger
        )
        learning_time = time.time() - start_time

        # Update learning time in report
        learning_report["learning_time"] = str(learning_time)

        # Export learned domain to learner-specific directory
        learner_result_dir = self.learner_result_dirs[learner_config.name]
        domains_backup_dir = learner_result_dir / "domains_backup"
        domains_backup_dir.mkdir(exist_ok=True)

        domain_filename = f"fold_{fold_num}_{learned_domain.name}_{len(masked_observations)}_trajectories.pddl"
        self.export_learned_domain(learned_domain, domains_backup_dir, domain_filename)

        # Also export to test directory for validation
        self.export_learned_domain(learned_domain, test_set_dir_path, f"{learner_config.name}_{self.domain_file_name}")

        return learning_report

    def learn_model_offline(
        self, fold_num: int, train_set_dir_path: Path, test_set_dir_path: Path
    ) -> Dict[str, Dict[str, str]]:
        """
        Run all learners on a single fold.

        :param fold_num: The fold number.
        :param train_set_dir_path: Path to training data.
        :param test_set_dir_path: Path to test data.
        :return: Dictionary mapping learner names to their learning reports.
        """
        self.logger.info(f"Starting multi-learner learning for fold {fold_num}")

        # Copy masking info to train directory
        self.copy_masking_info_to_train_dir(train_set_dir_path)

        # Parse domain
        partial_domain_path = train_set_dir_path / self.domain_file_name
        partial_domain = DomainParser(domain_path=partial_domain_path, partial_parsing=True).parse_domain()

        # Load masked observations (once, shared by all learners)
        masked_observations = self.load_masked_observations(train_set_dir_path, partial_domain)
        self.logger.info(f"Loaded {len(masked_observations)} masked observations for fold {fold_num}")

        # Run each learner
        all_reports: Dict[str, Dict[str, str]] = {}
        for learner_config in self.learner_configs:
            report = self.run_learner_on_fold(
                learner_config=learner_config,
                fold_num=fold_num,
                partial_domain=partial_domain,
                masked_observations=masked_observations,
                test_set_dir_path=test_set_dir_path
            )
            all_reports[learner_config.name] = report

        return all_reports

    def save_fold_reports(
        self, fold_num: int, all_reports: Dict[str, Dict[str, str]]
    ) -> None:
        """Save learning reports for each learner to their result directories."""
        for learner_name, report in all_reports.items():
            learner_dir = self.learner_result_dirs[learner_name]
            report_file = learner_dir / f"fold_{fold_num}_report.csv"

            # Add fold number to report
            report_with_fold = {"fold": str(fold_num), **report}

            # Convert to DataFrame and save
            df = pd.DataFrame([report_with_fold])
            df.to_csv(report_file, index=False)
            self.logger.info(f"Saved report for {learner_name} fold {fold_num} to {report_file}")

    def combine_learner_results(self) -> Path:
        """
        Combine all learners' results into a single comparison CSV.

        :return: Path to the combined results file.
        """
        self.logger.info("Combining results from all learners")

        all_results = []

        for learner_config in self.learner_configs:
            learner_dir = self.learner_result_dirs[learner_config.name]

            # Find all fold reports for this learner
            fold_reports = sorted(learner_dir.glob("fold_*_report.csv"))

            for report_file in fold_reports:
                df = pd.read_csv(report_file)
                df["learner"] = learner_config.name
                all_results.append(df)

        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)

            # Reorder columns to have learner first
            cols = ["learner", "fold"] + [c for c in combined_df.columns if c not in ["learner", "fold"]]
            combined_df = combined_df[cols]

            # Save combined results
            results_dir = self.working_directory_path / "results_directory"
            combined_file = results_dir / "combined_learner_comparison.csv"
            combined_df.to_csv(combined_file, index=False)
            self.logger.info(f"Combined results saved to {combined_file}")

            return combined_file
        else:
            self.logger.warning("No results to combine")
            return None

    def run_cross_validation(self) -> Path:
        """
        Run cross-validation with all learners.

        :return: Path to the combined results file.
        """
        self.logger.info(f"Starting multi-learner cross-validation with {len(self.learner_configs)} learners")

        # Create results directory
        results_dir = self.working_directory_path / "results_directory"
        results_dir.mkdir(exist_ok=True)

        for fold_num, (train_dir_path, test_dir_path) in enumerate(self.k_fold.create_k_fold()):
            self.logger.info(f"Processing fold {fold_num + 1}")

            # Run all learners on this fold
            all_reports = self.learn_model_offline(fold_num, train_dir_path, test_dir_path)

            # Save reports
            self.save_fold_reports(fold_num, all_reports)

            self.logger.info(f"Completed fold {fold_num + 1}")

        # Combine all results
        combined_file = self.combine_learner_results()

        self.logger.info("Cross-validation completed")
        return combined_file


def example_usage():
    """Example usage of LabSimulatorRunner."""
    import logging
    logging.basicConfig(level=logging.INFO)

    # Define learner configurations
    learners = [
        PISAMLearnerConfig(name="pisam"),
        NoisyConflictSearchLearnerConfig(
            name="noisy_conflict",
            fluent_patch_cost=1,
            model_patch_cost=1,
            max_search_nodes=100,
            seed=42
        ),
    ]

    # Create and run lab simulator
    lab = LabSimulatorRunner(
        working_directory_path=Path("path/to/experiment"),
        domain_file_name="blocks.pddl",
        learner_configs=learners,
        n_split=5
    )

    results_file = lab.run_cross_validation()
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    example_usage()
