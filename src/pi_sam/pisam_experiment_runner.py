import shutil
from pathlib import Path
from typing import Tuple, Dict, List

from experiments.basic_experiment_runner import OfflineBasicExperimentRunner
from experiments.experiments_consts import DEFAULT_SPLIT
from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Observation
from sam_learning.core import LearnerDomain
from utilities import LearningAlgorithmType, NegativePreconditionPolicy

from src.pi_sam import PISAMLearner
from src.utils.masking import load_masked_observation


#  TODO: this code should be moved to the sam_learning package when done
class OfflinePiSamExperimentRunner(OfflineBasicExperimentRunner):
    """Class to conduct offline numeric action model learning vip_experiments."""

    def __init__(
        self,
        working_directory_path: Path,
        domain_file_name: str,
        problem_prefix: str = "problem",
        n_split: int = DEFAULT_SPLIT,
        negative_precondition_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.hard
    ):
        super().__init__(
            working_directory_path=working_directory_path,
            domain_file_name=domain_file_name,
            learning_algorithm=LearningAlgorithmType.sam_learning,
            problem_prefix=problem_prefix,
            n_split=n_split,
            negative_precondition_policy=negative_precondition_policy
        )

    def copy_masking_info_to_train_dir(self, train_set_dir_path: Path) -> None:
        """
        Copy masking info files from the working directory to the train set directory.

        During cross-validation, trajectory files are copied to fold-specific train directories,
        but masking info files remain in the parent working directory. This method copies
        the necessary .masking_info files to match the trajectories in the train directory.

        :param train_set_dir_path: Directory containing trajectory files (where masking info should be copied to).
        """
        # Find all trajectory files in the train directory
        trajectory_files = sorted(train_set_dir_path.glob("*.trajectory"))

        self.logger.info(f"Copying masking info files for {len(trajectory_files)} trajectories to {train_set_dir_path}")

        copied_count = 0
        for traj_file in trajectory_files:
            problem_name = traj_file.stem  # e.g., "problem1"

            # Source: masking info in working directory
            source_masking_file = self.working_directory_path / f"{problem_name}.masking_info"

            # Destination: masking info in train directory
            dest_masking_file = train_set_dir_path / f"{problem_name}.masking_info"

            # Copy if source exists and destination doesn't
            if source_masking_file.exists():
                if not dest_masking_file.exists():
                    shutil.copy(source_masking_file, dest_masking_file)
                    copied_count += 1
                    self.logger.debug(f"Copied masking info for {problem_name}")
                else:
                    self.logger.debug(f"Masking info for {problem_name} already exists in train directory")
            else:
                self.logger.warning(f"Masking info file not found in working directory for {problem_name}: {source_masking_file}")

        self.logger.info(f"Copied {copied_count} masking info files to train directory")

    def load_masked_observations(self, train_set_dir_path: Path, domain: Domain) -> List[Observation]:
        """
        Load and prepare masked observations for all trajectory files in the training directory.

        This method:
        1. Finds all .trajectory files in the directory
        2. For each trajectory, loads the corresponding .masking_info file
        3. Uses load_masked_observation() to ground and mask each trajectory
        4. Returns a list of ready-to-use masked observations

        :param train_set_dir_path: Directory containing trajectory and masking info files
        :param domain: The domain to use for parsing
        :return: A list of masked observations ready for PI-SAM learning
        """
        masked_observations = []
        trajectory_files = sorted(train_set_dir_path.glob("*.trajectory"))

        self.logger.info(f"Loading and masking {len(trajectory_files)} trajectories")

        for traj_file in trajectory_files:
            problem_name = traj_file.stem  # e.g., "problem1"
            masking_info_file = train_set_dir_path / f"{problem_name}.masking_info"

            if not masking_info_file.exists():
                self.logger.warning(f"Masking info file not found for {problem_name}, skipping")
                continue

            # Use the new unified loading method
            masked_obs = load_masked_observation(traj_file, masking_info_file, domain)
            masked_observations.append(masked_obs)

        self.logger.info(f"Loaded and masked {len(masked_observations)} observations")
        return masked_observations

    def learn_model_offline(self, fold_num: int, train_set_dir_path: Path, test_set_dir_path: Path) -> None:
        """
        PI-SAM specific learning that handles grounding and masking before learning.

        This method overrides the parent's learn_model_offline to add PI-SAM specific
        preprocessing steps:
        1. Copy masking info files from working directory to train directory
        2. Load, ground, and mask all observations using load_masked_observations()
        3. Learn from masked observations

        :param fold_num: the index of the current fold that is currently running.
        :param train_set_dir_path: the directory containing the trajectories to learn from.
        :param test_set_dir_path: the directory containing the test set problems.
        """
        self.logger.info(f"Starting PI-SAM learning phase for fold {fold_num}")

        # Copy masking info files from working directory to train directory
        # This is necessary because cross-validation splits only copy .trajectory and .pddl files
        self.copy_masking_info_to_train_dir(train_set_dir_path)

        # Parse domain
        partial_domain_path = train_set_dir_path / self.domain_file_name
        partial_domain = DomainParser(domain_path=partial_domain_path, partial_parsing=True).parse_domain()

        # Load, ground, and mask all observations in one unified call
        masked_observations = self.load_masked_observations(train_set_dir_path, partial_domain)

        # Now learn from masked observations
        self.logger.info(f"Learning action model from {len(masked_observations)} masked observations")
        learned_model, learning_report = self._apply_learning_algorithm(
            partial_domain, masked_observations, test_set_dir_path
        )

        # Continue with validation and statistics (same as parent)
        self.learning_statistics_manager.add_to_action_stats(
            masked_observations, learned_model, learning_report, policy=self.negative_precondition_policy
        )
        learned_domain_path = self.validate_learned_domain(
            masked_observations, learned_model, test_set_dir_path, fold_num, float(learning_report["learning_time"])
        )
        self.semantic_performance_calc.calculate_performance(
            learned_domain_path, len(masked_observations), self.negative_precondition_policy
        )
        self.learning_statistics_manager.export_action_learning_statistics(fold_number=fold_num)
        self.semantic_performance_calc.export_semantic_performance(fold_num + 1)
        self.domain_validator.write_statistics(fold_num)

    def _apply_learning_algorithm(
        self, partial_domain: Domain, allowed_observations: List[Observation], test_set_dir_path: Path
    ) -> Tuple[LearnerDomain, Dict[str, str]]:
        """
        Learns the action model using PI-SAM learning algorithm.

        NOTE: This method now receives ALREADY MASKED observations from learn_model_offline.
        No masking is performed here - observations are ready for learning.

        :param partial_domain: the partial domain without the actions' preconditions and effects.
        :param allowed_observations: the allowed observations (already grounded and masked).
        :param test_set_dir_path: the path to the directory containing the test problems.
        :return: the learned action model and the learned action model's learning statistics.
        """
        pisam_learner = PISAMLearner(
            partial_domain,
            negative_preconditions_policy=self.negative_precondition_policy
        )

        # Observations are already grounded and masked - just learn!
        learnt_domain, learning_report = pisam_learner.learn_action_model(allowed_observations)

        return learnt_domain, learning_report
