import shutil
from pathlib import Path
from typing import Tuple, Dict, List, Set

from experiments.basic_experiment_runner import OfflineBasicExperimentRunner
from experiments.experiments_consts import DEFAULT_SPLIT
from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Observation, GroundedPredicate
from sam_learning.core import LearnerDomain
from utilities import LearningAlgorithmType

from src.pi_sam import PISAMLearner
from src.pi_sam.utils import load_masking_info
from src.utils.pddl import ground_observation_completely, mask_observation


#  TODO: this code should be moved to the sam_learning package when done
class OfflinePiSamExperimentRunner(OfflineBasicExperimentRunner):
    """Class to conduct offline numeric action model learning vip_experiments."""

    def __init__(
        self,
        working_directory_path: Path,
        domain_file_name: str,
        problem_prefix: str = "problem",
        n_split: int = DEFAULT_SPLIT
    ):
        super().__init__(
            working_directory_path=working_directory_path,
            domain_file_name=domain_file_name,
            learning_algorithm=LearningAlgorithmType.sam_learning,
            problem_prefix=problem_prefix,
            n_split=n_split
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

    def collect_masking_info(self, train_set_dir_path: Path, domain: Domain) -> List[List[Set[GroundedPredicate]]]:
        """
        Load masking info for all trajectory files in the training directory.

        This method loads .masking_info files that were saved alongside trajectories,
        matching each trajectory file with its corresponding masking info.

        :param train_set_dir_path: Directory containing trajectory and masking info files.
        :param domain: The domain to use for parsing predicates.
        :return: A list of masking info, one per trajectory file (in sorted order).
        """
        masking_infos = []
        trajectory_files = sorted(train_set_dir_path.glob("*.trajectory"))

        self.logger.info(f"Loading masking info for {len(trajectory_files)} trajectories")

        for traj_file in trajectory_files:
            problem_name = traj_file.stem  # e.g., "problem1"
            masking_info_file = train_set_dir_path / f"{problem_name}.masking_info"

            if not masking_info_file.exists():
                self.logger.warning(f"Masking info file not found for {problem_name}, skipping")
                continue

            # Pass the already-computed file path directly to avoid redundant path construction
            masking_info = load_masking_info(masking_info_file, domain)
            masking_infos.append(masking_info)

        self.logger.info(f"Loaded masking info for {len(masking_infos)} trajectories")
        return masking_infos

    def learn_model_offline(self, fold_num: int, train_set_dir_path: Path, test_set_dir_path: Path) -> None:
        """
        PI-SAM specific learning that handles grounding and masking before learning.

        This method overrides the parent's learn_model_offline to add PI-SAM specific
        preprocessing steps:
        1. Copy masking info files from working directory to train directory
        2. Load observations (from parent)
        3. Load masking info from disk
        4. Ground all observations completely
        5. Apply masks to observations
        6. Learn from masked observations

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

        # Collect observations using parent's method
        allowed_observations = self.collect_observations(train_set_dir_path, partial_domain)
        self.logger.info(f"Collected {len(allowed_observations)} observations")

        # Load masking info from disk (now available in train directory)
        masking_infos = self.collect_masking_info(train_set_dir_path, partial_domain)

        if len(masking_infos) != len(allowed_observations):
            raise ValueError(
                f"Mismatch: {len(allowed_observations)} observations but {len(masking_infos)} masking infos"
            )

        # Ground all observations completely using utility function
        self.logger.info("Grounding observations completely...")
        grounded_observations = [
            ground_observation_completely(partial_domain, obs)
            for obs in allowed_observations
        ]

        # Mask observations using loaded masking info and utility function
        self.logger.info("Applying masks to observations...")
        masked_observations = [
            mask_observation(obs, mask_info)
            for obs, mask_info in zip(grounded_observations, masking_infos)
        ]

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
