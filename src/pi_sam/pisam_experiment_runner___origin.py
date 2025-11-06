from pathlib import Path
from typing import Tuple, Dict, List

from experiments.basic_experiment_runner import OfflineBasicExperimentRunner
from experiments.experiments_consts import DEFAULT_SPLIT
from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Observation
from sam_learning.core import LearnerDomain
from utilities import LearningAlgorithmType

from src.pi_sam import PISAMLearner, PredicateMasker
from src.pi_sam.masking.masking_strategies import MaskingType


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

    def learn_model_offline(self, fold_num: int, train_set_dir_path: Path, test_set_dir_path: Path) -> None:
        """Learns the model of the environment by learning from the input trajectories.
        in this setting, we also have to load maskings information per trajectory (observcation)
        as we cannot represesnt unknown fluents in the trajectory files (in current implementation).

        :param fold_num: the index of the current folder that is currently running.
        :param train_set_dir_path: the directory containing the trajectories in which the algorithm is learning from.
        :param test_set_dir_path: the directory containing the test set problems in which the learned model should be
            used to solve.
        """
        self.logger.info(f"Starting the learning phase for the fold - {fold_num}!")
        partial_domain_path = train_set_dir_path / self.domain_file_name
        partial_domain = DomainParser(domain_path=partial_domain_path, partial_parsing=True).parse_domain()

        allowed_observations = self.collect_observations(train_set_dir_path, partial_domain)
        masking_infos = self.collect_masking_info(train_set_dir_path)

        self.logger.info(f"Learning the action model using {len(allowed_observations)} trajectories!")
        learned_model, learning_report = self._apply_learning_algorithm(partial_domain, allowed_observations,
                                                                        test_set_dir_path)
        self.learning_statistics_manager.add_to_action_stats(
            allowed_observations, learned_model, learning_report, policy=self.negative_precondition_policy
        )
        learned_domain_path = self.validate_learned_domain(
            allowed_observations, learned_model, test_set_dir_path, fold_num, float(learning_report["learning_time"])
        )
        self.semantic_performance_calc.calculate_performance(learned_domain_path, len(allowed_observations),
                                                             self.negative_precondition_policy)
        self.learning_statistics_manager.export_action_learning_statistics(fold_number=fold_num)
        self.semantic_performance_calc.export_semantic_performance(fold_num + 1)
        self.domain_validator.write_statistics(fold_num)

    def _apply_learning_algorithm(
        self, partial_domain: Domain, allowed_observations: List[Observation], test_set_dir_path: Path
    ) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Learns the action model using the numeric action model learning algorithms.

        :param partial_domain: the partial domain without the actions' preconditions and effects.
        :param allowed_observations: the allowed observations.
        :param test_set_dir_path: the path to the directory containing the test problems.
        :return: the learned action model and the learned action model's learning compute_statistics.
        """

        # TODO: refactor the masking strategy to not be part of the PISAMLearner class, but rather a separate functionality.
        # TODO (cont): this is due to the experiment runner inability to set the masking strategy per experiment, and parsing the observations directly from .trajectory files.
        pisam_learner = PISAMLearner(partial_domain,
                                     predicate_masker=PredicateMasker(
                                         masking_strategy=MaskingType.PERCENTAGE,
                                         masking_kwargs={"masking_ratio": 0.3}))
        masking_info = pisam_learner.mask_observations_by_strategy(allowed_observations)


        # return (PISAMLearner(partial_domain)
        #         .learn_action_model(allowed_observations, masking_info=masking_info))
        learnt_domain, learning_report = pisam_learner.learn_action_model(allowed_observations, masking_info=masking_info)
        return learnt_domain, learning_report
