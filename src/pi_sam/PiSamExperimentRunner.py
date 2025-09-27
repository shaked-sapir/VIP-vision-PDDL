from pathlib import Path

from experiments.basic_experiment_runner import OfflineBasicExperimentRunner
from experiments.experiments_consts import DEFAULT_SPLIT
from pddl_plus_parser.models import Domain, Observation
from sam_learning.core import LearnerDomain
from sam_learning.learners import SAMLearner
from typing import Tuple, Dict, List
from utilities import LearningAlgorithmType, NegativePreconditionPolicy

from src.pi_sam import PISAMLearner, PredicateMasker
from src.pi_sam.masking.masking_strategies import MaskingStrategy, MaskingType


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

    def _apply_learning_algorithm(
        self, partial_domain: Domain, allowed_observations: List[Observation], test_set_dir_path: Path
    ) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Learns the action model using the numeric action model learning algorithms.

        :param partial_domain: the partial domain without the actions' preconditions and effects.
        :param allowed_observations: the allowed observations.
        :param test_set_dir_path: the path to the directory containing the test problems.
        :return: the learned action model and the learned action model's learning compute_statistics.
        """

        # TODO: refactor the masking strategy to not be part of the PISAMLearner class, but rather a separate functionality
        pisam_learner = PISAMLearner(partial_domain,
                                     predicate_masker=PredicateMasker(
                                         masking_strategy=MaskingType.PERCENTAGE,
                                         masking_kwargs={"masking_ratio": 0.8}))
        masking_info = pisam_learner.mask_observations_by_strategy(allowed_observations)


        # return (PISAMLearner(partial_domain)
        #         .learn_action_model(allowed_observations, masking_info=masking_info))
        learnt_domain, learning_report = pisam_learner.learn_action_model(allowed_observations, masking_info=masking_info)
        return learnt_domain, learning_report
