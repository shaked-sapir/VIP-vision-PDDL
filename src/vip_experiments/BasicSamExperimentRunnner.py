from pathlib import Path

from experiments.basic_experiment_runner import OfflineBasicExperimentRunner
from experiments.experiments_consts import DEFAULT_SPLIT
from pddl_plus_parser.models import Domain, Observation
from sam_learning.core import LearnerDomain
from sam_learning.learners import SAMLearner
from typing import Tuple, Dict, List
from utilities import LearningAlgorithmType, NegativePreconditionPolicy


class OfflineBasicSamExperimentRunner(OfflineBasicExperimentRunner):
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
        # return (SAMLearner(partial_domain, negative_preconditions_policy=NegativePreconditionPolicy.hard)
        #         .learn_action_model(allowed_observations))
        sam_learner = SAMLearner(partial_domain, negative_preconditions_policy=NegativePreconditionPolicy.hard)
        learnt_domain, learning_report = sam_learner.learn_action_model(allowed_observations)

        return learnt_domain, learning_report
