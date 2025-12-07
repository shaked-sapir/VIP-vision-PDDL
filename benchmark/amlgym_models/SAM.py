import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

from amlgym.algorithms.AlgorithmAdapter import AlgorithmAdapter
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from sam_learning.learners import SAMLearner
from utilities import NegativePreconditionPolicy

from src.pi_sam import PISAMLearner
from src.utils.masking import load_masked_observation


@dataclass
class SAM(AlgorithmAdapter):
    """
    Adapter class for running the SAM algorithm: "Safe Learning of Lifted Action Models",
    B. Juba and H. S. Le, and R. Stern, Proceedings of the 18th International Conference
    on Principles of Knowledge Representation and Reasoning, 2021.
    https://proceedings.kr.org/2021/36/

    Example:
        .. code-block:: python

            from amlgym.algorithms import get_algorithm
            sam = get_algorithm('SAM')
            model = sam.learn('path/to/domain.pddl', ['path/to/trace0', 'path/to/trace1'])
            print(model)

    """

    def learn(self,
              domain_path: str,
              trajectory_paths: List[str],
              use_problems: bool = False) -> str:
        """
        Learns a PDDL action model from:
         (i)    a (possibly empty) input model which is required to specify the predicates and operators signature;
         (ii)   a list of trajectory file paths.

        :parameter domain_path: input PDDL domain file path
        :parameter trajectory_paths: list of trajectory file paths
        :parameter use_problems: boolean flag indicating whether to provide the set of objects
            specified in the problem from which the trajectories have been generated

        :return: a string representing the learned PDDL model
        """

        # Instantiate SAM algorithm
        partial_domain = DomainParser(Path(domain_path), partial_parsing=True).parse_domain()
        sam = SAMLearner(partial_domain=partial_domain, negative_preconditions_policy=NegativePreconditionPolicy.hard)

        # Parse input trajectories

        allowed_observations = []
        if use_problems:
            raise NotImplementedError("use_problems=True is not implemented yet for PISAM.")
        else: # we use our own trajectories and masks, not amlgym's
            for traj_path in trajectory_paths:
                traj_path = Path(traj_path)

                problem_path = traj_path.with_suffix('.pddl')
                problem = ProblemParser(Path(problem_path), partial_domain).parse_problem()
                allowed_observations.append(TrajectoryParser(partial_domain, problem).parse_trajectory(traj_path))

        # Learn action model
        learned_model, learning_report = sam.learn_action_model(allowed_observations)

        return learned_model.to_pddl()
