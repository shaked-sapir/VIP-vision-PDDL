import sys
import os

from amlgym.algorithms.AlgorithmAdapter import AlgorithmAdapter
from typing import List
from pathlib import Path
from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser
from pddl_plus_parser.lisp_parsers import ProblemParser

from amlgym.algorithms.rosame.experiment_runner.rosame_runner import Rosame_Runner

from benchmark.PO_ROSAME.po_rosame_runner import PORosame_Runner

# Add project root to path for our masking utilities
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.pddl import ground_observation_completely
from src.utils.masking import load_masking_info, mask_observation


class PO_ROSAME(AlgorithmAdapter):
    """
    Adapter class for running an *unofficial* implementation of the ROSAME
    algorithm: "Neuro-Symbolic Learning of Lifted Action Models from
    Visual Traces", Kai Xi, Stephen Gould, Sylvie Thiebaux, ICAPS 2024.
    https://ojs.aaai.org/index.php/ICAPS/article/download/31528/33688

    Example:
        .. code-block:: python

            from amlgym.algorithms import get_algorithm
            rosame = get_algorithm('ROSAME')
            model = rosame.learn('path/to/domain.pddl', ['path/to/trace0', 'path/to/trace1'])
            print(model)

    """

    def __init__(self, **kwargs):
        super(PO_ROSAME, self).__init__(**kwargs)

    @staticmethod
    def learn(domain_path: str,
              trajectory_paths: List[str],
              use_problems: bool = True) -> str:
        """
        Learns a PDDL action model from:
         (i)    a (possibly empty) input model which is required to specify the predicates and operators signature;
         (ii)   a list of trajectory file paths.

        :parameter domain_path: input PDDL domain file path
        :parameter trajectory_paths: list of trajectory file paths, which also include the masking information
        for each trajectory to learn a partially observable action model.
        :parameter use_problems: boolean flag indicating whether to provide the set of objects
            specified in the problem from which the trajectories have been generated.
            When True, for each trajectory path, expects:
            - Problem file: same directory, same name with .pddl suffix
            - Masking file: same directory, same name with .masking_info suffix

        :return: a string representing the learned PDDL model
        """

        # Instantiate PO_ROSAME algorithm
        partial_domain = DomainParser(Path(domain_path), partial_parsing=True).parse_domain()
        rosame = PORosame_Runner(domain_path)

        # Parse input trajectories
        if not use_problems:
            # Direct parsing without problems (trajectories already have all objects)
            allowed_observations = [TrajectoryParser(partial_domain).parse_trajectory(Path(traj_path))
                                    for traj_path in trajectory_paths]
        else:
            # Parse with problems and apply masking
            for traj_path in trajectory_paths:
                traj_path = Path(traj_path)

                # Derive problem path: same directory, same name but .pddl suffix
                # e.g., trace_0/problem1.trajectory → trace_0/problem1.pddl
                problem_path = traj_path.with_suffix('.pddl')

                # Derive masking_info path: same directory, same name but .masking_info suffix
                # e.g., trace_0/problem1.trajectory → trace_0/problem1.masking_info
                masking_info_path = traj_path.parent / f"{traj_path.stem}.masking_info"

                # Parse problem
                problem = ProblemParser(problem_path, partial_domain).parse_problem()
                rosame.add_problem(problem)

                # Parse trajectory
                observation = TrajectoryParser(partial_domain).parse_trajectory(traj_path)

                # Ground observation completely
                grounded_observation = ground_observation_completely(partial_domain, observation)

                # Load and apply masking info
                masking_info = load_masking_info(masking_info_path, partial_domain)
                masked_observation = mask_observation(grounded_observation, masking_info)

                # Learn from masked observation
                rosame.ground_new_trajectory()
                rosame.learn_rosame(masked_observation)

        return rosame.rosame_to_pddl()
