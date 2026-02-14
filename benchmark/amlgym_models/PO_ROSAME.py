import sys
import time
from pathlib import Path
from typing import List

from amlgym.algorithms.AlgorithmAdapter import AlgorithmAdapter
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser

from benchmark.amlgym_models.po_rosame_runner import PORosame_Runner

# Add project root to path for our masking utilities
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.masking import load_masking_info, mask_observation
from src.utils.pddl import ground_observation_completely


def _record_timing(profiler, category, step_name, elapsed, traj_idx, problem_name):
    """Helper to record timing with consistent metadata."""
    if profiler:
        profiler.add_detailed_timing(category, step_name, elapsed,
            {'trajectory_index': traj_idx, 'problem_name': problem_name})


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
              use_problems: bool = False,
              profiler=None) -> str:
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
        partial_domain = DomainParser(Path(domain_path), partial_parsing=True).parse_domain()
        pi_rosame = PORosame_Runner(domain_path)

        if use_problems:
            allowed_observations = [TrajectoryParser(partial_domain).parse_trajectory(Path(traj_path))
                                    for traj_path in trajectory_paths]
            return pi_rosame.rosame_to_pddl()
        
        for traj_idx, traj_path in enumerate(trajectory_paths):
            traj_path = Path(traj_path)
            problem_path = traj_path.parent / f"{traj_path.parent.stem}.pddl"
            masking_info_path = traj_path.parent / f"{traj_path.parent.stem}.masking_info"
            
            problem = ProblemParser(problem_path, partial_domain).parse_problem()
            pi_rosame.add_problem(problem)

            problem_name = traj_path.stem
            category = "rosame_trajectory_processing"
            
            def _time_step(step_name, func):
                start = time.perf_counter() if profiler else None
                result = func()
                _record_timing(profiler, category, step_name, 
                    time.perf_counter() - start if profiler else 0, traj_idx, problem_name)
                return result

            observation = _time_step("parse_trajectory", 
                lambda: TrajectoryParser(partial_domain).parse_trajectory(traj_path))
            grounded_observation = _time_step("ground_observation_completely",
                lambda: ground_observation_completely(partial_domain, observation))
            masking_info = _time_step("load_masking_info",
                lambda: load_masking_info(masking_info_path, partial_domain))
            masked_observation = _time_step("mask_observation",
                lambda: mask_observation(grounded_observation, masking_info))

            pi_rosame.ground_new_trajectory()
            _time_step("learn_rosame_single", lambda: pi_rosame.learn_rosame(masked_observation))

        return pi_rosame.rosame_to_pddl()
