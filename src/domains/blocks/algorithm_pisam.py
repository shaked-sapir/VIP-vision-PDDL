import json
import os
import shutil
from pathlib import Path

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import (
    Observation,
    Domain, Problem
)
from sam_learning.learners.sam_learning import SAMLearner
from utilities import NegativePreconditionPolicy

from src.action_model.gym2SAM_parser import create_observation_from_trajectory
from src.experiments.BasicSamExperimentRunnner import OfflineBasicSamExperimentRunner
from src.pi_sam import PISAMLearner, PredicateMasker, MaskingType
from src.pi_sam.PiSamExperimentRunner import OfflinePiSamExperimentRunner
from src.trajectory_handlers.blocks_image_trajectory_handler import BlocksImageTrajectoryHandler

BLOCKS_DOMAIN_FILE_PATH = Path("blocks.pddl")
BLOCKS_PROBLEM_DIR_PATH = Path("problems")
EXPERIMENTS_DIRECTORY_PATH = Path("experiments")
PDDL_FILES_SUFFIX = ".pddl"


# TODO IMM: this should be refactored into an Experiments package, and not be contained inside "domains" package
if __name__ == "__main__":
    """
    This is after refactoring to the BlocksImageTrajectoryHandler class.
    The loop is for generating the data to the cross validation evaluation.
    #  TODO: separate the simulations from the evaluation.
    """
    for num_steps in [10, 25, 50, 100]:
        blocks_domain_name = "PDDLEnvBlocks-v0"
        for blocks_problem_name in [
            "problem1",
            "problem3",
            "problem5",
            "problem7",
            "problem9"
        ]:
            PISAM_DIR = EXPERIMENTS_DIRECTORY_PATH / "pisam"
            PISAM_DIR.mkdir(exist_ok=True)
            WORKING_DIRECTORY_PATH = PISAM_DIR / f"pisam__num_steps={num_steps}"
            WORKING_DIRECTORY_PATH.mkdir(exist_ok=True)
            BLOCKS_OUTPUT_DIR_PATH_TEMP = WORKING_DIRECTORY_PATH / f"{blocks_domain_name}_{blocks_problem_name}_temp"
            BLOCKS_OUTPUT_DIR_PATH_TEMP.mkdir(exist_ok=True)
            #
            print("alg started")
            image_trajectory_handler = BlocksImageTrajectoryHandler(blocks_domain_name)

            # TODO: rename the method to a more indicative name/make it return something, right now it looks a bit odd
            # TODO: consider renaming the problems to contain the .pddl suffix
            image_trajectory_handler.image_trajectory_pipeline(problem_name=blocks_problem_name, output_path=BLOCKS_OUTPUT_DIR_PATH_TEMP, num_steps=num_steps)
            pddl_plus_blocks_domain: Domain = DomainParser(BLOCKS_DOMAIN_FILE_PATH).parse_domain()
            pddl_plus_blocks_problem: Problem = ProblemParser(Path(f"{BLOCKS_PROBLEM_DIR_PATH}/{blocks_problem_name}{PDDL_FILES_SUFFIX}"),
                                                              pddl_plus_blocks_domain).parse_problem()

            # check that we could load the trajectory from the resulted path if needed
            with open(f"{BLOCKS_OUTPUT_DIR_PATH_TEMP}/{blocks_problem_name}_trajectory.json", 'r') as file:
                GT_trajectory = json.load(file)
            GT_observation: Observation = create_observation_from_trajectory(GT_trajectory, pddl_plus_blocks_domain,
                                                                             pddl_plus_blocks_problem)

            # Output the resulting Observation object
            print("printing GT observation:")
            for component in GT_observation.components:
                print(str(component))

            print("*****************************")

            trajectory_parser = TrajectoryParser(pddl_plus_blocks_domain, pddl_plus_blocks_problem)
            imaged_observation = trajectory_parser.parse_trajectory(BLOCKS_OUTPUT_DIR_PATH_TEMP / f'{blocks_problem_name}.trajectory')

            # Output the resulting Observation object
            print("printing imaged observation:")
            for component in imaged_observation.components:
                print(str(component))

            print("*****************************")
            # initial_state_predicates = set.union(*(imaged_observation.components[0].previous_state.state_predicates.values()))
            # pisam_learner = PISAMLearner(
            #     partial_domain=pddl_plus_blocks_domain
            # )
            # masking_info = pisam_learner.mask_observations_by_strategy([imaged_observation])
            # partial_domain, report = pisam_learner.learn_action_model([imaged_observation], masking_info=masking_info)
            # print(partial_domain.to_pddl())
            # print(report)

            """ copy problem file and trajectory file into the experiments dir to automate the process"""
            shutil.copy(f'./problems/{blocks_problem_name}.pddl', WORKING_DIRECTORY_PATH / f'{blocks_problem_name}.pddl')
            shutil.copy(BLOCKS_OUTPUT_DIR_PATH_TEMP / f'{blocks_problem_name}.trajectory', WORKING_DIRECTORY_PATH / f'{blocks_problem_name}.trajectory')

        """ copy the GT domain file into the experiments dir to automate the process"""
        shutil.copy(BLOCKS_DOMAIN_FILE_PATH, WORKING_DIRECTORY_PATH / BLOCKS_DOMAIN_FILE_PATH)

        experiment_runner = OfflinePiSamExperimentRunner(
            working_directory_path=WORKING_DIRECTORY_PATH,
            domain_file_name=BLOCKS_DOMAIN_FILE_PATH.name,
            problem_prefix="problem"
        )

        experiment_runner.run_cross_validation()
