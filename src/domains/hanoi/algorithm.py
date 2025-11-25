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
from typing import List
from utilities import NegativePreconditionPolicy

from src.action_model.gym2SAM_parser import create_observation_from_trajectory
from src.trajectory_handlers.hanoi_image_trajectory_handler import HanoiImageTrajectoryHandler
from src.vip_experiments.BasicSamExperimentRunnner import OfflineBasicSamExperimentRunner

from src.trajectory_handlers.blocks_image_trajectory_handler import BlocksImageTrajectoryHandler

HANOI_DOMAIN_FILE_PATH = Path("hanoi.pddl")
HANOI_PROBLEM_DIR_PATH = Path("problems")
EXPERIMENTS_DIRECTORY_PATH = Path("experiments")
PDDL_FILES_SUFFIX = ".pddl"


# TODO IMM: this should be refactored into an Experiments package, and not be contained inside "domains" package
if __name__ == "__main__":
    """
    This is after refactoring to the BlocksImageTrajectoryHandler class.
    The loop is for generating the data to the cross validation evaluation.
    #  TODO: separate the simulations from the evaluation.
    """
    for num_steps in [
        10,
        # 25,
        # 50,
        # 100
    ]:
        hanoi_domain_name = "PDDLEnvHanoi_operator_actions-v0"
        for hanoi_problem_name in [
            "problem0",
            # "problem1",
            # "problem2",
            # "problem3",
            # "problem4"
            # "problem5"
        ]:
            WORKING_DIRECTORY_PATH = EXPERIMENTS_DIRECTORY_PATH / f"num_steps={num_steps}"
            WORKING_DIRECTORY_PATH.mkdir(exist_ok=True)
            HANOI_OUTPUT_DIR_PATH_TEMP = WORKING_DIRECTORY_PATH / f"{hanoi_domain_name}_{hanoi_problem_name}_temp"
            HANOI_OUTPUT_DIR_PATH_TEMP.mkdir(exist_ok=True)
            #
            print("alg started")
            image_trajectory_handler = HanoiImageTrajectoryHandler(hanoi_domain_name)

            # TODO: rename the method to a more indicative name/make it return something, right now it looks a bit odd
            # TODO: consider renaming the problems to contain the .pddl suffix
            ground_actions: List[str] = image_trajectory_handler.create_trajectory_from_gym(
                problem_name=hanoi_problem_name, images_output_path=HANOI_OUTPUT_DIR_PATH_TEMP, num_steps=num_steps)
            image_trajectory_handler.image_trajectory_pipeline(problem_name=hanoi_problem_name, actions=ground_actions, images_path=HANOI_OUTPUT_DIR_PATH_TEMP)
            pddl_plus_hanoi_domain: Domain = DomainParser(HANOI_DOMAIN_FILE_PATH).parse_domain()
            pddl_plus_hanoi_problem: Problem = ProblemParser(Path(f"{HANOI_PROBLEM_DIR_PATH}/{hanoi_problem_name}{PDDL_FILES_SUFFIX}"),
                                                             pddl_plus_hanoi_domain).parse_problem()

            # check that we could load the trajectory from the resulted path if needed
            with open(f"{HANOI_OUTPUT_DIR_PATH_TEMP}/{hanoi_problem_name}_trajectory.json", 'r') as file:
                GT_trajectory = json.load(file)
            GT_observation: Observation = create_observation_from_trajectory(GT_trajectory, pddl_plus_hanoi_domain,
                                                                             pddl_plus_hanoi_problem)

            # Output the resulting Observation object
            print("printing GT observation:")
            for component in GT_observation.components:
                print(str(component))

            print("*****************************")

            trajectory_parser = TrajectoryParser(pddl_plus_hanoi_domain, pddl_plus_hanoi_problem)
            imaged_observation = trajectory_parser.parse_trajectory(HANOI_OUTPUT_DIR_PATH_TEMP / f'{hanoi_problem_name}.trajectory')

            # Output the resulting Observation object
            print("printing imaged observation:")
            for component in imaged_observation.components:
                print(str(component))

            print("*****************************")
            # initial_state_predicates = set.union(*(imaged_observation.components[0].previous_state.state_predicates.values()))
            # sam_learner = SAMLearner(pddl_plus_blocks_domain, negative_preconditions_policy=NegativePreconditionPolicy.hard)
            #
            # partial_domain, report = sam_learner.learn_action_model([imaged_observation])
            # print(partial_domain.to_pddl())
            # print(report)

            """ copy problem file and trajectory file into the vip_experiments dir to automate the process"""
            shutil.copy(f'./problems/{hanoi_problem_name}.pddl', WORKING_DIRECTORY_PATH / f'{hanoi_problem_name}.pddl')
            shutil.copy(HANOI_OUTPUT_DIR_PATH_TEMP / f'{hanoi_problem_name}.trajectory', WORKING_DIRECTORY_PATH / f'{hanoi_problem_name}.trajectory')

        """ copy the GT domain file into the vip_experiments dir to automate the process"""
        shutil.copy(HANOI_DOMAIN_FILE_PATH, WORKING_DIRECTORY_PATH / HANOI_DOMAIN_FILE_PATH)

        experiment_runner = OfflineBasicSamExperimentRunner(
            working_directory_path=WORKING_DIRECTORY_PATH,
            domain_file_name=HANOI_DOMAIN_FILE_PATH.name,
            problem_prefix="problem"
        )

        experiment_runner.run_cross_validation()
