import json
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
from src.pi_sam.pi_sam_learning import PISAMLearner, MaskingType
from src.trajectory_handlers.blocks_image_trajectory_handler import BlocksImageTrajectoryHandler

BLOCKS_DOMAIN_FILE_PATH = Path("blocks.pddl")
BLOCKS_PROBLEM_DIR_PATH = Path("problems")
WORKING_DIRECTORY_PATH = Path("experiments")
PDDL_FILES_SUFFIX = ".pddl"


# TODO IMM: this should be refactored into an Experiments package, and not be contained inside "domains" package
if __name__ == "__main__":
    # """
    # This is after refactoring to the BlocksImageTrajectoryHandler class.
    # The loop is for generating the data to the cross validation evaluation.
    # #  TODO: separate the simulations from the evaluation.
    # """
    # blocks_domain_name = "PDDLEnvBlocks-v0"
    # for blocks_problem_name in [
    #     "problem1",
    #     "problem3",
    #     "problem5",
    #     "problem7",
    #     "problem9"
    # ]:
    #     BLOCKS_OUTPUT_DIR_PATH_TEMP = Path(f"{blocks_domain_name}_{blocks_problem_name}_temp")
    #     #
    #     print("alg started")
    #     image_trajectory_handler = BlocksImageTrajectoryHandler(blocks_domain_name)
    #
    #     # TODO: rename the method to a more indicative name/make it return something, right now it looks a bit odd
    #     image_trajectory_handler.image_trajectory_pipeline(problem_name=blocks_problem_name, output_path=BLOCKS_OUTPUT_DIR_PATH_TEMP)
    #
    #     pddl_plus_blocks_domain: Domain = DomainParser(BLOCKS_DOMAIN_FILE_PATH).parse_domain()
    #     pddl_plus_blocks_problem: Problem = ProblemParser(Path(f"{BLOCKS_PROBLEM_DIR_PATH}/{blocks_problem_name}{PDDL_FILES_SUFFIX}"),
    #                                                       pddl_plus_blocks_domain).parse_problem()
    #
    #     # check that we could load the trajectory from the resulted path if needed
    #     with open(f"{BLOCKS_OUTPUT_DIR_PATH_TEMP}/{blocks_problem_name}_trajectory.json", 'r') as file:
    #         GT_trajectory = json.load(file)
    #     GT_observation: Observation = create_observation_from_trajectory(GT_trajectory, pddl_plus_blocks_domain,
    #                                                                      pddl_plus_blocks_problem)
    #
    #     # Output the resulting Observation object
    #     print("printing GT observation:")
    #     for component in GT_observation.components:
    #         print(str(component))
    #
    #     print("*****************************")
    #
    #     trajectory_parser = TrajectoryParser(pddl_plus_blocks_domain, pddl_plus_blocks_problem)
    #     imaged_observation = trajectory_parser.parse_trajectory(BLOCKS_OUTPUT_DIR_PATH_TEMP / f'{blocks_problem_name}.trajectory')
    #
    #     # Output the resulting Observation object
    #     print("printing imaged observation:")
    #     for component in imaged_observation.components:
    #         print(str(component))
    #
    #     print("*****************************")
    #
    #     """
    #     This is a trial for the masking procedures
    #     """
    #     initial_state_predicates = set.union(*(imaged_observation.components[0].previous_state.state_predicates.values()))
    #     pi_state = imaged_observation.components[0].previous_state
    #     pi_state.state_predicates = {
    #         sig: {predicate for predicate in predicate_set}
    #         for sig, predicate_set in pi_state.state_predicates.items()
    #     }
    #     # maskable_predicates = set([MaskableGroundedPredicate.from_grounded_predicate(predicate) for predicate in initial_state_predicates])
    #     # masking_strategy = PercentageMasking()
    #     pi_sam_learner = PISAMLearner(pddl_plus_blocks_domain)
    #     for sig, predicate_set in pi_state.state_predicates.items():
    #         predicate_set = pi_sam_learner.mask(predicate_set, masking_strategy=MaskingType.PERCENTAGE, masking_ratio=0.25)
    #     # masked_predicates = masking_strategy.mask(maskable_predicates, masking_ratio=0.5)
    #     print(pi_state)
    #
    #     """
    #     Trial end
    #     """
    #     sam_learner = SAMLearner(pddl_plus_blocks_domain, negative_preconditions_policy=NegativePreconditionPolicy.hard)
    #
    #     partial_domain, report = sam_learner.learn_action_model([imaged_observation])
    #     print(partial_domain.to_pddl())
    #     print(report)

    experiment_runner = OfflineBasicSamExperimentRunner(
        working_directory_path=WORKING_DIRECTORY_PATH,
        domain_file_name=BLOCKS_DOMAIN_FILE_PATH.name,
        problem_prefix="problem"
    )

    experiment_runner.run_cross_validation()
