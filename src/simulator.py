import shutil
from pathlib import Path
from time import time
from typing import List, Dict, Tuple

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import Domain, Problem, Observation, GroundedPredicate
from sam_learning.core import LearnerDomain
from utilities import NegativePreconditionPolicy

from src.action_model.gym2SAM_parser import create_observation_from_trajectory, parse_grounded_predicates
from src.pi_sam.pi_sam_learning import PISAMLearner
from src.pi_sam.pisam_experiment_runner import OfflinePiSamExperimentRunner
from src.trajectory_handlers.image_trajectory_handler import ImageTrajectoryHandler
from src.trajectory_handlers.llm_blocks_trajectory_handler import LLMBlocksImageTrajectoryHandler
from src.utils.config import load_config
from src.utils.masking import save_masking_info, mask_observation
from src.utils.pddl import ground_observation_completely
from src.utils.time import create_experiment_timestamp


class Simulator:
    """
    VIP-vision-PDDL Simulator that combines trajectory generation, LLM-based object detection,
    fluent classification, and PI-SAM learning for action model learning from visual observations.
    """

    def __init__(self, domain_name: str, openai_apikey: str,
                 pddl_domain_file: Path, pddl_problem_dir: Path, experiment_dir_path: Path = Path("vip_experiments"),
                 llm_model_name: str = "gpt-4o"):
        """
        Initialize the simulator.

        :param domain_name: Name of the PDDL gym domain (e.g., 'PDDLEnvBlocks-v0').
        :param openai_apikey: OpenAI API key for LLM-based components.
        :param pddl_domain_file: Path to the PDDL domain file.
        :param pddl_problem_dir: Directory containing PDDL problem files.
        :param experiment_dir_path: Directory for experiment outputs.
        """
        self.domain_name = domain_name
        self.experiment_dir_path = experiment_dir_path
        self.openai_apikey = openai_apikey

        # Initialize trajectory handler for blocks domain
        self.image_trajectory_handler: ImageTrajectoryHandler = LLMBlocksImageTrajectoryHandler(domain_name, openai_apikey, llm_model_name, llm_model_name)

        # Parse domain and set problem directory
        self.domain: Domain = DomainParser(pddl_domain_file).parse_domain()
        self.problem_dir: Path = pddl_problem_dir

        # Store domain file path for experiment runner
        self.pddl_domain_file = pddl_domain_file

        # Ensure output directory exists
        self.experiment_dir_path.mkdir(parents=True, exist_ok=True)

        print(f"Simulator initialized for domain: {domain_name}")
        print(f"Output directory: {experiment_dir_path}")
        print(f"PDDL domain file: {pddl_domain_file}")
        print(f"PDDL problems directory: {pddl_problem_dir}")

    def create_trajectory(self, problem_name: str, num_steps: int = 25) -> List[dict]:
        """
        Create a trajectory by running the environment and generating images.

        :param problem_name: Name of the problem to solve.
        :param num_steps: Number of steps in the trajectory.
        :return: Imaged trajectory (list of trajectory steps with visual annotations).
        """
        print(f"Running {self.domain_name} | problem: {problem_name} with {num_steps} steps")

        experiment_path = self.experiment_dir_path / f"{self.domain_name}_{problem_name}_steps={num_steps}___{create_experiment_timestamp()}"
        experiment_path.mkdir(parents=True, exist_ok=True)

        # Create temporary output directory for intermediate files
        temp_output_dir = experiment_path / f"{problem_name}_temp_output"
        temp_output_dir.mkdir(parents=True, exist_ok=True)

        # Create images directory for this problem
        images_dir = temp_output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Generate trajectory and save images using the trajectory handler
        ground_actions = self.image_trajectory_handler.create_trajectory_from_gym(
            problem_name, images_dir, num_steps
        )

        # Initialize visual components (object detector and fluent classifier)
        init_state_image_path = images_dir / "state_0000.png"
        self.image_trajectory_handler.init_visual_components(init_state_image_path)

        # Generate imaged trajectory using LLM classification
        imaged_trajectory: List[dict] = self.image_trajectory_handler.image_trajectory_pipeline(
            problem_name=problem_name, actions=ground_actions, images_path=images_dir
        )

        # Extract masking info from trajectory (unknown predicates)
        trajectory_masking_info: list[set[GroundedPredicate]] = (
            [parse_grounded_predicates(imaged_trajectory[0]['current_state']['unknown'], self.domain)] +
            [
                parse_grounded_predicates(step['next_state']['unknown'], self.domain)
                for step in imaged_trajectory
            ]
        )

        # Save masking info to file for later use
        save_masking_info(experiment_path, problem_name, trajectory_masking_info)

        # Copy problem file and trajectory file into the experiments dir
        shutil.copy(f'{self.problem_dir}/{problem_name}.pddl', experiment_path / f'{problem_name}.pddl')
        shutil.copy(images_dir / f'{problem_name}.trajectory',
                    experiment_path / f'{problem_name}.trajectory')

        print(f"Trajectory and images saved to: {experiment_path}")
        return imaged_trajectory

    def build_observation_from_trajectory(self, problem_name: str, imaged_trajectory: List[dict]) -> Observation:
        """
        Convert imaged trajectory to PDDL+ Observation for PI-SAM consumption.

        :param problem_name: Name of the problem.
        :param imaged_trajectory: Imaged trajectory from create_trajectory().
        :return: PDDL+ Observation object.
        """
        problem_file = self.problem_dir / f"{problem_name}.pddl"
        problem: Problem = ProblemParser(problem_file, self.domain).parse_problem()

        # Create observation from trajectory using existing parser
        observation = create_observation_from_trajectory(imaged_trajectory, self.domain, problem)

        print(f"Observation created with {len(observation.components)} components")
        return observation

    def run_pisam(self, masked_observation: Observation) -> Tuple[LearnerDomain, Dict[str, str]]:
        """
        Run PI-SAM learning on an already-masked observation.

        IMPORTANT: The observation should be grounded and masked before calling this method.

        :param masked_observation: A grounded and masked observation ready for learning.
        :return: Tuple of (learned domain, learning report).
        """
        print("Running PI-SAM learning...")

        pisam_learner = PISAMLearner(
            self.domain,
            negative_preconditions_policy=NegativePreconditionPolicy.hard
        )

        # Learn action model from already-masked observation
        learnt_domain, learning_report = pisam_learner.learn_action_model([masked_observation])

        print("PI-SAM learning completed")
        print(f"Learning report: {learning_report}")

        return learnt_domain, learning_report

    def run_cross_validation_with_llm(
        self,
        problems: List[str],
        num_steps: int = 25,
        experiment_name: str = "llm_cv"
    ) -> Path:
        """
        Run complete cross-validation pipeline with LLM-based masking.

        This function:
        1. Generates trajectories for all problems using LLM-based detection/classification
        2. Uses LLM-derived masking info (unknown predicates)
        3. Runs cross-validation with PiSamExperimentRunner

        :param problems: List of problem names to process.
        :param num_steps: Number of steps per trajectory.
        :param experiment_name: Name prefix for the experiment directory.
        :return: Path to experiment results directory.
        """
        print(f"="*80)
        print(f"Running Cross-Validation with LLM-Based Masking")
        print(f"Domain: {self.domain_name}")
        print(f"Problems: {problems}")
        print(f"Steps per trajectory: {num_steps}")
        print(f"="*80)

        # Create working directory for this experiment run
        timestamp = create_experiment_timestamp()
        working_dir = self.experiment_dir_path / f"{experiment_name}__steps={num_steps}__{timestamp}"
        working_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nWorking directory: {working_dir}")

        # Step 1: Generate trajectories for all problems using LLM
        print(f"\n{'='*80}")
        print(f"Step 1: Generating Trajectories with LLM-based detection/classification")
        print(f"{'='*80}\n")

        for i, problem_name in enumerate(problems, 1):
            print(f"\n[{i}/{len(problems)}] Processing problem: {problem_name}")

            # Create trajectory with LLM-based detection/classification
            # This automatically extracts unknown predicates as masking info
            experiment_path = working_dir / f"{self.domain_name}_{problem_name}_steps={num_steps}"
            experiment_path.mkdir(parents=True, exist_ok=True)

            temp_output_dir = experiment_path / f"{problem_name}_temp_output"
            temp_output_dir.mkdir(parents=True, exist_ok=True)

            images_dir = temp_output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            # Generate trajectory using LLM
            ground_actions = self.image_trajectory_handler.create_trajectory_from_gym(
                problem_name, images_dir, num_steps
            )

            init_state_image_path = images_dir / "state_0000.png"
            self.image_trajectory_handler.init_visual_components(init_state_image_path)

            imaged_trajectory = self.image_trajectory_handler.image_trajectory_pipeline(
                problem_name=problem_name, actions=ground_actions, images_path=images_dir
            )

            # Extract masking info from LLM results (unknown predicates)
            trajectory_masking_info = (
                [parse_grounded_predicates(imaged_trajectory[0]['current_state']['unknown'], self.domain)] +
                [parse_grounded_predicates(step['next_state']['unknown'], self.domain)
                 for step in imaged_trajectory]
            )

            # Save to working directory (where experiment runner will find it)
            save_masking_info(working_dir, problem_name, trajectory_masking_info)

            # Copy files to working directory
            shutil.copy(f'{self.problem_dir}/{problem_name}.pddl', working_dir / f'{problem_name}.pddl')
            shutil.copy(images_dir / f'{problem_name}.trajectory', working_dir / f'{problem_name}.trajectory')

            print(f"✓ Completed {problem_name}")

        # Step 2: Copy domain file to working directory
        print(f"\n{'='*80}")
        print(f"Step 2: Preparing Experiment Files")
        print(f"{'='*80}\n")

        shutil.copy(self.pddl_domain_file, working_dir / self.pddl_domain_file.name)
        print(f"✓ Copied domain file: {self.pddl_domain_file.name}")

        # Step 3: Run cross-validation with PiSamExperimentRunner
        print(f"\n{'='*80}")
        print(f"Step 3: Running PI-SAM Cross-Validation")
        print(f"{'='*80}\n")
        #

        # working_dir = Path("experiments/llm_cv_test__steps=3__31-10-2025T00:57:34")
        experiment_runner = OfflinePiSamExperimentRunner(
            working_directory_path=working_dir,
            domain_file_name=self.pddl_domain_file.name,
            problem_prefix="problem"
        )

        experiment_runner.run_cross_validation()

        print(f"\n{'='*80}")
        print(f"Cross-Validation Complete!")
        print(f"{'='*80}")

        # Check for results
        results_file = working_dir / "results_directory/sam_learning_blocks_combined_semantic_performance.csv"
        if results_file.exists():
            print(f"✓ Results saved to: {results_file}")
        else:
            print(f"⚠ Warning: Results file not found at expected location")

        print(f"\nAll experiment files saved to: {working_dir}")

        return working_dir


# Example usage with config loading
if __name__ == '__main__':
    # Load configuration
    config = load_config()

    # Get API key from config
    openai_apikey = config['openai']['api_key']
    llm_model_name = config['openai'].get('llm_model_name', 'gpt-4o')

    if openai_apikey == "your-api-key-here":
        raise ValueError(
            "Please set your OpenAI API key in config.yaml\n"
            "Copy config.example.yaml to config.yaml and add your API key."
        )

    # Choose which example to run
    RUN_SIMPLE_EXAMPLE = False  # Set to True to run simple single-problem example
    RUN_CROSS_VALIDATION = True  # Set to True to run cross-validation with LLM

    if RUN_SIMPLE_EXAMPLE:
        print("\n" + "="*80)
        print("RUNNING SIMPLE SINGLE-PROBLEM EXAMPLE")
        print("="*80 + "\n")

        domain = 'blocks'
        curr_problem = "problem1"
        simulator = Simulator(
            domain_name=config['domains'][domain]['gym_domain_name'],
            experiment_dir_path=Path(config['paths']['experiments_dir']),
            openai_apikey=openai_apikey,
            pddl_domain_file=Path(config['domains'][domain]['domain_file']),
            pddl_problem_dir=Path(config['domains'][domain]['problems_dir']),
            llm_model_name=llm_model_name
        )

        simulation_start_time = time()
        imaged_trajectory = simulator.create_trajectory(curr_problem, num_steps=3)
        observation = simulator.build_observation_from_trajectory(curr_problem, imaged_trajectory)
        print(f"Observation has {len(observation.components)} components")

        # Ground and mask observation
        grounded_observation = ground_observation_completely(simulator.domain, observation)

        # Use LLM-derived masking info (from unknown predicates in trajectory)
        trajectory_masking_info = (
            [parse_grounded_predicates(
                imaged_trajectory[0]['current_state']['literals'] +
                imaged_trajectory[0]['current_state']['unknown'],
                simulator.domain)]
            +
            [parse_grounded_predicates(
                step['next_state']['literals'] + step['next_state']['unknown'],
                simulator.domain)
             for step in imaged_trajectory]
        )

        masked_observation = mask_observation(grounded_observation, trajectory_masking_info)

        # Learn from masked observation
        learnt_domain, learnt_report = simulator.run_pisam(masked_observation)

        simulation_end_time = time()
        print(f"Total simulation time: {simulation_end_time - simulation_start_time:.2f} seconds")
        print(learnt_domain.to_pddl())
        print(learnt_report)

    if RUN_CROSS_VALIDATION:
        print("\n" + "="*80)
        print("RUNNING CROSS-VALIDATION WITH LLM-BASED MASKING")
        print("="*80 + "\n")

        # Configuration for cross-validation experiment
        domain = 'blocks'
        domain_name = config['domains'][domain]['gym_domain_name']
        problems = [
            "problem1",
            "problem3",
            "problem5",
            "problem7",
            "problem9"
        ]  # List of problems for cross-validation
        num_steps = 1  # Number of steps per trajectory
        experiment_name = f"llm_cv_test__{domain_name}__{llm_model_name}"  # Name for this experiment
        # Create simulator
        simulator = Simulator(
            domain_name=domain_name,
            experiment_dir_path=Path(config['paths']['experiments_dir']),
            openai_apikey=openai_apikey,
            pddl_domain_file=Path(config['domains'][domain]['domain_file']),
            pddl_problem_dir=Path(config['domains'][domain]['problems_dir']),
            llm_model_name=llm_model_name
        )

        # Run cross-validation with LLM-based procedures
        cv_start_time = time()
        results_dir = simulator.run_cross_validation_with_llm(
            problems=problems,
            num_steps=num_steps,
            experiment_name=experiment_name
        )
        cv_end_time = time()

        print("\n" + "="*80)
        print(f"CROSS-VALIDATION COMPLETE!")
        print("="*80)
        print(f"Total time: {cv_end_time - cv_start_time:.2f} seconds")
        print(f"Results directory: {results_dir}")

        # Check if results file exists and print location
        results_file = results_dir / "results_directory/sam_learning_blocks_combined_semantic_performance.csv"
        if results_file.exists():
            print(f"\nResults CSV: {results_file}")
            print("\nYou can analyze the results by opening the CSV file or checking the fold directories.")
        else:
            print("\nNote: Results file not found at expected location. Check the results_directory folder.")
            print(f"All outputs saved to: {results_dir}")
