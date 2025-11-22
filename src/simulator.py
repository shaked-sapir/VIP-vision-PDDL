import shutil
import os
from pathlib import Path
from time import time
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import set_start_method

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
from src.utils.masking import save_masking_info, mask_observation, load_masking_info
from src.utils.pddl import ground_observation_completely
from src.utils.time import create_experiment_timestamp


# Module-level function for process-based parallelization
def _create_single_problem_trajectory(
    domain_name: str,
    problem_name: str,
    num_steps: int,
    working_dir: Path,
    problem_index: int,
    total_problems: int,
    openai_apikey: str,
    visual_components_model_name: str,
    visual_components_temperature: float,
    pddl_domain_file: Path,
    problem_dir: Path
) -> Tuple[str, bool]:
    """
    Create trajectory for a single problem (module-level function for multiprocessing).

    This function is designed to run in a separate process, so it cannot access
    instance variables and must receive all necessary parameters.

    :param domain_name: Name of the gym domain.
    :param problem_name: Name of the problem to process.
    :param num_steps: Number of steps in the trajectory.
    :param working_dir: Working directory for saving outputs.
    :param problem_index: Index of this problem (for progress reporting).
    :param total_problems: Total number of problems being processed.
    :param openai_apikey: OpenAI API key.
    :param visual_components_model_name: LLM model name.
    :param visual_components_temperature: Temperature for LLM components.
    :param pddl_domain_file: Path to PDDL domain file.
    :param problem_dir: Directory containing problem files.
    :return: Tuple of (problem_name, success_status).
    """
    try:
        print(f"\n[{problem_index}/{total_problems}] Starting problem: {problem_name} (PID: {os.getpid()})")

        # Parse domain for this process
        domain = DomainParser(pddl_domain_file).parse_domain()

        # Create trajectory handler for this process
        image_trajectory_handler = LLMBlocksImageTrajectoryHandler(
            domain_name,
            openai_apikey,
            object_detector_model=visual_components_model_name,
            object_detection_temperature=visual_components_temperature,
            fluent_classifier_model=visual_components_model_name,
            fluent_classification_temperature=visual_components_temperature
        )

        # Create experiment directory structure
        experiment_path = working_dir / f"{domain_name}_{problem_name}_steps={num_steps}"
        experiment_path.mkdir(parents=True, exist_ok=True)

        temp_output_dir = experiment_path / f"{problem_name}_temp_output"
        temp_output_dir.mkdir(parents=True, exist_ok=True)

        images_dir = temp_output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Generate trajectory using LLM
        ground_actions = image_trajectory_handler.create_trajectory_from_gym(
            problem_name, images_dir, num_steps
        )

        init_state_image_path = images_dir / "state_0000.png"
        image_trajectory_handler.init_visual_components(init_state_image_path)

        imaged_trajectory = image_trajectory_handler.image_trajectory_pipeline(
            problem_name=problem_name, actions=ground_actions, images_path=images_dir
        )

        # Extract masking info from LLM results (unknown predicates)
        trajectory_masking_info = (
            [parse_grounded_predicates(imaged_trajectory[0]['current_state']['unknown'], domain)] +
            [parse_grounded_predicates(step['next_state']['unknown'], domain)
             for step in imaged_trajectory]
        )

        # Save to working directory (where experiment runner will find it)
        save_masking_info(working_dir, problem_name, trajectory_masking_info)

        # Copy files to working directory
        shutil.copy(f'{problem_dir}/{problem_name}.pddl', working_dir / f'{problem_name}.pddl')
        shutil.copy(images_dir / f'{problem_name}.trajectory', working_dir / f'{problem_name}.trajectory')

        print(f"✓ [{problem_index}/{total_problems}] Completed {problem_name} (PID: {os.getpid()})")
        return (problem_name, True)

    except Exception as e:
        print(f"✗ [{problem_index}/{total_problems}] Failed {problem_name}: {str(e)} (PID: {os.getpid()})")
        import traceback
        traceback.print_exc()
        return (problem_name, False)


class Simulator:
    """
    VIP-vision-PDDL Simulator that combines trajectory generation, LLM-based object detection,
    fluent classification, and PI-SAM learning for action model learning from visual observations.
    """

    def __init__(self, domain_name: str, openai_apikey: str,
                 pddl_domain_file: Path, pddl_problem_dir: Path,
                 visual_components_model_name: str,
                 visual_components_temperature: float = 1.0,
                 experiment_dir_path: Path = Path("vip_experiments")):
        """
        Initialize the simulator.

        :param domain_name: Name of the PDDL gym domain (e.g., 'PDDLEnvBlocks-v0').
        :param openai_apikey: OpenAI API key for LLM-based components.
        :param pddl_domain_file: Path to the PDDL domain file.
        :param pddl_problem_dir: Directory containing PDDL problem files.
        :param visual_components_model_name: Name of the LLM model to use (e.g., 'gpt-4o').
        :param experiment_dir_path: Directory for experiment outputs.
        """
        self.domain_name = domain_name
        self.experiment_dir_path = experiment_dir_path
        self.openai_apikey = openai_apikey
        self.visual_components_model_name = visual_components_model_name
        self.visual_components_temperature = visual_components_temperature

        # Initialize trajectory handler for blocks domain, object detection & fluent classification done with same model
        self.image_trajectory_handler: ImageTrajectoryHandler = LLMBlocksImageTrajectoryHandler(
            domain_name,
            openai_apikey,
            object_detector_model=visual_components_model_name,
            object_detection_temperature=visual_components_temperature,
            fluent_classifier_model=visual_components_model_name,
            fluent_classification_temperature=visual_components_temperature
        )

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

    def create_trajectory(self, problem_name: str, num_steps: int = 25) -> Tuple[List[dict], Path]:
        """
        Create a trajectory by running the environment and generating images.

        :param problem_name: Name of the problem to solve.
        :param num_steps: Number of steps in the trajectory.
        :return: Tuple of (imaged trajectory, experiment directory path).
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
        return imaged_trajectory, experiment_path

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

    def run_simple_example(self, problem_name: str, num_steps: int = 25) -> Tuple[LearnerDomain, Dict[str, str], Path]:
        """
        Run a simple single-problem example:
        1. Create trajectory with LLM-based detection/classification
        2. Build observation from trajectory
        3. Ground and mask observation using saved masking info
        4. Run PI-SAM learning

        All outputs (trajectory, masking info, learned model) are saved to the experiment directory.

        :param problem_name: Name of the problem to solve.
        :param num_steps: Number of steps in the trajectory.
        :return: Tuple of (learned domain, learning report, experiment directory path).
        """
        print(f"\n{'='*80}")
        print(f"RUNNING SIMPLE EXAMPLE: {problem_name}")
        print(f"{'='*80}\n")

        # Step 1: Create trajectory (saves masking info to experiment_path)
        print(f"Step 1: Creating trajectory with {num_steps} steps...")
        imaged_trajectory, experiment_path = self.create_trajectory(problem_name, num_steps=num_steps)

        # Step 2: Build observation from trajectory
        print(f"\nStep 2: Building observation from trajectory...")
        observation = self.build_observation_from_trajectory(problem_name, imaged_trajectory)
        print(f"  Observation has {len(observation.components)} components")

        # Step 3: Ground observation
        print(f"\nStep 3: Grounding observation...")
        grounded_observation = ground_observation_completely(self.domain, observation)
        print(f"  Observation grounded")

        # Step 4: Load masking info from experiment directory
        print(f"\nStep 4: Loading masking info from experiment directory...")
        masking_file = experiment_path / f"{problem_name}.masking_info"
        trajectory_masking_info = load_masking_info(masking_file, self.domain)
        print(f"  Loaded masking info for {len(trajectory_masking_info)} states")

        # Step 5: Mask observation
        print(f"\nStep 5: Masking observation...")
        masked_observation = mask_observation(grounded_observation, trajectory_masking_info)
        print(f"  Observation masked")

        # Step 6: Run PI-SAM learning
        print(f"\nStep 6: Running PI-SAM learning...")
        learnt_domain, learning_report = self.run_pisam(masked_observation)

        print(f"\n{'='*80}")
        print(f"SIMPLE EXAMPLE COMPLETE")
        print(f"{'='*80}")
        print(f"Experiment directory: {experiment_path}")
        print(f"  - Trajectory: {experiment_path / problem_name}.trajectory")
        print(f"  - Masking info: {experiment_path / problem_name}.masking_info")
        print(f"  - Problem file: {experiment_path / problem_name}.pddl")

        return learnt_domain, learning_report, experiment_path

    def run_cross_validation_with_llm(
        self,
        problems: List[str],
        num_steps: int = 25,
        experiment_name: str = "llm_cv",
        max_workers: int = None
    ) -> Path:
        """
        Run complete cross-validation pipeline with LLM-based masking.

        This function:
        1. Generates trajectories for all problems using LLM-based detection/classification (in parallel)
        2. Uses LLM-derived masking info (unknown predicates)
        3. Runs cross-validation with PiSamExperimentRunner

        :param problems: List of problem names to process.
        :param num_steps: Number of steps per trajectory.
        :param experiment_name: Name prefix for the experiment directory.
        :param max_workers: Maximum number of parallel workers. None means CPU count.
        :return: Path to experiment results directory.
        """
        print(f"="*80)
        print(f"Running Cross-Validation with LLM-Based Masking")
        print(f"Domain: {self.domain_name}")
        print(f"Problems: {problems}")
        print(f"Steps per trajectory: {num_steps}")
        print(f"Parallel workers: {max_workers if max_workers else 'auto'}")
        print(f"="*80)

        # Create working directory for this experiment run
        timestamp = create_experiment_timestamp()
        working_dir = self.experiment_dir_path / f"{experiment_name}__steps={num_steps}__{timestamp}"
        working_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nWorking directory: {working_dir}")

        # Step 1: Generate trajectories for all problems using LLM (in parallel)
        print(f"\n{'='*80}")
        print(f"Step 1: Generating Trajectories with LLM-based detection/classification (PARALLEL)")
        print(f"{'='*80}\n")

        # Use ProcessPoolExecutor for parallel trajectory creation
        # Process-based to avoid state sharing issues between parallel trajectory generations
        # Each process gets its own domain parser and trajectory handler
        successful_problems = []
        failed_problems = []

        print(f"Processing {len(problems)} problems in parallel...")
        trajectory_start_time = time()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all trajectory creation tasks to separate processes
            future_to_problem = {
                executor.submit(
                    _create_single_problem_trajectory,  # Module-level function for multiprocessing
                    self.domain_name,
                    problem_name,
                    num_steps,
                    working_dir,
                    i,
                    len(problems),
                    self.openai_apikey,
                    self.image_trajectory_handler.object_detector_model,
                    self.image_trajectory_handler.object_detector_temperature,
                    self.pddl_domain_file,
                    self.problem_dir
                ): problem_name
                for i, problem_name in enumerate(problems, 1)
            }

            # Collect results as they complete
            for future in as_completed(future_to_problem):
                problem_name, success = future.result()
                if success:
                    successful_problems.append(problem_name)
                else:
                    failed_problems.append(problem_name)

        trajectory_end_time = time()

        # Report results
        print(f"\n{'='*80}")
        print(f"Trajectory Generation Complete")
        print(f"{'='*80}")
        print(f"✓ Successful: {len(successful_problems)}/{len(problems)}")
        if failed_problems:
            print(f"✗ Failed: {len(failed_problems)}/{len(problems)}")
            print(f"  Failed problems: {', '.join(failed_problems)}")
        print(f"Total trajectory generation time: {trajectory_end_time - trajectory_start_time:.2f} seconds")
        print()

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
    visual_components_model_name = config['openai']['visual_components_model']['model_name']

    if openai_apikey == "your-api-key-here":
        raise ValueError(
            "Please set your OpenAI API key in config.yaml\n"
            "Copy config.example.yaml to config.yaml and add your API key."
        )

    # Choose which example to run
    RUN_SIMPLE_EXAMPLE = True  # Set to True to run simple single-problem example
    RUN_CROSS_VALIDATION = False  # Set to True to run cross-validation with LLM

    if RUN_SIMPLE_EXAMPLE:
        domain = 'blocks'
        problem_name = "problem9"
        num_steps = 10

        simulator = Simulator(
            domain_name=config['domains'][domain]['gym_domain_name'],
            experiment_dir_path=Path(config['paths']['experiments_dir']),
            openai_apikey=openai_apikey,
            pddl_domain_file=Path(config['domains'][domain]['domain_file']),
            pddl_problem_dir=Path(config['domains'][domain]['problems_dir']),
            visual_components_model_name=visual_components_model_name,
            visual_components_temperature=config['openai']['visual_components_model']['temperature']
        )

        simulation_start_time = time()

        # Run the simple example - all outputs saved to experiment directory
        learnt_domain, learnt_report, experiment_path = simulator.run_simple_example(
            problem_name, num_steps=num_steps
        )

        simulation_end_time = time()

        print(f"\n{'='*80}")
        print(f"RESULTS")
        print(f"{'='*80}")
        print(f"\nTotal simulation time: {simulation_end_time - simulation_start_time:.2f} seconds")
        print(f"\nLearned Domain:\n{learnt_domain.to_pddl()}")
        print(f"\nLearning Report:\n{learnt_report}")
        print(f"\nAll files saved to: {experiment_path}")

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
        num_steps = 25  # Number of steps per trajectory
        experiment_name = f"llm_cv_test__{domain_name}__{visual_components_model_name}"  # Name for this experiment
        # Create simulator
        simulator = Simulator(
            domain_name=domain_name,
            experiment_dir_path=Path(config['paths']['experiments_dir']),
            openai_apikey=openai_apikey,
            pddl_domain_file=Path(config['domains'][domain]['domain_file']),
            pddl_problem_dir=Path(config['domains'][domain]['problems_dir']),
            visual_components_model_name=visual_components_model_name,
            visual_components_temperature=config['openai']['visual_components_model']['temperature']
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
