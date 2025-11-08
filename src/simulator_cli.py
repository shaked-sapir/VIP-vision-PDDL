import argparse
import shutil
import yaml
from pathlib import Path
from time import time
from typing import List, Dict, Tuple, Optional

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import Domain, Problem, Observation, GroundedPredicate
from sam_learning.core import LearnerDomain
from utilities import NegativePreconditionPolicy

from src.action_model.gym2SAM_parser import create_observation_from_trajectory, parse_grounded_predicates
from src.pi_sam.pi_sam_learning import PISAMLearner
from src.pi_sam.predicate_masking import PredicateMasker
from src.pi_sam.masking.masking_strategies import MaskingType
from src.pi_sam.pisam_experiment_runner import OfflinePiSamExperimentRunner
from src.trajectory_handlers.image_trajectory_handler import ImageTrajectoryHandler
from src.trajectory_handlers.blocks_image_trajectory_handler import BlocksImageTrajectoryHandler
from src.trajectory_handlers.llm_blocks_trajectory_handler import LLMBlocksImageTrajectoryHandler
from src.trajectory_handlers.hanoi_image_trajectory_handler import HanoiImageTrajectoryHandler
from src.trajectory_handlers.llm_hanoi_trajectory_handler import LLMHanoiImageTrajectoryHandler
from src.utils.masking import save_masking_info
from src.utils.time import create_experiment_timestamp
from src.utils.pddl import ground_observation_completely, mask_observation


# ============================================================================
# Configuration Management
# ============================================================================

def load_config(config_path: Path = None) -> dict:
    """
    Load configuration from YAML file.

    :param config_path: Path to config file. If None, uses default 'config.yaml' in project root.
    :return: Configuration dictionary.
    """
    if config_path is None:
        # Find project root (where config.yaml should be)
        current_dir = Path(__file__).parent
        project_root = current_dir.parent  # Go up from src/ to project root
        config_path = project_root / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}.\n"
            f"Please copy config.example.yaml to config.yaml and fill in your values."
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


# ============================================================================
# Masking Strategy Functions
# ============================================================================

def mask_observation_with_percentage(
    domain: Domain,
    observation: Observation,
    masking_ratio: float = 0.8
) -> Tuple[Observation, List[List[set[GroundedPredicate]]]]:
    """
    Mask observation using PERCENTAGE masking strategy.

    Masks a percentage of predicates in each state.

    :param domain: The PDDL domain.
    :param observation: The observation to mask.
    :param masking_ratio: Percentage of predicates to mask (0.0-1.0).
    :return: Tuple of (masked_observation, masking_info).
    """
    print(f"Masking observation using PERCENTAGE strategy (ratio={masking_ratio})")

    # Ground observation completely
    grounded_observation = ground_observation_completely(domain, observation)

    # Create percentage masker
    masker = PredicateMasker(
        masking_strategy=MaskingType.PERCENTAGE,
        masking_kwargs={"masking_ratio": masking_ratio}
    )

    # Generate and apply masking
    masking_info = masker.mask_observation(grounded_observation)
    masked_observation = mask_observation(grounded_observation, masking_info)

    print(f"Observation masked successfully with PERCENTAGE strategy")
    return masked_observation, masking_info


def mask_observation_with_random(
    domain: Domain,
    observation: Observation,
    masking_probability: float = 0.3
) -> Tuple[Observation, List[List[set[GroundedPredicate]]]]:
    """
    Mask observation using RANDOM masking strategy.

    Each predicate has an independent probability of being masked.

    :param domain: The PDDL domain.
    :param observation: The observation to mask.
    :param masking_probability: Probability that each predicate is masked (0.0-1.0).
    :return: Tuple of (masked_observation, masking_info).
    """
    print(f"Masking observation using RANDOM strategy (probability={masking_probability})")

    # Ground observation completely
    grounded_observation = ground_observation_completely(domain, observation)

    # Create random masker
    masker = PredicateMasker(
        masking_strategy=MaskingType.RANDOM,
        masking_kwargs={"masking_proba": masking_probability}
    )

    # Generate and apply masking
    masking_info = masker.mask_observation(grounded_observation)
    masked_observation = mask_observation(grounded_observation, masking_info)

    print(f"Observation masked successfully with RANDOM strategy")
    return masked_observation, masking_info


def mask_observation_with_llm(
    domain: Domain,
    observation: Observation,
    imaged_trajectory: List[dict]
) -> Tuple[Observation, List[List[set[GroundedPredicate]]]]:
    """
    Extract masking info from LLM-generated trajectory (unknown predicates).

    :param domain: The PDDL domain.
    :param observation: The observation to mask.
    :param imaged_trajectory: Imaged trajectory with LLM-derived unknown predicates.
    :return: Tuple of (masked_observation, masking_info).
    """
    print("Extracting masking info from LLM-generated unknown predicates")

    # Ground observation completely
    grounded_observation = ground_observation_completely(domain, observation)

    # Extract masking info from unknown predicates in trajectory
    trajectory_masking_info = (
        [parse_grounded_predicates(
            imaged_trajectory[0]['current_state']['literals'] +
            imaged_trajectory[0]['current_state']['unknown'],
            domain)]
        +
        [parse_grounded_predicates(
            step['next_state']['literals'] + step['next_state']['unknown'],
            domain)
         for step in imaged_trajectory]
    )

    # Apply masking
    masked_observation = mask_observation(grounded_observation, trajectory_masking_info)

    print(f"Observation masked successfully using LLM-derived unknown predicates")
    return masked_observation, trajectory_masking_info


def mask_observation_with_strategy(
    domain: Domain,
    observation: Observation,
    strategy: str,
    **strategy_kwargs
) -> Tuple[Observation, List[List[set[GroundedPredicate]]]]:
    """
    Mask observation using specified masking strategy.

    Note: For 'llm' strategy, you must provide 'imaged_trajectory' in strategy_kwargs.

    :param domain: The PDDL domain.
    :param observation: The observation to mask.
    :param strategy: Masking strategy name ('percentage', 'random', or 'llm').
    :param strategy_kwargs: Strategy-specific parameters.
    :return: Tuple of (masked_observation, masking_info).
    """
    if strategy.lower() == "percentage":
        masking_ratio = strategy_kwargs.get('masking_ratio', 0.8)
        return mask_observation_with_percentage(domain, observation, masking_ratio)
    elif strategy.lower() == "random":
        masking_probability = strategy_kwargs.get('masking_probability', 0.3)
        return mask_observation_with_random(domain, observation, masking_probability)
    elif strategy.lower() == "llm":
        imaged_trajectory = strategy_kwargs.get('imaged_trajectory')
        if imaged_trajectory is None:
            raise ValueError("LLM masking strategy requires 'imaged_trajectory' parameter")
        return mask_observation_with_llm(domain, observation, imaged_trajectory)
    else:
        raise ValueError(f"Unknown masking strategy: {strategy}. Use 'percentage', 'random', or 'llm'.")


# ============================================================================
# Trajectory Handler Selection
# ============================================================================

def create_trajectory_handler(
    domain: str,
    domain_name: str,
    masking_strategy: str,
    openai_apikey: str
) -> ImageTrajectoryHandler:
    """
    Create appropriate trajectory handler based on domain and masking strategy.

    For 'percentage' or 'random' masking: Use deterministic object detector and fluent classifier.
    For 'llm' masking: Use LLM-based object detector and fluent classifier.

    :param domain: Domain identifier ('blocks' or 'hanoi').
    :param domain_name: PDDL gym domain name (e.g., 'PDDLEnvBlocks-v0').
    :param masking_strategy: Masking strategy ('percentage', 'random', or 'llm').
    :param openai_apikey: OpenAI API key (required for LLM-based handlers).
    :return: Configured ImageTrajectoryHandler instance.
    """
    if masking_strategy == 'llm':
        # Use LLM-based detection and classification
        if domain == 'blocks':
            print("Using LLM-based trajectory handler for Blocks")
            print("  - Object detector: LLMBlocksObjectDetector (GPT-4 Vision)")
            print("  - Fluent classifier: LLMBlocksFluentClassifier (GPT-4 Vision)")
            return LLMBlocksImageTrajectoryHandler(domain_name, openai_apikey)
        elif domain == 'hanoi':
            print("Using LLM-based trajectory handler for Hanoi")
            print("  - Object detector: LLMHanoiObjectDetector (GPT-4 Vision)")
            print("  - Fluent classifier: LLMHanoiFluentClassifier (GPT-4 Vision)")
            return LLMHanoiImageTrajectoryHandler(domain_name, openai_apikey)
        else:
            raise ValueError(f"Unknown domain: {domain}")
    else:
        # Use deterministic detection and classification
        if domain == 'blocks':
            print("Using deterministic trajectory handler for Blocks")
            print("  - Object detector: ColorObjectDetector (color-based)")
            print("  - Fluent classifier: BlocksContourFluentClassifier (geometric)")
            return BlocksImageTrajectoryHandler(domain_name)
        elif domain == 'hanoi':
            print("Using deterministic trajectory handler for Hanoi")
            print("  - Object detector: HanoiObjectDetector (position/size-based)")
            print("  - Fluent classifier: HanoiFluentClassifier (geometric)")
            return HanoiImageTrajectoryHandler(domain_name)
        else:
            raise ValueError(f"Unknown domain: {domain}")


# ============================================================================
# Simulator Class
# ============================================================================

class Simulator:
    """
    VIP-vision-PDDL Simulator that combines trajectory generation, LLM-based object detection,
    fluent classification, and PI-SAM learning for action model learning from visual observations.
    """

    def __init__(self, domain_name: str, openai_apikey: str,
                 pddl_domain_file: Path, pddl_problem_dir: Path, experiment_dir_path: Path = Path("vip_experiments"),
                 trajectory_handler: Optional[ImageTrajectoryHandler] = None):
        """
        Initialize the simulator.

        :param domain_name: Name of the PDDL gym domain (e.g., 'PDDLEnvBlocks-v0').
        :param openai_apikey: OpenAI API key for LLM-based components.
        :param pddl_domain_file: Path to the PDDL domain file.
        :param pddl_problem_dir: Directory containing PDDL problem files.
        :param experiment_dir_path: Directory for experiment outputs.
        :param trajectory_handler: Optional custom trajectory handler. If None, uses LLM-based handler.
        """
        self.domain_name = domain_name
        self.experiment_dir_path = experiment_dir_path
        self.openai_apikey = openai_apikey

        # Initialize trajectory handler
        if trajectory_handler is not None:
            self.image_trajectory_handler = trajectory_handler
        else:
            # Default to LLM-based handler for backwards compatibility
            self.image_trajectory_handler = LLMBlocksImageTrajectoryHandler(domain_name, openai_apikey)

        # Parse domain and set problem directory
        self.domain: Domain = DomainParser(pddl_domain_file).parse_domain()
        self.problem_dir: Path = pddl_problem_dir

        # Store domain file path for experiment runner
        self.pddl_domain_file = pddl_domain_file

        # Ensure output directory exists
        self.experiment_dir_path.mkdir(parents=True, exist_ok=True)

        print(f"Simulator initialized for domain: {domain_name}")
        print(f"Trajectory handler: {type(self.image_trajectory_handler).__name__}")
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
            [parse_grounded_predicates(imaged_trajectory[0]['current_state']['literals'] +
                                       imaged_trajectory[0]['current_state']['unknown'], self.domain)]
            +
            [
                parse_grounded_predicates(step['next_state']['literals'] +
                                          step['next_state']['unknown'], self.domain)
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

    def run_simple_pipeline(
        self,
        problem_name: str,
        num_steps: int = 25,
        masking_strategy: str = "percentage",
        **masking_kwargs
    ) -> Dict:
        """
        Run a simple VIP-vision-PDDL pipeline for a single problem:
        1. Generate trajectory with images
        2. Create imaged trajectory using object detection/classification
        3. Build observation from trajectory
        4. Mask observation for partial observability
        5. Run PI-SAM learning

        :param problem_name: Name of the problem to solve.
        :param num_steps: Number of steps in the trajectory.
        :param masking_strategy: Masking strategy ('percentage', 'random', or 'llm').
        :param masking_kwargs: Strategy-specific parameters.
        :return: Dictionary with pipeline results.
        """
        print(f"Starting simple VIP-vision-PDDL pipeline for {problem_name}")

        # Step 1 & 2: Create imaged trajectory
        imaged_trajectory = self.create_trajectory(problem_name, num_steps)

        # Step 3: Build observation
        observation = self.build_observation_from_trajectory(problem_name, imaged_trajectory)

        # Step 4: Mask observation
        # For LLM masking, pass the imaged_trajectory to extract unknown predicates
        if masking_strategy == 'llm':
            masking_kwargs['imaged_trajectory'] = imaged_trajectory

        masked_observation, masking_info = mask_observation_with_strategy(
            self.domain, observation, masking_strategy, **masking_kwargs
        )

        # Step 5: Run PI-SAM
        learnt_domain, learning_report = self.run_pisam(masked_observation)

        print("Simple pipeline completed successfully!")

        return {
            'imaged_trajectory': imaged_trajectory,
            'observation': observation,
            'masked_observation': masked_observation,
            'masking_info': masking_info,
            'learnt_domain': learnt_domain,
            'learning_report': learning_report
        }


# ============================================================================
# Full Pipeline with Cross-Validation (PiSamExperimentRunner)
# ============================================================================

def run_full_pipeline_with_cross_validation(
    domain: str,
    domain_name: str,
    openai_apikey: str,
    pddl_domain_file: Path,
    pddl_problem_dir: Path,
    experiment_dir: Path,
    problems: List[str],
    num_steps: int = 25,
    masking_strategy: str = "percentage",
    trajectory_handler: Optional[ImageTrajectoryHandler] = None,
    **masking_kwargs
) -> Path:
    """
    Run the complete VIP-vision-PDDL pipeline with cross-validation:
    1. Generate trajectories for all problems
    2. Create observations and save masking info
    3. Run PI-SAM cross-validation experiments
    4. Return results

    :param domain: Domain identifier ('blocks' or 'hanoi').
    :param domain_name: Name of the PDDL gym domain.
    :param openai_apikey: OpenAI API key.
    :param pddl_domain_file: Path to PDDL domain file.
    :param pddl_problem_dir: Directory with PDDL problem files.
    :param experiment_dir: Directory for experiments output.
    :param problems: List of problem names to process.
    :param num_steps: Number of steps in each trajectory.
    :param masking_strategy: Masking strategy name.
    :param trajectory_handler: Optional trajectory handler (if None, will create based on masking strategy).
    :param masking_kwargs: Strategy-specific parameters.
    :return: Path to experiment results directory.
    """
    print(f"="*80)
    print(f"Starting Full PI-SAM Pipeline with Cross-Validation")
    print(f"Domain: {domain_name}")
    print(f"Problems: {problems}")
    print(f"Steps per trajectory: {num_steps}")
    print(f"Masking strategy: {masking_strategy}")
    print(f"="*80)

    # Create working directory for this experiment run
    timestamp = create_experiment_timestamp()
    working_dir = experiment_dir / f"pisam_cv_{masking_strategy}__steps={num_steps}__{timestamp}"
    working_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nWorking directory: {working_dir}")

    # Initialize simulator
    simulator = Simulator(
        domain_name=domain_name,
        openai_apikey=openai_apikey,
        pddl_domain_file=pddl_domain_file,
        pddl_problem_dir=pddl_problem_dir,
        experiment_dir_path=working_dir,
        trajectory_handler=trajectory_handler
    )

    # Step 1: Generate trajectories for all problems
    print(f"\n{'='*80}")
    print(f"Step 1: Generating Trajectories for {len(problems)} problems")
    print(f"{'='*80}\n")

    for i, problem_name in enumerate(problems, 1):
        print(f"\n[{i}/{len(problems)}] Processing problem: {problem_name}")

        # Create trajectory
        imaged_trajectory = simulator.create_trajectory(problem_name, num_steps)

        # Build observation
        observation = simulator.build_observation_from_trajectory(problem_name, imaged_trajectory)

        # Mask observation with specified strategy
        # For LLM masking, pass the imaged_trajectory to extract unknown predicates
        if masking_strategy == 'llm':
            masking_kwargs['imaged_trajectory'] = imaged_trajectory

        masked_observation, masking_info = mask_observation_with_strategy(
            simulator.domain, observation, masking_strategy, **masking_kwargs
        )

        # Save masking info
        save_masking_info(working_dir, problem_name, masking_info)

        print(f"✓ Completed {problem_name}")

    # Step 2: Copy domain file to working directory
    print(f"\n{'='*80}")
    print(f"Step 2: Preparing Experiment Files")
    print(f"{'='*80}\n")

    shutil.copy(pddl_domain_file, working_dir / pddl_domain_file.name)
    print(f"✓ Copied domain file: {pddl_domain_file.name}")

    # Step 3: Run cross-validation with PiSamExperimentRunner
    print(f"\n{'='*80}")
    print(f"Step 3: Running PI-SAM Cross-Validation")
    print(f"{'='*80}\n")

    experiment_runner = OfflinePiSamExperimentRunner(
        working_directory_path=working_dir,
        domain_file_name=pddl_domain_file.name,
        problem_prefix="problem"  # Assumes problems are named problem1, problem2, etc.
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


# ============================================================================
# CLI Interface with argparse
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for the simulator CLI."""

    parser = argparse.ArgumentParser(
        description="VIP-vision-PDDL Simulator: Learn PDDL action models from visual observations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simple pipeline with deterministic detection and percentage masking
  python -m src.simulator_cli simple --problem problem1 --steps 25 --masking percentage --ratio 0.8

  # Run simple pipeline with LLM-based detection and masking
  python -m src.simulator_cli simple --problem problem1 --steps 25 --masking llm

  # Run full cross-validation with deterministic detection and random masking
  python -m src.simulator_cli full --domain blocks --problems problem1 problem3 problem5 --steps 50 --masking random --probability 0.3

  # Run full cross-validation with LLM-based detection and masking
  python -m src.simulator_cli full --domain blocks --problems problem1 problem3 --masking llm

  # Override domain name in simple mode
  python -m src.simulator_cli simple --problem problem1 --domain-name PDDLEnvBlocks-v1 --masking llm

  # Use custom configuration file
  python -m src.simulator_cli simple --config my_config.yaml --problem problem1 --masking llm
        """
    )

    # Global arguments
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to configuration file (default: config.yaml in project root)'
    )

    parser.add_argument(
        '--domain',
        type=str,
        default='blocks',
        choices=['blocks', 'hanoi'],
        help='Domain to use (default: blocks)'
    )

    parser.add_argument(
        '--experiment-dir',
        type=Path,
        default=None,
        help='Experiment output directory (overrides config)'
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Execution mode')

    # Simple mode: single problem, single run
    simple_parser = subparsers.add_parser(
        'simple',
        help='Run simple pipeline for a single problem'
    )
    simple_parser.add_argument(
        '--problem',
        type=str,
        required=True,
        help='Problem name (e.g., problem1)'
    )
    simple_parser.add_argument(
        '--domain-name',
        type=str,
        default=None,
        help='Override PDDL gym domain name from config (e.g., PDDLEnvBlocks-v0)'
    )
    simple_parser.add_argument(
        '--steps',
        type=int,
        default=None,
        help='Number of steps in trajectory (overrides config)'
    )

    # Full mode: multiple problems with cross-validation
    full_parser = subparsers.add_parser(
        'full',
        help='Run full pipeline with cross-validation'
    )
    full_parser.add_argument(
        '--problems',
        type=str,
        nargs='+',
        required=True,
        help='List of problem names (e.g., problem1 problem3 problem5)'
    )
    full_parser.add_argument(
        '--steps',
        type=int,
        default=None,
        help='Number of steps per trajectory (overrides config)'
    )

    # Masking strategy arguments (common to both modes)
    for subparser in [simple_parser, full_parser]:
        masking_group = subparser.add_argument_group('masking options')
        masking_group.add_argument(
            '--masking',
            type=str,
            default=None,
            choices=['percentage', 'random', 'llm'],
            help='Masking strategy: percentage/random use deterministic detection, llm uses LLM-based detection (overrides config)'
        )
        masking_group.add_argument(
            '--ratio',
            type=float,
            default=None,
            help='Masking ratio for percentage strategy (0.0-1.0)'
        )
        masking_group.add_argument(
            '--probability',
            type=float,
            default=None,
            help='Masking probability for random strategy (0.0-1.0)'
        )

    return parser


def main():
    """Main entry point for the simulator CLI."""

    parser = create_argument_parser()
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration...")
    config = load_config(args.config)
    print(f"✓ Configuration loaded\n")

    # Get domain configuration
    domain_config = config['domains'][args.domain]
    openai_apikey = config['openai']['api_key']

    if openai_apikey == "your-api-key-here":
        raise ValueError(
            "Please set your OpenAI API key in config.yaml\n"
            "Copy config.example.yaml to config.yaml and add your API key."
        )

    # Resolve paths
    project_root = Path(__file__).parent.parent  # Go up from src/ to project root
    pddl_domain_file = project_root / domain_config['domain_file']
    pddl_problem_dir = project_root / domain_config['problems_dir']

    # Determine experiment directory
    if args.experiment_dir:
        experiment_dir = args.experiment_dir
    else:
        experiment_dir = project_root / config['paths']['experiments_dir']

    # Determine number of steps
    num_steps = args.steps if args.steps is not None else config['trajectory']['default_num_steps']

    # Determine masking strategy and parameters
    masking_strategy = args.masking if args.masking is not None else config['masking']['default_strategy']

    masking_kwargs = {}
    if masking_strategy == 'percentage':
        if args.ratio is not None:
            masking_kwargs['masking_ratio'] = args.ratio
        else:
            masking_kwargs['masking_ratio'] = config['masking']['percentage']['default_ratio']
    elif masking_strategy == 'random':
        if args.probability is not None:
            masking_kwargs['masking_probability'] = args.probability
        else:
            masking_kwargs['masking_probability'] = config['masking']['random']['default_probability']
    # Note: For 'llm' masking, no additional parameters needed (masking derived from unknown predicates)

    # Execute based on mode
    start_time = time()

    if args.mode == 'simple':
        # Handle --domain-name override for simple mode
        domain_name = getattr(args, 'domain_name', None) or domain_config['domain_name']

        print(f"Running SIMPLE mode for problem: {args.problem}")
        print(f"Domain name: {domain_name}\n")

        # Create appropriate trajectory handler based on masking strategy
        trajectory_handler = create_trajectory_handler(
            domain=args.domain,
            domain_name=domain_name,
            masking_strategy=masking_strategy,
            openai_apikey=openai_apikey
        )

        simulator = Simulator(
            domain_name=domain_name,
            openai_apikey=openai_apikey,
            pddl_domain_file=pddl_domain_file,
            pddl_problem_dir=pddl_problem_dir,
            experiment_dir_path=experiment_dir,
            trajectory_handler=trajectory_handler
        )

        results = simulator.run_simple_pipeline(
            problem_name=args.problem,
            num_steps=num_steps,
            masking_strategy=masking_strategy,
            **masking_kwargs
        )

        print(f"\n{'='*80}")
        print(f"Learned Domain:")
        print(f"{'='*80}")
        print(results['learnt_domain'].to_pddl())

    elif args.mode == 'full':
        print(f"Running FULL mode with cross-validation\n")

        # Create appropriate trajectory handler based on masking strategy
        trajectory_handler = create_trajectory_handler(
            domain=args.domain,
            domain_name=domain_config['domain_name'],
            masking_strategy=masking_strategy,
            openai_apikey=openai_apikey
        )

        working_dir = run_full_pipeline_with_cross_validation(
            domain=args.domain,
            domain_name=domain_config['domain_name'],
            openai_apikey=openai_apikey,
            pddl_domain_file=pddl_domain_file,
            pddl_problem_dir=pddl_problem_dir,
            experiment_dir=experiment_dir,
            problems=args.problems,
            num_steps=num_steps,
            masking_strategy=masking_strategy,
            trajectory_handler=trajectory_handler,
            **masking_kwargs
        )

    end_time = time()

    print(f"\n{'='*80}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
