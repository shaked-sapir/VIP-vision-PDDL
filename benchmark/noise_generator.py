"""
LLM Noise Generator for Benchmark System

Adds LLM vision noise to ground truth trajectories by:
1. Running LLM fluent classification on images
2. Generating .trajectory and .masking_info files for our algorithms
3. Generating probability-based observations for ROSAME
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.object_detection.llm_blocks_object_detector import LLMBlocksObjectDetector
from benchmark.domains.blocks.equalized_fluent_classifier import EqualizedBlocksFluentClassifier
from src.utils.config import load_config
from src.utils.pddl import build_trajectory_file


def add_llm_noise_to_blocks_data(
    data_dir: Path,
    use_uncertain: bool = True
) -> None:
    """
    Add LLM noise to blocks training data.

    For each trace:
    1. Detects objects from first image
    2. Runs fluent classification on all images
    3. Saves .trajectory and .masking_info for our algorithms (split from ROSAME)
    4. Saves probability-based observations for ROSAME

    Args:
        data_dir: Directory containing blocks training data
        use_uncertain: Whether to allow uncertain (score 1) predictions
    """
    print("="*80)
    print("ADDING LLM NOISE TO BLOCKS DATA")
    print("="*80)
    print()

    # Load configuration
    config = load_config()
    openai_apikey = config['openai']['api_key']
    domain = 'blocks'
    object_detection_model = config['domains'][domain]['object_detection']['model_name']
    object_detection_temp = config['domains'][domain]['object_detection']['temperature']
    fluent_classification_model = config['domains'][domain]['fluent_classification']['model_name']
    fluent_classification_temp = config['domains'][domain]['fluent_classification']['temperature']

    print(f"Configuration:")
    print(f"  Object detection: {object_detection_model}")
    print(f"  Fluent classification: {fluent_classification_model}")
    print(f"  Allow uncertain: {use_uncertain}")
    print()

    # Initialize object detector
    object_detector = LLMBlocksObjectDetector(
        openai_apikey=openai_apikey,
        model=object_detection_model,
        temperature=object_detection_temp
    )

    # Process ROSAME trace FIRST to generate full trajectory and masking files
    rosame_trace_dir = data_dir / "rosame_trace"
    rosame_imaged_trajectory = None

    if rosame_trace_dir.exists():
        print("Processing ROSAME trace (full trajectory)...")
        rosame_imaged_trajectory = _process_trace_for_rosame(
            trace_dir=rosame_trace_dir,
            object_detector=object_detector,
            openai_apikey=openai_apikey,
            fluent_model=fluent_classification_model,
            fluent_temp=fluent_classification_temp,
            use_uncertain=use_uncertain
        )
        print()

    # Process our algorithm traces by splitting the ROSAME trajectory
    our_traces_dir = data_dir / "our_algorithms_traces"
    if our_traces_dir.exists() and rosame_imaged_trajectory:
        print("Processing our algorithm traces (splitting from ROSAME trajectory)...")
        trace_dirs = sorted([d for d in our_traces_dir.iterdir() if d.is_dir()])

        for trace_dir in trace_dirs:
            print(f"  Processing {trace_dir.name}...")
            _process_trace_for_our_algorithms(
                trace_dir=trace_dir,
                rosame_imaged_trajectory=rosame_imaged_trajectory,
                object_detector=object_detector,
                openai_apikey=openai_apikey,
                fluent_model=fluent_classification_model,
                fluent_temp=fluent_classification_temp,
                use_uncertain=use_uncertain
            )
        print()

    print("="*80)
    print("LLM NOISE GENERATION COMPLETE")
    print("="*80)


def _process_trace_for_rosame(
    trace_dir: Path,
    object_detector: LLMBlocksObjectDetector,
    openai_apikey: str,
    fluent_model: str,
    fluent_temp: float,
    use_uncertain: bool
) -> List[dict]:
    """
    Process a trace for ROSAME: generate probability-based observations and full imaged trajectory.

    Probabilities:
    - TRUE (score 2) → 1.0
    - FALSE (score 0) → 0.0
    - UNCERTAIN (score 1) → 0.5

    Returns:
        Full imaged trajectory for splitting into smaller traces
    """
    images_dir = trace_dir / "images"
    if not images_dir.exists():
        print(f"  Warning: No images directory found in {trace_dir}")
        return []

    # Load actions
    actions_file = trace_dir / "actions.json"
    with open(actions_file, 'r') as f:
        actions_data = json.load(f)
    actions = actions_data['actions']
    problem_name = actions_data.get('problem_name', 'unknown').replace('.pddl', '')

    # Detect objects from first image
    first_image = images_dir / "state_0000.png"
    detected_objects = object_detector.detect(str(first_image))

    # Initialize fluent classifier
    fluent_classifier = EqualizedBlocksFluentClassifier(
        openai_apikey=openai_apikey,
        type_to_objects=detected_objects,
        model=fluent_model,
        temperature=fluent_temp,
        use_uncertain=use_uncertain
    )

    print(f"  Detected objects: {detected_objects}")
    print(f"  Classifying {len(actions) + 1} states...")

    # Construct imaged trajectory from images
    imaged_trajectory = _construct_trajectory_from_images(
        images_dir=images_dir,
        actions=actions,
        fluent_classifier=fluent_classifier
    )

    # Save full .trajectory and .masking_info files for ROSAME
    build_trajectory_file(imaged_trajectory, problem_name, trace_dir)
    _save_masking_info_file(imaged_trajectory, problem_name, trace_dir)

    # Also generate probability-based observations for ROSAME
    # Get all image files sorted
    image_files = sorted([f for f in images_dir.iterdir() if f.name.endswith('.png')])
    probability_observations = []

    for img_idx, img_path in enumerate(image_files):
        # Get predicates from imaged trajectory or classify fresh
        if img_idx == 0:
            predicates_with_scores = fluent_classifier.classify(str(img_path))
        else:
            predicates_with_scores = fluent_classifier.classify(str(img_path))

        # Convert scores to probabilities
        predicate_probabilities = {}
        for predicate, score in predicates_with_scores.items():
            if score == 2:  # TRUE
                prob = 1.0
            elif score == 0:  # FALSE
                prob = 0.0
            else:  # UNCERTAIN (score == 1)
                prob = 0.5

            predicate_probabilities[predicate] = prob

        observation = {
            "state_index": img_idx,
            "image": img_path.name,
            "predicates": predicate_probabilities
        }

        if img_idx < len(actions):
            observation["action"] = actions[img_idx]

        probability_observations.append(observation)

    # Save probability observations
    prob_file = trace_dir / "rosame_probability_observations.json"
    with open(prob_file, 'w') as f:
        json.dump({
            "problem_name": problem_name,
            "num_states": len(probability_observations),
            "num_actions": len(actions),
            "observations": probability_observations
        }, f, indent=2)

    print(f"  ✓ Saved {problem_name}.trajectory and {problem_name}.masking_info")
    print(f"  ✓ Saved probability observations: {prob_file.name}")

    return imaged_trajectory


def _process_trace_for_our_algorithms(
    trace_dir: Path,
    rosame_imaged_trajectory: List[dict],
    object_detector: LLMBlocksObjectDetector,
    openai_apikey: str,
    fluent_model: str,
    fluent_temp: float,
    use_uncertain: bool
) -> None:
    """
    Process a trace for our algorithms: split .trajectory and .masking_info from ROSAME trace.

    Args:
        trace_dir: Directory for this specific trace
        rosame_imaged_trajectory: Full imaged trajectory from ROSAME processing
        ... (other args)
    """
    # Load trace metadata to know which states to extract
    metadata_file = trace_dir / "trace_metadata.json"
    if not metadata_file.exists():
        print(f"    Warning: No trace_metadata.json found in {trace_dir}")
        return

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    start_state = metadata['start_state']
    end_state = metadata['end_state']

    # Load actions to get problem name
    actions_file = trace_dir / "actions.json"
    with open(actions_file, 'r') as f:
        actions_data = json.load(f)
    problem_name = actions_data.get('problem_name', 'unknown').replace('.pddl', '')

    # Extract the relevant part of the trajectory
    # The imaged_trajectory has steps numbered 1 to N
    # We need to extract steps that correspond to actions at indices start_state to end_state-1
    trace_trajectory = rosame_imaged_trajectory[start_state:end_state]

    if not trace_trajectory:
        print(f"    Warning: No trajectory data for states {start_state}-{end_state}")
        return

    # Save trajectory file
    build_trajectory_file(trace_trajectory, problem_name, trace_dir)

    # Save masking info file
    _save_masking_info_file(trace_trajectory, problem_name, trace_dir)

    print(f"    ✓ Split .trajectory (states {start_state}-{end_state}) and .masking_info files")


def _save_masking_info_file(
    imaged_trajectory: List[dict],
    problem_name: str,
    trace_dir: Path
) -> None:
    """
    Save .masking_info file from imaged trajectory.

    Format: One line per state, with comma-separated unknown predicates.
    """
    from src.utils.pddl import parse_gym_to_pddl_literal

    masking_info_path = trace_dir / f"{problem_name}.masking_info"

    with open(masking_info_path, 'w') as f:
        # Write masking for initial state (first current_state)
        if imaged_trajectory:
            first_unknown = imaged_trajectory[0]['current_state'].get('unknown', [])
            unknown_pddl = [parse_gym_to_pddl_literal(pred) for pred in first_unknown]
            f.write(', '.join(unknown_pddl) + '\n')

            # Write masking for each next state
            for step in imaged_trajectory:
                next_unknown = step['next_state'].get('unknown', [])
                unknown_pddl = [parse_gym_to_pddl_literal(pred) for pred in next_unknown]
                f.write(', '.join(unknown_pddl) + '\n')


def _construct_trajectory_from_images(
    images_dir: Path,
    actions: List[str],
    fluent_classifier: EqualizedBlocksFluentClassifier
) -> List[dict]:
    """
    Construct imaged trajectory from images (similar to ImageTrajectoryHandler.construct_trajectory_from_images).
    """
    from src.action_model.pddl2gym_parser import (
        parse_image_predicate_to_gym,
        is_positive_gym_predicate,
        is_unknown_gym_predicate
    )

    imaged_trajectory = []

    # Get sorted image files
    image_files = sorted([f for f in images_dir.iterdir() if f.name.endswith('.png')])

    # Process first state separately
    first_image_path = str(image_files[0])
    current_state_predicates = fluent_classifier.classify(first_image_path)

    # Process each transition
    for i, action in enumerate(actions):
        # Load next image and classify predicates
        next_image_path = str(image_files[i + 1])
        next_state_predicates = fluent_classifier.classify(next_image_path)

        # Convert predicates to PDDL format for trajectory
        current_literals = [parse_image_predicate_to_gym(pred, truth_value)
                            for pred, truth_value in current_state_predicates.items()]
        next_literals = [parse_image_predicate_to_gym(pred, truth_value)
                         for pred, truth_value in next_state_predicates.items()]

        # Build trajectory step
        trajectory_step = {
            "step": i + 1,
            "current_state": {
                "literals": [pred for pred in current_literals if is_positive_gym_predicate(pred)],
                "unknown": [pred for pred in current_literals if is_unknown_gym_predicate(pred)]
            },
            "ground_action": action,
            "next_state": {
                "literals": [pred for pred in next_literals if is_positive_gym_predicate(pred)],
                "unknown": [pred for pred in next_literals if is_unknown_gym_predicate(pred)]
            }
        }

        imaged_trajectory.append(trajectory_step)

        # Optimization: reuse next_state as current_state for next iteration
        current_state_predicates = next_state_predicates

    return imaged_trajectory


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Add LLM noise to benchmark training data"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="blocks",
        choices=["blocks", "hanoi", "slidetile"],
        help="Domain to process (default: blocks)"
    )
    parser.add_argument(
        "--no-uncertain",
        action="store_true",
        help="Disable uncertain predictions (only allow 0 or 2)"
    )

    args = parser.parse_args()

    data_dir = Path(__file__).parent / "data" / args.domain / "training"

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Please run data_generator.py first to generate training data.")
        sys.exit(1)

    if args.domain == "blocks":
        add_llm_noise_to_blocks_data(
            data_dir=data_dir,
            use_uncertain=not args.no_uncertain
        )
    else:
        print(f"Domain '{args.domain}' not yet implemented")
