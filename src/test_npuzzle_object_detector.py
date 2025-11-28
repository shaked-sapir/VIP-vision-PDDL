"""
Test script for N-Puzzle (Slidetile) LLM Object Detector and Fluent Classifier.

This script:
1. Generates trajectory images using LLMNpuzzleImageTrajectoryHandler
2. Records observations from the environment (ground truth)
3. Detects objects from the first image using LLMNpuzzleObjectDetector
4. Runs fluent classification on all images using LLMNpuzzleFluentClassifier
5. Saves observations and results to JSON files
"""

import argparse
import json
import sys
import shutil
from collections import defaultdict
from pathlib import Path

from src.object_detection.llm_npuzzle_object_detector import LLMNpuzzleObjectDetector
from src.trajectory_handlers.llm_npuzzle_trajectory_handler import LLMNpuzzleImageTrajectoryHandler
from src.utils.config import load_config


def test_npuzzle_full_pipeline(num_steps: int = 10, problem_name: str = "eight01x.pddl", verbose: bool = False):
    """
    Complete test of N-Puzzle vision pipeline:
    1. Generate trajectory using LLMNpuzzleImageTrajectoryHandler
    2. Load ground truth observations
    3. Load LLM classifications (from imaged trajectory)
    4. Display and compare results
    """

    # Load configuration
    config = load_config()
    openai_apikey = config['openai']['api_key']
    domain = 'n_puzzle'
    gym_domain_name = config['domains'][domain]['gym_domain_name']
    object_detection_model = config['domains'][domain]['object_detection']['model_name']
    object_detection_temp = config['domains'][domain]['object_detection']['temperature']
    fluent_classification_model = config['domains'][domain]['fluent_classification']['model_name']
    fluent_classification_temp = config['domains'][domain]['fluent_classification']['temperature']

    if openai_apikey == "your-api-key-here":
        print("❌ Error: Please set your OpenAI API key in config.yaml")
        sys.exit(1)

    print("="*80)
    print("N-PUZZLE COMPLETE VISION PIPELINE TEST")
    print("="*80)
    print(f"Gym Environment: {gym_domain_name}")
    print(f"Problem: {problem_name}")
    print(f"Object Detection Model: {object_detection_model}")
    print(f"Fluent Classification Model: {fluent_classification_model}")
    print()

    # Setup output directory
    output_dir = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/domains/n_puzzle/test_trajectory")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # ========== STEP 1: Generate Trajectory with Gym ==========
    print("="*80)
    print("STEP 1: GENERATING TRAJECTORY WITH PDDLGYM")
    print("="*80)
    print()

    trajectory_handler = LLMNpuzzleImageTrajectoryHandler(
        domain_name=gym_domain_name,
        openai_apikey=openai_apikey,
        object_detector_model=object_detection_model,
        object_detection_temperature=object_detection_temp,
        fluent_classifier_model=fluent_classification_model,
        fluent_classification_temperature=fluent_classification_temp
    )

    # Generate trajectory and save images + ground truth
    print(f"Creating trajectory from gym environment...")
    ground_actions = trajectory_handler.create_trajectory_from_gym(
        problem_name=problem_name,
        images_output_path=images_dir,
        num_steps=num_steps
    )

    print(f"\n✓ Generated {len(ground_actions)} trajectory steps")
    print(f"  Ground actions: {ground_actions[:3]}..." if len(ground_actions) > 3 else f"  Ground actions: {ground_actions}")
    print()

    # ========== STEP 2: Object Detection & Fluent Classification ==========
    print("="*80)
    print("STEP 2: RUNNING LLM VISION PIPELINE")
    print("="*80)
    print()

    # Initialize visual components (object detector + fluent classifier)
    init_image_path = images_dir / "state_0000.png"

    # Run image trajectory pipeline (classifies all images)
    print(f"Initializing visual components from: {init_image_path.name}")
    trajectory_handler.init_visual_components(init_image_path)
    print()
    print(f"Running fluent classification on all {len(ground_actions) + 1} images...")
    imaged_trajectory = trajectory_handler.image_trajectory_pipeline(
        problem_name=problem_name,
        actions=ground_actions,
        images_path=images_dir
    )

    print(f"✓ Completed LLM classification for {len(imaged_trajectory)} trajectory steps")
    print()

    # ========== STEP 3: Load and Save Results ==========
    print("="*80)
    print("STEP 3: PROCESSING RESULTS")
    print("="*80)
    print()

    # Load ground truth trajectory
    gt_trajectory_file = images_dir / f"{problem_name}_trajectory.json"
    with open(gt_trajectory_file, 'r') as f:
        gt_trajectory = json.load(f)

    print(f"✓ Loaded ground truth trajectory: {gt_trajectory_file.name}")

    # Save imaged trajectory with confidence scores
    imaged_trajectory_file = output_dir / "llm_imaged_trajectory.json"
    with open(imaged_trajectory_file, 'w') as f:
        json.dump(imaged_trajectory, f, indent=2)

    print(f"✓ Saved LLM imaged trajectory: {imaged_trajectory_file.name}")

    # Create summary comparison
    summary = {
        "problem": problem_name,
        "num_steps": len(ground_actions),
        "ground_truth_trajectory": gt_trajectory_file.name,
        "llm_trajectory": imaged_trajectory_file.name,
        "detected_objects": trajectory_handler.object_detector.detect(init_image_path) if hasattr(trajectory_handler, 'object_detector') else None,
        "summary_by_step": []
    }

    # Compare GT vs LLM for each step
    for i, step in enumerate(imaged_trajectory):
        gt_step = gt_trajectory[i] if i < len(gt_trajectory) else None

        current_literals = step.get('current_state', {}).get('literals', [])
        unknown_literals = step.get('current_state', {}).get('unknown', [])

        summary["summary_by_step"].append({
            "step": i + 1,
            "action": step.get('ground_action'),
            "gt_action": gt_step['ground_action'] if gt_step else None,
            "llm_certain_predicates": len(current_literals),
            "llm_unknown_predicates": len(unknown_literals),
            "gt_total_predicates": len(gt_step['current_state']['literals']) if gt_step else None
        })

    # Save summary
    summary_file = output_dir / "comparison_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Saved comparison summary: {summary_file.name}")
    print()

    # ========== STEP 4: Display Summary ==========
    print("="*80)
    print("SUMMARY - Sample Results (Step 1)")
    print("="*80)

    if imaged_trajectory:
        step = imaged_trajectory[0]
        current_state = step.get('current_state', {})
        certain_preds = current_state.get('literals', [])
        unknown_preds = current_state.get('unknown', [])

        print(f"\nAction: {step.get('ground_action')}")
        print(f"LLM Predicates:")
        print(f"  Certain: {len(certain_preds)}")
        print(f"  Unknown: {len(unknown_preds)}")

        if certain_preds:
            print(f"\nSample certain predicates (first 10):")
            for pred in certain_preds[:10]:
                print(f"    {pred}")

        if unknown_preds:
            print(f"\nSample unknown predicates (first 5):")
            for pred in unknown_preds[:5]:
                print(f"    {pred}")

    print()
    print("="*80)
    print("TEST COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print(f"  - Images: {images_dir}")
    print(f"  - Ground Truth: {gt_trajectory_file.name}")
    print(f"  - LLM Trajectory: {imaged_trajectory_file.name}")
    print(f"  - Comparison: {summary_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test N-Puzzle LLM Vision Pipeline using LLMNpuzzleImageTrajectoryHandler"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output including step details"
    )
    parser.add_argument(
        "--num-steps", "-n",
        type=int,
        default=10,
        help="Number of trajectory steps to generate (default: 10)"
    )
    parser.add_argument(
        "--problem", "-p",
        type=str,
        default="eight01x.pddl",
        help="Problem name to use (default: npuzzle_prob_0)"
    )

    args = parser.parse_args()
    test_npuzzle_full_pipeline(
        num_steps=args.num_steps,
        problem_name=args.problem,
        verbose=args.verbose
    )
