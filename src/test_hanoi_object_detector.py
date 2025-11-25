"""
Test script for Hanoi LLM Object Detector and Fluent Classifier.

This script tests:
1. LLM-based object detection (discs and pegs)
2. LLM-based fluent classification (predicates in the image)
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from src.object_detection.llm_hanoi_object_detector import LLMHanoiObjectDetector
from src.fluent_classification.llm_hanoi_fluent_classifier import LLMHanoiFluentClassifier
from src.utils.config import load_config


def test_hanoi_detector_and_classifier(verbose: bool = False, num_images: int = 5, test_fluents: bool = True):
    """Test the Hanoi object detector and fluent classifier on experiment images."""

    # Load configuration
    config = load_config()
    openai_apikey = config['openai']['api_key']
    model_name = config['openai']['visual_components_model']['model_name']

    if openai_apikey == "your-api-key-here":
        print("❌ Error: Please set your OpenAI API key in config.yaml")
        sys.exit(1)

    # Initialize components
    print("="*80)
    print("HANOI OBJECT DETECTOR & FLUENT CLASSIFIER TEST")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Test fluent classification: {test_fluents}")
    print()

    detector = LLMHanoiObjectDetector(
        openai_apikey=openai_apikey,
        model=model_name
    )

    # Find experiment images
    experiments_dir = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/domains/hanoi/experiments/num_steps=10/PDDLEnvHanoi_operator_actions-v0_problem0_temp")

    if not experiments_dir.exists():
        print(f"❌ Error: Experiment directory not found: {experiments_dir}")
        sys.exit(1)

    # Get all state images (test on first N states for variety)
    image_files = sorted(experiments_dir.glob("state_*.png"))[:num_images]

    if not image_files:
        print(f"❌ Error: No images found in {experiments_dir}")
        sys.exit(1)

    print(f"Found {len(image_files)} images to test")
    print()

    # Test each image
    for i, image_path in enumerate(image_files, 1):
        print("="*80)
        print(f"IMAGE {i}/{len(image_files)}: {image_path.name}")
        print("="*80)
        print(f"Path: {image_path}")
        print()

        try:
            # ========== STEP 1: Object Detection ==========
            if verbose:
                print("STEP 1: Running object detection...")
                print()

            detected_objects = detector.detect(image_path)

            # Display detected objects
            print("✓ Object Detection Complete!")
            print()
            print("Detected Objects:")
            print("-" * 40)

            # Extract and sort object names alphabetically
            all_disc_names = sorted(detected_objects.get('disc', []))
            all_peg_names = sorted(detected_objects.get('peg', []))

            # Create sorted type_to_objects for fluent classifier
            sorted_type_to_objects = {
                'disc': all_disc_names,
                'peg': all_peg_names
            }

            # Display by type
            if all_disc_names:
                print(f"  Discs ({len(all_disc_names)}):")
                for disc in all_disc_names:
                    print(f"    - {disc}")
            else:
                print("  Discs: None detected")

            print()

            if all_peg_names:
                print(f"  Pegs ({len(all_peg_names)}):")
                for peg in all_peg_names:
                    print(f"    - {peg}")
            else:
                print("  Pegs: None detected")

            # Validation
            total_objects = len(all_disc_names) + len(all_peg_names)
            print()
            print(f"Total objects detected: {total_objects}")

            expected_discs = 3
            expected_pegs = 3

            print()
            print("Validation:")
            disc_status = "✓" if len(all_disc_names) == expected_discs else "✗"
            peg_status = "✓" if len(all_peg_names) == expected_pegs else "✗"

            print(f"  {disc_status} Discs: {len(all_disc_names)}/{expected_discs} expected")
            print(f"  {peg_status} Pegs: {len(all_peg_names)}/{expected_pegs} expected")

            # ========== STEP 2: Fluent Classification ==========
            if test_fluents:
                print()
                print("="*80)
                if verbose:
                    print("STEP 2: Running fluent classification...")
                    print()

                # Initialize classifier with detected objects
                # This ensures the prompt is generated with the correct object names
                if verbose:
                    print(f"Initializing classifier with sorted objects:")
                    print(f"  Discs: {all_disc_names}")
                    print(f"  Pegs: {all_peg_names}")
                    print()

                classifier = LLMHanoiFluentClassifier(
                    openai_apikey=openai_apikey,
                    type_to_objects=sorted_type_to_objects,
                    model=model_name
                )

                # Run classification
                fluents_with_confidence = classifier.classify(str(image_path))

                # Display results
                print("✓ Fluent Classification Complete!")
                print()
                print("Detected Predicates (with confidence):")
                print("-" * 40)

                # Group predicates by type
                predicates_by_type = defaultdict(list)
                for fluent, confidence in fluents_with_confidence.items():
                    # Extract predicate type from fluent
                    pred_type = fluent.split('(')[0]
                    predicates_by_type[pred_type].append((fluent, confidence))

                # Display by predicate type
                for pred_type in sorted(predicates_by_type.keys()):
                    preds = predicates_by_type[pred_type]
                    print(f"\n  {pred_type} predicates ({len(preds)}):")

                    # Sort by confidence (descending) then alphabetically
                    sorted_preds = sorted(preds, key=lambda x: (-x[1], x[0]))

                    for fluent, confidence in sorted_preds:
                        # Show confidence level with visual indicator
                        if confidence == 2:
                            conf_str = "✓✓ (certain)"
                        elif confidence == 1:
                            conf_str = "?  (uncertain)"
                        else:
                            conf_str = "✗✗ (false)"

                        print(f"    {conf_str} {fluent}")

                # Summary
                total_predicates = len(fluents_with_confidence)
                certain_count = sum(1 for c in fluents_with_confidence.values() if c == 2)
                uncertain_count = sum(1 for c in fluents_with_confidence.values() if c == 1)
                false_count = sum(1 for c in fluents_with_confidence.values() if c == 0)

                print()
                print(f"Total predicates extracted: {total_predicates}")
                print(f"  Certain (confidence=2): {certain_count}")
                print(f"  Uncertain (confidence=1): {uncertain_count}")
                print(f"  False (confidence=0): {false_count}")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

        print()

    print("="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Hanoi LLM Object Detector and Fluent Classifier on experiment images"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output including step details"
    )
    parser.add_argument(
        "--num-images", "-n",
        type=int,
        default=5,
        help="Number of images to test (default: 5)"
    )
    parser.add_argument(
        "--no-fluents",
        action="store_true",
        help="Skip fluent classification (only test object detection)"
    )

    args = parser.parse_args()
    test_hanoi_detector_and_classifier(
        verbose=args.verbose,
        num_images=args.num_images,
        test_fluents=not args.no_fluents
    )
