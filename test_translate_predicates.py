"""
Test script for translate_pddlgym_state_to_image_predicates function.

This demonstrates how to translate predicates from PDDLGym format to image object format.
"""

import json
from pathlib import Path

# Example: Using blocksworld fluent classifier
from src.fluent_classification.llm_blocks_fluent_classifier import LLMBlocksFluentClassifier


def test_translate_predicates():
    """Test the translation function with blocksworld example."""

    # Example trajectory file
    trajectory_file = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/data/blocksworld/experiment_02-12-2025T11:45:40__steps=500/training/rosame_trace/problem7/problem7_trajectory.json")

    if not trajectory_file.exists():
        print(f"Trajectory file not found: {trajectory_file}")
        print("Please update the path to an existing trajectory file.")
        return

    # Load trajectory
    with open(trajectory_file, 'r') as f:
        trajectory_data = json.load(f)

    # Get first state
    first_step = trajectory_data[0]
    current_state_literals = first_step['current_state']['literals']

    print("="*80)
    print("TESTING PREDICATE TRANSLATION")
    print("="*80)
    print()

    print("Original PDDLGym state literals:")
    for lit in current_state_literals:
        print(f"  {lit}")
    print()

    # Create a mock fluent classifier (we don't need full initialization for translation)
    # We just need the imaged_obj_to_gym_obj_name mapping

    # Example mapping for blocksworld
    # In a real scenario, this would be populated by object detection
    example_mapping = {
        "red_block": "a",
        "blue_block": "b",
        "green_block": "c",
        "yellow_block": "d",
        "purple_block": "e",
        "orange_block": "f",
        "robot": "robot"
    }

    # Create classifier instance with mapping
    classifier = LLMBlocksFluentClassifier(
        openai_apikey="dummy_key",  # Not needed for translation
        type_to_objects={"block": ["red_block", "blue_block", "green_block", "yellow_block", "purple_block", "orange_block"]},
        model="gpt-4o",
        temperature=1.0
    )

    # Override the mapping (normally set by object detection)
    classifier.imaged_obj_to_gym_obj_name = example_mapping

    # Translate predicates
    print("Translating predicates...")
    translated_predicates = classifier.extract_predicates_from_gt_state(current_state_literals)

    print()
    print("Translated predicates (with image object names):")
    for pred in translated_predicates:
        print(f"  {pred}")
    print()

    print("="*80)
    print("TRANSLATION COMPLETE")
    print("="*80)
    print()
    print("Summary:")
    print(f"  Original predicates: {len(current_state_literals)}")
    print(f"  Translated predicates: {len(translated_predicates)}")
    print()

    # Show some mappings
    print("Object mapping used:")
    for image_obj, gym_obj in example_mapping.items():
        print(f"  {gym_obj} -> {image_obj}")
    print()


if __name__ == "__main__":
    test_translate_predicates()
