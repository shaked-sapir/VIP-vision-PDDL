"""
Test script for extract_objects_from_gt_trajectory function.

This demonstrates how to extract and translate objects from a ground truth trajectory.
"""

from pathlib import Path

# Example: Using blocksworld object detector
from src.object_detection.llm_blocks_object_detector import LLMBlocksObjectDetector


def test_extract_objects():
    """Test the object extraction function with blocksworld example."""

    # Example trajectory file
    trajectory_file = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/data/blocksworld/experiment_02-12-2025T11:45:40__steps=500/training/rosame_trace/problem7/problem7_trajectory.json")
    init_image = trajectory_file.parent / "state_0000.png"

    if not trajectory_file.exists():
        print(f"Trajectory file not found: {trajectory_file}")
        print("Please update the path to an existing trajectory file.")
        return

    print("="*80)
    print("TESTING OBJECT EXTRACTION FROM GT TRAJECTORY")
    print("="*80)
    print()

    # Create object detector instance
    detector = LLMBlocksObjectDetector(
        openai_apikey="dummy_key",  # Not needed for extraction
        model="gpt-4o",
        temperature=1.0,
        init_state_image_path=init_image
    )

    # Set up the object mapping (normally done during detection)
    # Example mapping for blocksworld
    example_mapping = {
        "red_block": "a",
        "blue_block": "b",
        "green_block": "c",
        "yellow_block": "d",
        "purple_block": "e",
        "orange_block": "f",
        "robot": "robot"
    }

    detector.imaged_obj_to_gym_obj_name = example_mapping

    print("Object mapping (image -> gym):")
    for image_obj, gym_obj in example_mapping.items():
        print(f"  {image_obj} -> {gym_obj}")
    print()

    # Extract objects from first state (state_index=0)
    print("Extracting objects from first state (index 0)...")
    translated_objects = detector.extract_objects_from_gt_state(
        gt_trajectory_path=trajectory_file,
        state_index=0
    )

    print()
    print("Extracted objects (with image object names):")
    for obj in translated_objects:
        print(f"  {obj}")
    print()

    print("="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print()
    print("Summary:")
    print(f"  Total objects extracted: {len(translated_objects)}")
    print()

    # Show breakdown by type
    from collections import defaultdict
    type_counts = defaultdict(int)
    for obj in translated_objects:
        if ':' in obj:
            _, obj_type = obj.split(':', 1)
            type_counts[obj_type] += 1

    print("Objects by type:")
    for obj_type, count in type_counts.items():
        print(f"  {obj_type}: {count}")
    print()


if __name__ == "__main__":
    test_extract_objects()
