"""
Test LLM Hiking Object Detection and Fluent Classification

This script:
1. Runs LLMHikingObjectDetector on the initial image
2. Runs LLMHikingFluentClassifier on images 0000-0009
3. Saves results to a JSON file
"""

import json
from pathlib import Path
from datetime import datetime

from src.object_detection.llm_hiking_object_detector import LLMHikingObjectDetector
from src.fluent_classification.llm_hiking_fluent_classifier import LLMHikingFluentClassifier
from src.fluent_classification.base_fluent_classifier import PredicateTruthValue
from src.utils.config import load_config


def test_hiking_llm_classification(num_images=None):
    """
    Test hiking LLM object detection and fluent classification.

    Args:
        num_images: Number of images to classify (default: all available images)
    """
    print("="*80)
    print("TESTING HIKING LLM CLASSIFICATION")
    print("="*80)
    print()

    # Load configuration
    config = load_config()
    openai_apikey = config['openai']['api_key']
    object_detection_model = config['domains']['hiking']['object_detection']['model_name']
    object_detection_temp = config['domains']['hiking']['object_detection']['temperature']
    fluent_classification_model = config['domains']['hiking']['fluent_classification']['model_name']
    fluent_classification_temp = config['domains']['hiking']['fluent_classification']['temperature']

    # Setup paths
    images_dir = Path(__file__).parent / "hiking_test_sequence"
    output_dir = Path(__file__).parent / "hiking_llm_test_results"
    output_dir.mkdir(exist_ok=True)

    # Count available images
    available_images = sorted(images_dir.glob("state_*.png"))
    max_images = len(available_images)

    if num_images is None:
        num_images = max_images
    else:
        num_images = min(num_images, max_images)

    print(f"Images directory: {images_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Available images: {max_images}")
    print(f"Images to classify: {num_images}")
    print()

    # Step 1: Run object detection on initial image
    print("="*80)
    print("STEP 1: OBJECT DETECTION")
    print("="*80)
    initial_image = images_dir / "state_0000.png"
    print(f"Running object detection on: {initial_image}")
    print()

    object_detector = LLMHikingObjectDetector(
        openai_apikey=openai_apikey,
        model=object_detection_model,
        temperature=object_detection_temp
    )

    # detected_objects = object_detector.detect(str(initial_image))
    detected_objects = {'loc': ['r6_c5', 'r1_c8', 'r5_c7', 'r0_c1', 'r5_c3', 'r4_c4', 'r1_c3', 'r5_c8', 'r4_c7', 'r3_c6', 'r2_c7', 'r2_c0', 'r4_c2', 'r6_c1', 'r4_c3', 'r0_c5', 'r7_c5', 'r7_c0', 'r5_c2', 'r6_c7', 'r7_c1', 'r3_c8', 'r2_c4', 'r0_c2', 'r0_c3', 'r4_c8', 'r3_c1', 'r0_c6', 'r5_c6', 'r7_c8', 'r3_c4', 'r3_c9', 'r5_c4', 'r0_c8', 'r6_c0', 'r2_c9', 'r2_c1', 'r1_c4', 'r7_c2', 'r0_c9', 'r4_c5', 'r2_c5', 'r7_c7', 'r5_c0', 'r1_c7', 'r7_c6', 'r7_c9', 'r6_c9', 'r7_c4', 'r1_c6', 'r4_c9', 'r0_c7', 'r5_c1', 'r2_c2', 'r2_c3', 'r4_c0', 'r5_c9', 'r3_c5', 'r7_c3', 'r3_c7', 'r6_c8', 'r1_c2', 'r6_c4', 'r4_c6', 'r6_c3', 'r1_c1', 'r3_c2', 'r1_c0', 'r2_c6', 'r1_c9', 'r3_c3', 'r4_c1', 'r6_c6', 'r1_c5', 'r0_c4', 'r6_c2', 'r2_c8', 'r3_c0', 'r5_c5', 'r0_c0']}

    print("Detected objects by type:")
    for obj_type, obj_list in detected_objects.items():
        print(f"  {obj_type}: {obj_list}")
    print()

    # Step 2: Run fluent classification on first N images
    print("="*80)
    print("STEP 2: FLUENT CLASSIFICATION")
    print("="*80)
    print(f"Running fluent classification on {num_images} images...")
    print()

    fluent_classifier = LLMHikingFluentClassifier(
        openai_apikey=openai_apikey,
        type_to_objects=detected_objects,
        model=fluent_classification_model,
        temperature=fluent_classification_temp
    )

    classification_results = []

    for img_idx in range(num_images):
        image_path = images_dir / f"state_{img_idx:04d}.png"
        print(f"  Classifying image {img_idx}: {image_path.name}...")

        # classify returns Dict[str, PredicateTruthValue]
        predicate_classifications = fluent_classifier.classify(str(image_path))

        # Separate predicates by truth value
        positive_fluents = [pred for pred, val in predicate_classifications.items() if val == PredicateTruthValue.TRUE]
        negative_fluents = [pred for pred, val in predicate_classifications.items() if val == PredicateTruthValue.FALSE]
        unknown_fluents = [pred for pred, val in predicate_classifications.items() if val == PredicateTruthValue.UNCERTAIN]

        # Convert to serializable format
        state_result = {
            "image_index": img_idx,
            "image_name": image_path.name,
            "positive_fluents": positive_fluents,
            "negative_fluents": negative_fluents,
            "unknown_fluents": unknown_fluents
        }

        classification_results.append(state_result)

        print(f"    Positive: {len(positive_fluents)} fluents")
        print(f"    Negative: {len(negative_fluents)} fluents")
        print(f"    Unknown: {len(unknown_fluents)} fluents")

    print()
    print("✓ Fluent classification complete")
    print()

    # Step 3: Save results to JSON
    print("="*80)
    print("SAVING RESULTS")
    print("="*80)

    timestamp = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
    results = {
        "timestamp": timestamp,
        "num_images": num_images,
        "object_detection": {
            "model": object_detection_model,
            "temperature": object_detection_temp,
            "detected_objects": detected_objects
        },
        "fluent_classification": {
            "model": fluent_classification_model,
            "temperature": fluent_classification_temp,
            "results": classification_results
        }
    }

    output_file = output_dir / f"hiking_llm_test_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    # Print summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Object types detected: {len(detected_objects)}")
    for obj_type, obj_list in detected_objects.items():
        print(f"  {obj_type}: {len(obj_list)} objects")
    print()
    print(f"Images classified: {num_images}")
    print(f"Results file: {output_file.name}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test hiking LLM object detection and fluent classification"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Number of images to classify (default: all available)"
    )

    args = parser.parse_args()

    test_hiking_llm_classification(num_images=args.num_images)
