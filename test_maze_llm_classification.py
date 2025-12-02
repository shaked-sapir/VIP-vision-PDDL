"""
Test LLM Hiking Object Detection and Fluent Classification

This script:
1. Runs LLMHikingObjectDetector on the initial image
2. Runs LLMHikingFluentClassifier on images 0000-0009
3. Saves results to a JSON file
"""

import json
from datetime import datetime
from pathlib import Path

from src.fluent_classification.base_fluent_classifier import PredicateTruthValue
from src.fluent_classification.llm_maze_fluent_classifier import LLMMazeFluentClassifier
from src.object_detection.llm_maze_object_detector import LLMMazeObjectDetector
from src.trajectory_handlers.llm_maze_trajectory_handler import LLMMazeImageTrajectoryHandler
from src.utils.config import load_config


def test_maze_llm_classification(num_images=None):
    """
    Test hiking LLM object detection and fluent classification.

    Args:
        num_images: Number of images to classify (default: all available images)
    """
    print("="*80)
    print("TESTING MAZE LLM CLASSIFICATION")
    print("="*80)
    print()

    # Load configuration
    config = load_config()
    gym_domain_name = config['domains']['maze']['gym_domain_name']
    domain_path = Path(config['domains']['maze']['domain_file'])
    openai_apikey = config['openai']['api_key']
    object_detection_model = config['domains']['maze']['object_detection']['model_name']
    object_detection_temp = config['domains']['maze']['object_detection']['temperature']
    fluent_classification_model = config['domains']['maze']['fluent_classification']['model_name']
    fluent_classification_temp = config['domains']['maze']['fluent_classification']['temperature']

    # Setup paths
    images_dir = Path(__file__).parent / "maze_test_sequence"
    output_dir = Path(__file__).parent / "maze_llm_test_results"
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

    trajectory_handler = LLMMazeImageTrajectoryHandler(
        domain_name=gym_domain_name,
        pddl_domain_file=domain_path,
        openai_apikey=openai_apikey,
        object_detector_model=object_detection_model,
        object_detection_temperature=object_detection_temp,
        fluent_classifier_model=fluent_classification_model,
        fluent_classification_temperature=fluent_classification_temp
    )
    trajectory_handler.init_visual_components(initial_image)

    # object_detector = LLMMazeObjectDetector(
    #     openai_apikey=openai_apikey,
    #     model=object_detection_model,
    #     temperature=object_detection_temp
    # )

    # detected_objects = object_detector.detect(str(initial_image))

    detected_objects = trajectory_handler.object_detector.detect(str(initial_image))
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

    # fluent_classifier = LLMMazeFluentClassifier(
    #     openai_apikey=openai_apikey,
    #     type_to_objects=detected_objects,
    #     model=fluent_classification_model,
    #     temperature=fluent_classification_temp,
    #     const_predicates=set()
    # )
    fluent_classifier = trajectory_handler.fluent_classifier

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

    output_file = output_dir / f"maze_llm_test_{timestamp}.json"
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
        description="Test maze LLM object detection and fluent classification"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Number of images to classify (default: all available)"
    )

    args = parser.parse_args()

    test_maze_llm_classification(num_images=args.num_images)
