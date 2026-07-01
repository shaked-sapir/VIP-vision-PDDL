"""
Quick smoke test for the Gripper LLM object detector and fluent classifier.

Usage:
    python tests/test_gripper_llm_components.py <images_dir>

Where <images_dir> contains:
    - state_0.png (at minimum)
    - A *_trajectory.json file (GT trajectory for few-shot examples)
      OR a .trajectory + .pddl file (auto-converted)
    - More state_N.png files to classify

Examples:
    # Full pipeline — detect objects, then classify all images:
    python tests/test_gripper_llm_components.py benchmark/data/gripper/problem0/

    # Object detection only:
    python tests/test_gripper_llm_components.py benchmark/data/gripper/problem0/ --skip-classification

    # Classification only (uses manual object list):
    python tests/test_gripper_llm_components.py benchmark/data/gripper/problem0/ --skip-detection

    # Classify a single image instead of all:
    python tests/test_gripper_llm_components.py benchmark/data/gripper/problem0/ --classify-index 3

    # Use Gemini:
    python tests/test_gripper_llm_components.py benchmark/data/gripper/problem0/ --vendor google

    # Hide FALSE predicates (show count only):
    python tests/test_gripper_llm_components.py benchmark/data/gripper/problem0/ --hide-false
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.trajectory_json_converter import convert_trajectory_to_json
from src.utils.config import load_config


def _create_backend(vendor: str, model_type: str):
    """Create an LLM backend without triggering the circular import in ImageLLMBackendFactory."""
    config = load_config()
    api_key = config[vendor]["api_key"]
    model_config_key = f"{model_type}_model"
    model = config[vendor][model_config_key]["model_name"]
    temperature = config[vendor][model_config_key]["temperature"]

    if vendor == "google":
        from src.fluent_classification.gemini_image_llm_backend import GeminiImageLLMBackend
        return GeminiImageLLMBackend(api_key=api_key, model=model, temperature=temperature)
    else:
        from src.fluent_classification.openai_image_llm_backend import OpenAIImageLLMBackend
        return OpenAIImageLLMBackend(api_key=api_key, model=model, temperature=temperature)


def _find_all_state_images(images_dir: Path) -> list[Path]:
    """Find all state_*.png images, sorted numerically."""
    images = list(images_dir.glob("state_*.png"))

    def sort_key(p: Path) -> int:
        stem = p.stem  # e.g. "state_0", "state_12"
        try:
            return int(stem.split("_", 1)[1])
        except (IndexError, ValueError):
            return 0
    return sorted(images, key=sort_key)


def ensure_trajectory_json(images_dir: Path) -> None:
    """Convert .trajectory + .pddl → trajectory.json if needed."""
    if list(images_dir.glob("*_trajectory.json")):
        return

    traj_files = list(images_dir.glob("*.trajectory"))
    pddl_files = list(images_dir.glob("*.pddl"))

    if traj_files and pddl_files:
        print(f"Converting {traj_files[0].name} → trajectory.json ...")
        convert_trajectory_to_json(traj_files[0], pddl_files[0])
    else:
        print("ERROR: No *_trajectory.json found, and no .trajectory + .pddl to convert.")
        sys.exit(1)


def test_object_detection(images_dir: Path, vendor: str = "openai") -> dict[str, list[str]]:
    """Test the gripper object detector on the first image."""
    init_image = images_dir / "state_0.png"
    if not init_image.exists():
        pngs = _find_all_state_images(images_dir)
        if not pngs:
            print("ERROR: No state_*.png images found in", images_dir)
            sys.exit(1)
        init_image = pngs[0]

    print(f"\n{'='*60}")
    print(f"OBJECT DETECTION — {init_image.name}")
    print(f"{'='*60}")

    from src.object_detection.llm_gripper_object_detector import LLMGripperObjectDetector

    backend = _create_backend(vendor=vendor, model_type="object_detection")

    detector = LLMGripperObjectDetector(
        llm_backend=backend,
        init_state_image_path=init_image,
        inference_mode=True,
    )

    detected = detector.detect(str(init_image))

    print("\nDetected objects by type:")
    for obj_type, names in sorted(detected.items()):
        print(f"  {obj_type}: {names}")

    return detected


def test_fluent_classification(
    images_dir: Path,
    type_to_objects: dict[str, list[str]],
    vendor: str = "openai",
    image_index: int = None,
    show_false: bool = True,
) -> None:
    """Test the gripper fluent classifier on one or all images.

    Args:
        image_index: If set, classify only that image. If None, classify all.
        show_false: If True (default), print FALSE predicates explicitly.
    """
    init_image = images_dir / "state_0.png"
    if not init_image.exists():
        pngs = _find_all_state_images(images_dir)
        if not pngs:
            print("ERROR: No state_*.png images found in", images_dir)
            sys.exit(1)
        init_image = pngs[0]

    from src.fluent_classification.llm_gripper_fluent_classifier import LLMGripperFluentClassifier

    backend = _create_backend(vendor=vendor, model_type="fluent_classification")

    classifier = LLMGripperFluentClassifier(
        llm_backend=backend,
        init_state_image_path=init_image,
        type_to_objects=type_to_objects,
    )

    all_preds = classifier._generate_all_possible_predicates()
    print(f"\nTotal possible predicates: {len(all_preds)}")

    # Determine which images to classify
    if image_index is not None:
        target = images_dir / f"state_{image_index}.png"
        if not target.exists():
            print(f"ERROR: {target.name} not found")
            sys.exit(1)
        targets = [target]
    else:
        targets = _find_all_state_images(images_dir)

    # Classify each image
    all_results = {}
    for target_image in targets:
        print(f"\n{'='*60}")
        print(f"FLUENT CLASSIFICATION — {target_image.name}")
        print(f"{'='*60}")

        result = classifier.classify(target_image)

        true_preds = [p for p, v in sorted(result.items()) if v.name == "TRUE"]
        uncertain_preds = [p for p, v in sorted(result.items()) if v.name == "UNCERTAIN"]
        false_preds = [p for p, v in sorted(result.items()) if v.name == "FALSE"]

        print(f"\n  TRUE ({len(true_preds)}):")
        for p in true_preds:
            print(f"    {p}")
        print(f"\n  UNCERTAIN ({len(uncertain_preds)}):")
        for p in uncertain_preds:
            print(f"    {p}")
        if show_false:
            print(f"\n  FALSE ({len(false_preds)}):")
            for p in false_preds:
                print(f"    {p}")
        else:
            print(f"\n  FALSE: {len(false_preds)} predicates")

        all_results[target_image.name] = {
            p: v.name for p, v in sorted(result.items())
        }

    # Save results to JSON
    output_path = images_dir / "classification_results.json"
    output_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test Gripper LLM components")
    parser.add_argument("images_dir", type=Path, help="Directory with images + trajectory data")
    parser.add_argument("--vendor", default="openai", choices=["openai", "google"],
                        help="LLM vendor (default: openai)")
    parser.add_argument("--classify-index", type=int, default=None,
                        help="Classify only this image index (default: all images)")
    parser.add_argument("--skip-detection", action="store_true",
                        help="Skip object detection, provide objects manually")
    parser.add_argument("--skip-classification", action="store_true",
                        help="Skip fluent classification")
    parser.add_argument("--hide-false", action="store_true",
                        help="Hide FALSE predicates (show count only)")
    args = parser.parse_args()

    if not args.images_dir.exists():
        print(f"ERROR: Directory not found: {args.images_dir}")
        sys.exit(1)

    ensure_trajectory_json(args.images_dir)

    # Object detection
    if not args.skip_detection:
        detected = test_object_detection(args.images_dir, args.vendor)
    else:
        detected = {
            "room": ["rooma", "roomb"],
            "ball": ["ball1", "ball2"],
            "gripper": ["left", "right"],
        }
        print("Using manual object list (--skip-detection)")

    # Fluent classification
    if not args.skip_classification:
        test_fluent_classification(
            args.images_dir, detected, args.vendor, args.classify_index,
            show_false=not args.hide_false,
        )

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
