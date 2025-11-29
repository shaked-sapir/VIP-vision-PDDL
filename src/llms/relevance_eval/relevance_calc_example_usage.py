"""Example usage of relevance comparison system."""

import json
from pathlib import Path
from src.llms.relevance_eval import RelevanceComparator, PromptVariant
from src.utils.config import load_config


def load_ground_truth(ground_truth_path: Path) -> dict:
    """Load ground truth from JSON file."""
    with open(ground_truth_path, 'r') as f:
        return json.load(f)


def example_basic_comparison(openai_apikey: str, ground_truth_path: Path, images_dir: Path, model: str):
    """Example: Basic comparison with a few images."""
    # Configuration
    block_colors = ["red", "blue", "green", "cyan"]

    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_path)

    # Get image paths
    image_paths = sorted(images_dir.glob("state_*.png"))[:7]  # First X images

    # Create comparator
    comparator = RelevanceComparator(
        openai_apikey=openai_apikey,
        block_colors=block_colors,
        model=model
    )

    # Run experiment
    metrics_with, metrics_without, per_image_results = comparator.run_comparison_experiment(
        image_paths=image_paths,
        ground_truth=ground_truth
    )

    # Save results
    comparator.save_results_to_csv(
        metrics_with=metrics_with,
        metrics_without=metrics_without,
        output_path=Path(f"results/relevance_comparison/{model}"),
        per_image_results=per_image_results
    )


def example_single_image(openai_apikey: str, image_path: Path, model: str):
    """Example: Extract predicates from a single image with both prompts."""
    # Configuration
    block_colors = ["red", "blue", "green", "cyan"]

    # Create comparator
    comparator = RelevanceComparator(
        openai_apikey=openai_apikey,
        block_colors=block_colors,
        model=model
    )

    # Extract with uncertain option
    print("Extracting WITH uncertain option...")
    preds_with = comparator.extract_predicates_with_relevance(
        image_path, PromptVariant.WITH_UNCERTAIN
    )

    print("\nPredicates (with uncertain):")
    for pred, score in sorted(preds_with.items()):
        print(f"  {pred}: {score}")

    # Filter to certain predictions
    preds_with_certain = comparator.filter_to_certain_predictions(preds_with)
    print(f"\nCertain predictions: {len(preds_with_certain)}/{len(preds_with)}")

    # Extract without uncertain option
    print("\nExtracting WITHOUT uncertain option...")
    preds_without = comparator.extract_predicates_with_relevance(
        image_path, PromptVariant.WITHOUT_UNCERTAIN
    )

    print("\nPredicates (without uncertain):")
    for pred, score in sorted(preds_without.items()):
        print(f"  {pred}: {score}")


def example_full_experiment(openai_apikey: str, ground_truth_path: Path, images_dir: Path, model: str, problem: str):
    """Example: Full experiment with all images and detailed analysis."""
    # Configuration
    block_colors = ["red", "blue", "green", "cyan"]

    # Paths
    output_dir = Path(f"results/relevance_comparison_full/{model}/{problem}")

    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_path)

    # Get all images
    image_paths = sorted(images_dir.glob("state_*.png"))
    print(f"Found {len(image_paths)} images")

    # Create comparator
    comparator = RelevanceComparator(
        openai_apikey=openai_apikey,
        block_colors=block_colors,
        model=model
    )

    # Run experiment
    print(f"\nRunning full comparison experiment with {model} on {problem}...")
    metrics_with, metrics_without, per_image_results = comparator.run_comparison_experiment(
        image_paths=image_paths,
        ground_truth=ground_truth
    )

    # Save results
    print("\nSaving results...")
    comparator.save_results_to_csv(
        metrics_with=metrics_with,
        metrics_without=metrics_without,
        output_path=output_dir,
        per_image_results=per_image_results
    )

    # Print summary
    pos_with, neg_with = metrics_with
    pos_without, neg_without = metrics_without

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\n--- Positive Class Precision ---")
    print(f"WITH uncertain:    {pos_with.precision:.4f}")
    print(f"WITHOUT uncertain: {pos_without.precision:.4f}")
    print(f"Difference:        {pos_with.precision - pos_without.precision:+.4f}")

    print("\n--- Negative Class Precision ---")
    print(f"WITH uncertain:    {neg_with.precision:.4f}")
    print(f"WITHOUT uncertain: {neg_without.precision:.4f}")
    print(f"Difference:        {neg_with.precision - neg_without.precision:+.4f}")

    # Analysis
    print("\n--- Analysis ---")
    if pos_with.precision > pos_without.precision:
        print("✓ Allowing 'uncertain' IMPROVES positive precision")
    else:
        print("✗ Allowing 'uncertain' DECREASES positive precision")

    if neg_with.precision > neg_without.precision:
        print("✓ Allowing 'uncertain' IMPROVES negative precision")
    else:
        print("✗ Allowing 'uncertain' DECREASES negative precision")


if __name__ == "__main__":
    print("Relevance Comparison Examples")
    print("="*60)
    config = load_config()
    openai_apikey = config['openai']['api_key']
    single_image_path = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/llms/domains/blocks/images/state_0000.png")
    gt_path = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/llms/ground_truth_problem3.json")
    problem_number="3"
    images_dir_suffix = f"images_problem_{problem_number}"
    # Uncomment to run examples:
    # example_single_image(openai_apikey=openai_apikey, image_path=single_image_path, model=config['openai']['visual_components_model_name'])
    # example_basic_comparison(
    #     openai_apikey=openai_apikey,
    #     ground_truth_path=gt_path,
    #     images_dir=Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/llms/domains/blocksworld/images/"),
    #     model="gpt-4o"
    # )
    example_full_experiment(
        openai_apikey=openai_apikey,
        ground_truth_path=gt_path,
        images_dir=Path(
            f"/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/llms/domains/blocks/{images_dir_suffix}/"),
        model="gpt-4.1-mini",
        problem=f"problem{problem_number}"

    )
    example_full_experiment(
        openai_apikey=openai_apikey,
        ground_truth_path=gt_path,
        images_dir=Path(f"/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/llms/domains/blocks/{images_dir_suffix}/"),
        model="gpt-4o",
        problem=f"problem{problem_number}"

    )
    example_full_experiment(
        openai_apikey=openai_apikey,
        ground_truth_path=gt_path,
        images_dir=Path(f"/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/llms/domains/blocks/{images_dir_suffix}/"),
        model="gpt-5-nano",
        problem=f"problem{problem_number}"
    )
    example_full_experiment(
        openai_apikey=openai_apikey,
        ground_truth_path=gt_path,
        images_dir=Path(f"/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/llms/domains/blocks/{images_dir_suffix}/"),
        model="gpt-5-mini",
        problem=f"problem{problem_number}"
    )
    example_full_experiment(
        openai_apikey=openai_apikey,
        ground_truth_path=gt_path,
        images_dir=Path(f"/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/llms/domains/blocks/{images_dir_suffix}/"),
        model="gpt-5",
        problem=f"problem{problem_number}"
    )