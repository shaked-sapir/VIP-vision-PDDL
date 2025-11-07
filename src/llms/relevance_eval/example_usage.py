"""Example usage of relevance comparison system."""

import json
from pathlib import Path
from src.llms.relevance_eval import RelevanceComparator, PromptVariant


def load_ground_truth(ground_truth_path: Path) -> dict:
    """Load ground truth from JSON file."""
    with open(ground_truth_path, 'r') as f:
        return json.load(f)


def example_basic_comparison():
    """Example: Basic comparison with a few images."""
    # Configuration
    openai_apikey = "your-openai-api-key"
    block_colors = ["red", "blue", "green", "cyan"]

    # Paths
    ground_truth_path = Path("src/llms/ground_truth.json")
    images_dir = Path("path/to/images")

    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_path)

    # Get image paths
    image_paths = sorted(images_dir.glob("state_*.png"))[:5]  # First 5 images

    # Create comparator
    comparator = RelevanceComparator(
        openai_apikey=openai_apikey,
        block_colors=block_colors
    )

    # Run experiment
    metrics_with, metrics_without = comparator.run_comparison_experiment(
        image_paths=image_paths,
        ground_truth=ground_truth
    )

    # Save results
    comparator.save_results_to_csv(
        metrics_with=metrics_with,
        metrics_without=metrics_without,
        output_path=Path("results/relevance_comparison")
    )


def example_single_image():
    """Example: Extract predicates from a single image with both prompts."""
    # Configuration
    openai_apikey = "your-openai-api-key"
    block_colors = ["red", "blue", "green"]
    image_path = Path("path/to/state_0001.png")

    # Create comparator
    comparator = RelevanceComparator(
        openai_apikey=openai_apikey,
        block_colors=block_colors
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


def example_full_experiment():
    """Example: Full experiment with all images and detailed analysis."""
    # Configuration
    openai_apikey = "your-openai-api-key"
    block_colors = ["red", "blue", "green", "cyan"]

    # Paths
    ground_truth_path = Path("src/llms/ground_truth.json")
    images_dir = Path("path/to/images")
    output_dir = Path("results/relevance_comparison_full")

    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_path)

    # Get all images
    image_paths = sorted(images_dir.glob("state_*.png"))
    print(f"Found {len(image_paths)} images")

    # Create comparator
    comparator = RelevanceComparator(
        openai_apikey=openai_apikey,
        block_colors=block_colors,
        model="gpt-4o"
    )

    # Run experiment
    print("\nRunning full comparison experiment...")
    metrics_with, metrics_without = comparator.run_comparison_experiment(
        image_paths=image_paths,
        ground_truth=ground_truth
    )

    # Save results
    print("\nSaving results...")
    comparator.save_results_to_csv(
        metrics_with=metrics_with,
        metrics_without=metrics_without,
        output_path=output_dir
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

    # Uncomment to run examples:
    # example_single_image()
    # example_basic_comparison()
    # example_full_experiment()
