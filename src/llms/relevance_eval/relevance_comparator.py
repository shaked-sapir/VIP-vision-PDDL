"""Compare LLM performance with and without the 'uncertain' relevance option."""

from pathlib import Path
from typing import Dict, List, Tuple
from enum import Enum

from src.fluent_classification.llm_blocks_fluent_classifier import LLMBlocksFluentClassifier
from src.llms.relevance_eval.metrics_calculator import MetricsCalculator, MetricsResult


class PromptVariant(Enum):
    """Which prompt variant to use."""
    WITH_UNCERTAIN = "with_uncertain"      # Allows scores: 0, 1, 2
    WITHOUT_UNCERTAIN = "without_uncertain"  # Allows scores: 0, 2 only


class RelevanceComparator:
    """
    Compares LLM performance with and without the 'uncertain' (1) option.

    This class runs experiments to determine whether allowing the model to express
    uncertainty (score 1) improves precision on its confident predictions (scores 0 and 2).

    The hypothesis: If the model can say "I'm uncertain" for ambiguous cases, its
    definite predictions (0 and 2) should be more accurate.

    Uses the existing LLMBlocksFluentClassifier with minimal additional code.
    """

    def __init__(
        self,
        openai_apikey: str,
        block_colors: List[str],
        model: str = "gpt-4o"
    ):
        """
        Initialize the relevance comparator.

        :param openai_apikey: OpenAI API key
        :param block_colors: List of block colors in the domain
        :param model: GPT-4 Vision model to use
        """
        self.openai_apikey = openai_apikey
        self.block_colors = block_colors
        self.model = model

        # Create type_to_objects mapping
        self.type_to_objects = {
            'block': block_colors,
            'gripper': ['gripper']
        }

        # Create classifier with uncertain option
        self.classifier_with_uncertain = LLMBlocksFluentClassifier(
            openai_apikey=openai_apikey,
            type_to_objects=self.type_to_objects,
            model=model,
            use_uncertain=True
        )

        # Create classifier without uncertain option
        self.classifier_without_uncertain = LLMBlocksFluentClassifier(
            openai_apikey=openai_apikey,
            type_to_objects=self.type_to_objects,
            model=model,
            use_uncertain=False
        )

    def extract_predicates_with_relevance(
        self,
        image_path: Path,
        prompt_variant: PromptVariant,
        temperature: float = 1.0
    ) -> Dict[str, int]:
        """
        Extract predicates with relevance scores from an image.

        Uses the existing LLMBlocksFluentClassifier.extract_facts_once() method.

        :param image_path: Path to the image
        :param prompt_variant: Which prompt variant to use
        :param temperature: Temperature for generation
        :return: Dictionary mapping predicates to relevance scores (0, 1, or 2)
        """
        # Select classifier
        if prompt_variant == PromptVariant.WITH_UNCERTAIN:
            classifier = self.classifier_with_uncertain
        else:
            classifier = self.classifier_without_uncertain

        # Extract predicates using the classifier's method
        # extract_facts_once returns set of (predicate, relevance) tuples
        predicates_set = classifier.extract_facts_once(image_path, temperature=temperature)

        # Convert to dictionary
        predicates_dict = {pred: relevance for pred, relevance in predicates_set}

        return predicates_dict

    @staticmethod
    def filter_to_certain_predictions(predictions: Dict[str, int]) -> Dict[str, int]:
        """
        Filter predictions to only include certain ones (scores 0 and 2).

        This removes all uncertain predictions (score 1).

        :param predictions: Dictionary with all predictions
        :return: Dictionary with only certain predictions (0 and 2)
        """
        return {pred: score for pred, score in predictions.items() if score in [0, 2]}

    def run_comparison_experiment(
        self,
        image_paths: List[Path],
        ground_truth: Dict[str, Dict[str, int]]
    ) -> Tuple[
        Tuple[MetricsResult, MetricsResult],
        Tuple[MetricsResult, MetricsResult]
    ]:
        """
        Run the full comparison experiment.

        :param image_paths: List of image paths
        :param ground_truth: Dictionary mapping image names to ground truth predicates
        :return: Tuple of ((pos_with, neg_with), (pos_without, neg_without))
        """
        print("Running comparison experiment...")
        print(f"Total images: {len(image_paths)}")

        # Collect all predictions
        all_predictions_with = {}
        all_predictions_without = {}
        all_ground_truth = {}

        for image_path in image_paths:
            image_name = image_path.stem  # e.g., "state_0001"

            if image_name not in ground_truth:
                print(f"Warning: No ground truth for {image_name}, skipping")
                continue

            print(f"\nProcessing {image_name}...")

            # Extract with uncertain option
            print("  - Extracting WITH uncertain option...")
            preds_with = self.extract_predicates_with_relevance(
                image_path, PromptVariant.WITH_UNCERTAIN
            )

            # Extract without uncertain option
            print("  - Extracting WITHOUT uncertain option...")
            preds_without = self.extract_predicates_with_relevance(
                image_path, PromptVariant.WITHOUT_UNCERTAIN
            )

            # Filter both to only certain predictions (0 and 2)
            preds_with_certain = self.filter_to_certain_predictions(preds_with)
            preds_without_certain = preds_without  # Already should only have 0 and 2

            print(f"    WITH uncertain: {len(preds_with)} total, {len(preds_with_certain)} certain")
            print(f"    WITHOUT uncertain: {len(preds_without_certain)} predictions")

            # Add to aggregated results
            for pred, score in preds_with_certain.items():
                all_predictions_with[f"{image_name}:{pred}"] = score

            for pred, score in preds_without_certain.items():
                all_predictions_without[f"{image_name}:{pred}"] = score

            for pred, gt_value in ground_truth[image_name].items():
                all_ground_truth[f"{image_name}:{pred}"] = gt_value

        # Compute metrics
        print("\n" + "="*60)
        print("Computing metrics...")

        metrics_with = MetricsCalculator.compute_metrics(
            all_predictions_with, all_ground_truth
        )

        metrics_without = MetricsCalculator.compute_metrics(
            all_predictions_without, all_ground_truth
        )

        # Print comparison
        MetricsCalculator.print_metrics_comparison(
            metrics_with, metrics_without,
            label="Overall Results"
        )

        return metrics_with, metrics_without

    def save_results_to_csv(
        self,
        metrics_with: Tuple[MetricsResult, MetricsResult],
        metrics_without: Tuple[MetricsResult, MetricsResult],
        output_path: Path
    ) -> None:
        """
        Save results to CSV files.

        :param metrics_with: Metrics with uncertain option
        :param metrics_without: Metrics without uncertain option
        :param output_path: Directory to save CSV files
        """
        output_path.mkdir(parents=True, exist_ok=True)

        pos_with, neg_with = metrics_with
        pos_without, neg_without = metrics_without

        # Save summary
        summary_path = output_path / "comparison_summary.csv"
        with open(summary_path, 'w') as f:
            f.write("variant,class,precision,recall,f1,tp,fp,tn,fn,total_samples\n")

            # With uncertain
            f.write(f"with_uncertain,positive,{pos_with.precision},{pos_with.recall},{pos_with.f1},"
                    f"{pos_with.true_positives},{pos_with.false_positives},"
                    f"{pos_with.true_negatives},{pos_with.false_negatives},{pos_with.total_samples}\n")

            f.write(f"with_uncertain,negative,{neg_with.precision},{neg_with.recall},{neg_with.f1},"
                    f"{neg_with.true_positives},{neg_with.false_positives},"
                    f"{neg_with.true_negatives},{neg_with.false_negatives},{neg_with.total_samples}\n")

            # Without uncertain
            f.write(f"without_uncertain,positive,{pos_without.precision},{pos_without.recall},{pos_without.f1},"
                    f"{pos_without.true_positives},{pos_without.false_positives},"
                    f"{pos_without.true_negatives},{pos_without.false_negatives},{pos_without.total_samples}\n")

            f.write(f"without_uncertain,negative,{neg_without.precision},{neg_without.recall},{neg_without.f1},"
                    f"{neg_without.true_positives},{neg_without.false_positives},"
                    f"{neg_without.true_negatives},{neg_without.false_negatives},{neg_without.total_samples}\n")

        print(f"\nResults saved to {summary_path}")
