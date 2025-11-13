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

        # Generate all possible predicates for this domain (used for counting masked)
        self.all_possible_predicates = self.classifier_with_uncertain._generate_all_possible_predicates()

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

    def count_masked_predicates(self, predictions: Dict[str, int]) -> int:
        """
        Count the number of masked/uncertain predicates.

        Masked predicates include:
        1. Predicates with score 1 (uncertain)
        2. Predicates not extracted at all (missing from predictions)

        :param predictions: Dictionary of extracted predictions
        :return: Number of masked predicates
        """
        # Count predicates with score 1 (uncertain)
        num_uncertain = sum(1 for score in predictions.values() if score == 1)

        # Count predicates that were not extracted at all
        extracted_predicates = set(predictions.keys())
        num_missing = len(self.all_possible_predicates - extracted_predicates)

        return num_uncertain + num_missing

    def run_comparison_experiment(
        self,
        image_paths: List[Path],
        ground_truth: Dict[str, Dict[str, int]]
    ) -> Tuple[
        Tuple[MetricsResult, MetricsResult],
        Tuple[MetricsResult, MetricsResult],
        List[Dict]  # Per-image results
    ]:
        """
        Run the full comparison experiment.

        :param image_paths: List of image paths
        :param ground_truth: Dictionary mapping image names to ground truth predicates
        :return: Tuple of ((pos_with, neg_with), (pos_without, neg_without), per_image_results)
        """
        print("Running comparison experiment...")
        print(f"Total images: {len(image_paths)}")

        # Collect all predictions
        all_predictions_with = {}
        all_predictions_without = {}
        all_ground_truth = {}
        per_image_results = []

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

            # Count masked predicates (includes score 1 + missing predicates)
            num_masked_with = self.count_masked_predicates(preds_with)
            num_masked_without = self.count_masked_predicates(preds_without)

            print(f"    WITH uncertain: {len(preds_with)} extracted, {len(preds_with_certain)} certain, {num_masked_with} masked")
            print(f"    WITHOUT uncertain: {len(preds_without_certain)} extracted, {num_masked_without} masked")

            # Compute per-image metrics
            image_gt = ground_truth[image_name]

            # Metrics with uncertain (only certain predictions)
            metrics_with_img = MetricsCalculator.compute_metrics(
                preds_with_certain, image_gt
            )

            # Metrics without uncertain
            metrics_without_img = MetricsCalculator.compute_metrics(
                preds_without_certain, image_gt
            )

            # Store per-image results
            per_image_results.append({
                'image_name': image_name,
                'num_masked_with': num_masked_with,
                'num_masked_without': num_masked_without,
                'metrics_with': metrics_with_img,
                'metrics_without': metrics_without_img
            })

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

        # TODO: filter predicates that do not appear in ground truth?
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

        return metrics_with, metrics_without, per_image_results

    def save_results_to_csv(
        self,
        metrics_with: Tuple[MetricsResult, MetricsResult],
        metrics_without: Tuple[MetricsResult, MetricsResult],
        output_path: Path,
        per_image_results: List[Dict] = None
    ) -> None:
        """
        Save results to CSV files.

        :param metrics_with: Metrics with uncertain option
        :param metrics_without: Metrics without uncertain option
        :param output_path: Directory to save CSV files
        :param per_image_results: Optional per-image results to save
        """
        output_path.mkdir(parents=True, exist_ok=True)

        pos_with, neg_with = metrics_with
        pos_without, neg_without = metrics_without

        # Save summary
        summary_path = output_path / "comparison_summary.csv"
        with open(summary_path, 'w') as f:
            f.write("variant,class,precision,recall,f1,tp,fp,tn,fn,total_samples\n")

            # With uncertain
            f.write(f"with_uncertain,positive,{pos_with.precision:.4f},{pos_with.recall:.4f},{pos_with.f1:.4f},"
                    f"{pos_with.true_positives},{pos_with.false_positives},"
                    f"{pos_with.true_negatives},{pos_with.false_negatives},{pos_with.total_samples}\n")

            f.write(f"with_uncertain,negative,{neg_with.precision:.4f},{neg_with.recall:.4f},{neg_with.f1:.4f},"
                    f"{neg_with.true_positives},{neg_with.false_positives},"
                    f"{neg_with.true_negatives},{neg_with.false_negatives},{neg_with.total_samples}\n")

            # Without uncertain
            f.write(f"without_uncertain,positive,{pos_without.precision:.4f},{pos_without.recall:.4f},{pos_without.f1:.4f},"
                    f"{pos_without.true_positives},{pos_without.false_positives},"
                    f"{pos_without.true_negatives},{pos_without.false_negatives},{pos_without.total_samples}\n")

            f.write(f"without_uncertain,negative,{neg_without.precision:.4f},{neg_without.recall:.4f},{neg_without.f1:.4f},"
                    f"{neg_without.true_positives},{neg_without.false_positives},"
                    f"{neg_without.true_negatives},{neg_without.false_negatives},{neg_without.total_samples}\n")

        print(f"\nResults saved to {summary_path}")

        # Save per-image results if provided
        if per_image_results:
            self.save_per_image_results_to_csv(per_image_results, output_path)

    def save_per_image_results_to_csv(
        self,
        per_image_results: List[Dict],
        output_path: Path
    ) -> None:
        """
        Save per-image results to CSV file.

        Each image will have 4 rows (with_uncertain positive/negative, without_uncertain positive/negative).

        :param per_image_results: List of per-image result dictionaries
        :param output_path: Directory to save CSV file
        """
        per_image_path = output_path / "comparison_per_image.csv"
        with open(per_image_path, 'w') as f:
            # Write header
            f.write("image_name,variant,class,num_masked,precision,recall,f1,tp,fp,tn,fn,total_samples\n")

            # Write data for each image
            for result in per_image_results:
                image_name = result['image_name']
                num_masked_with = result['num_masked_with']
                num_masked_without = result['num_masked_without']
                metrics_with = result['metrics_with']
                metrics_without = result['metrics_without']

                pos_with, neg_with = metrics_with
                pos_without, neg_without = metrics_without

                # With uncertain - positive class
                f.write(f"{image_name},with_uncertain,positive,{num_masked_with},"
                        f"{pos_with.precision:.4f},{pos_with.recall:.4f},{pos_with.f1:.4f},"
                        f"{pos_with.true_positives},{pos_with.false_positives},"
                        f"{pos_with.true_negatives},{pos_with.false_negatives},{pos_with.total_samples}\n")

                # With uncertain - negative class
                f.write(f"{image_name},with_uncertain,negative,{num_masked_with},"
                        f"{neg_with.precision:.4f},{neg_with.recall:.4f},{neg_with.f1:.4f},"
                        f"{neg_with.true_positives},{neg_with.false_positives},"
                        f"{neg_with.true_negatives},{neg_with.false_negatives},{neg_with.total_samples}\n")

                # Without uncertain - positive class
                f.write(f"{image_name},without_uncertain,positive,{num_masked_without},"
                        f"{pos_without.precision:.4f},{pos_without.recall:.4f},{pos_without.f1:.4f},"
                        f"{pos_without.true_positives},{pos_without.false_positives},"
                        f"{pos_without.true_negatives},{pos_without.false_negatives},{pos_without.total_samples}\n")

                # Without uncertain - negative class
                f.write(f"{image_name},without_uncertain,negative,{num_masked_without},"
                        f"{neg_without.precision:.4f},{neg_without.recall:.4f},{neg_without.f1:.4f},"
                        f"{neg_without.true_positives},{neg_without.false_positives},"
                        f"{neg_without.true_negatives},{neg_without.false_negatives},{neg_without.total_samples}\n")

        print(f"Per-image results saved to {per_image_path}")
