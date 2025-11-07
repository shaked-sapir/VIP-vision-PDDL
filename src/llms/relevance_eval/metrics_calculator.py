"""Metrics calculation for relevance evaluation."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict


@dataclass
class MetricsResult:
    """
    Results of metrics computation.

    Attributes:
        precision: Precision score
        recall: Recall score
        f1: F1 score
        true_positives: Number of true positives
        false_positives: Number of false positives
        true_negatives: Number of true negatives
        false_negatives: Number of false negatives
        total_samples: Total number of samples
    """
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_samples: int

    def __str__(self):
        return (f"Precision: {self.precision:.4f}, Recall: {self.recall:.4f}, F1: {self.f1:.4f}\n"
                f"TP: {self.true_positives}, FP: {self.false_positives}, "
                f"TN: {self.true_negatives}, FN: {self.false_negatives}")


class MetricsCalculator:
    """
    Calculator for precision, recall, and F1 metrics.

    Computes metrics for binary classification, handling both positive and negative classes.
    """

    @staticmethod
    def compute_metrics(
        predictions: Dict[str, int],
        ground_truth: Dict[str, int]
    ) -> Tuple[MetricsResult, MetricsResult]:
        """
        Compute metrics for both positive and negative classes.

        :param predictions: Dictionary mapping predicates to predicted relevance scores (0 or 2)
        :param ground_truth: Dictionary mapping predicates to ground truth values (0 or 1)
        :return: Tuple of (positive_metrics, negative_metrics)
        """
        # Compute positive class metrics (predicting 1/2, ground truth 1)
        pos_metrics = MetricsCalculator._compute_binary_metrics(
            predictions, ground_truth, positive_class=True
        )

        # Compute negative class metrics (predicting 0, ground truth 0)
        neg_metrics = MetricsCalculator._compute_binary_metrics(
            predictions, ground_truth, positive_class=False
        )

        return pos_metrics, neg_metrics

    @staticmethod
    def _compute_binary_metrics(
        predictions: Dict[str, int],
        ground_truth: Dict[str, int],
        positive_class: bool
    ) -> MetricsResult:
        """
        Compute metrics for a single class (positive or negative).

        :param predictions: Predicted relevance scores (0 or 2)
        :param ground_truth: Ground truth values (0 or 1)
        :param positive_class: If True, compute for positive class; else negative
        :return: MetricsResult for this class
        """
        tp = fp = tn = fn = 0

        # Get all predicates from both predictions and ground truth
        all_predicates = set(predictions.keys()) | set(ground_truth.keys())

        for pred_str in all_predicates:
            # Get predicted value (default to 0 if not predicted)
            predicted_value = predictions.get(pred_str, 0)

            # Get ground truth value (default to 0 if not in ground truth)
            true_value = ground_truth.get(pred_str, 0)

            # Convert relevance scores to binary predictions
            # For predictions: 2 = positive (1), 0 = negative (0)
            predicted_binary = 1 if predicted_value == 2 else 0

            if positive_class:
                # Positive class metrics
                if predicted_binary == 1 and true_value == 1:
                    tp += 1
                elif predicted_binary == 1 and true_value == 0:
                    fp += 1
                elif predicted_binary == 0 and true_value == 0:
                    tn += 1
                elif predicted_binary == 0 and true_value == 1:
                    fn += 1
            else:
                # Negative class metrics (flip the logic)
                if predicted_binary == 0 and true_value == 0:
                    tp += 1
                elif predicted_binary == 0 and true_value == 1:
                    fp += 1
                elif predicted_binary == 1 and true_value == 1:
                    tn += 1
                elif predicted_binary == 1 and true_value == 0:
                    fn += 1

        # Compute precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return MetricsResult(
            precision=precision,
            recall=recall,
            f1=f1,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            total_samples=len(all_predicates)
        )

    @staticmethod
    def compute_metrics_by_predicate_type(
        predictions: Dict[str, int],
        ground_truth: Dict[str, int]
    ) -> Dict[str, Tuple[MetricsResult, MetricsResult]]:
        """
        Compute metrics grouped by predicate type.

        :param predictions: Predicted relevance scores
        :param ground_truth: Ground truth values
        :return: Dictionary mapping predicate types to (pos_metrics, neg_metrics)
        """
        # Group by predicate type
        predictions_by_type = defaultdict(dict)
        ground_truth_by_type = defaultdict(dict)

        for pred_str in set(predictions.keys()) | set(ground_truth.keys()):
            # Extract predicate type (e.g., "on" from "on(block1, block2)")
            pred_type = pred_str.split('(')[0] if '(' in pred_str else pred_str

            if pred_str in predictions:
                predictions_by_type[pred_type][pred_str] = predictions[pred_str]
            if pred_str in ground_truth:
                ground_truth_by_type[pred_type][pred_str] = ground_truth[pred_str]

        # Compute metrics for each predicate type
        metrics_by_type = {}
        for pred_type in set(predictions_by_type.keys()) | set(ground_truth_by_type.keys()):
            preds = predictions_by_type[pred_type]
            gt = ground_truth_by_type[pred_type]
            metrics_by_type[pred_type] = MetricsCalculator.compute_metrics(preds, gt)

        return metrics_by_type

    @staticmethod
    def print_metrics_comparison(
        metrics_with_uncertain: Tuple[MetricsResult, MetricsResult],
        metrics_without_uncertain: Tuple[MetricsResult, MetricsResult],
        label: str = ""
    ) -> None:
        """
        Print a comparison of metrics with and without the uncertain option.

        :param metrics_with_uncertain: Metrics when uncertain (1) is allowed
        :param metrics_without_uncertain: Metrics when uncertain is not allowed
        :param label: Optional label for the comparison
        """
        pos_with, neg_with = metrics_with_uncertain
        pos_without, neg_without = metrics_without_uncertain

        print(f"\n{'='*60}")
        if label:
            print(f"Metrics Comparison: {label}")
        else:
            print("Metrics Comparison")
        print(f"{'='*60}")

        print("\n--- WITH Uncertain Option (allows '1') ---")
        print("Positive Class:")
        print(f"  {pos_with}")
        print("\nNegative Class:")
        print(f"  {neg_with}")

        print("\n--- WITHOUT Uncertain Option (only '0' or '2') ---")
        print("Positive Class:")
        print(f"  {pos_without}")
        print("\nNegative Class:")
        print(f"  {neg_without}")

        # Compute deltas
        print("\n--- Delta (WITH - WITHOUT) ---")
        print("Positive Class:")
        print(f"  Precision: {pos_with.precision - pos_without.precision:+.4f}")
        print(f"  Recall:    {pos_with.recall - pos_without.recall:+.4f}")
        print(f"  F1:        {pos_with.f1 - pos_without.f1:+.4f}")

        print("\nNegative Class:")
        print(f"  Precision: {neg_with.precision - neg_without.precision:+.4f}")
        print(f"  Recall:    {neg_with.recall - neg_without.recall:+.4f}")
        print(f"  F1:        {neg_with.f1 - neg_without.f1:+.4f}")

        print(f"{'='*60}\n")
