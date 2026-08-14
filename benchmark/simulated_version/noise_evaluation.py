"""Evaluation oracle for simulated conflict-search experiments.

Compares the conflict search's proposed ``FluentLevelPatch`` set against the
ground-truth flips injected by ``create_bounded_noisy_observation``.

Matching is done on the *normalised* fluent form (canonical positive string,
see ``FluentLevelPatch.normalized_fluent``) so that polarity differences
between the injected and proposed representations do not cause false mismatches.
"""

from dataclasses import dataclass
from typing import FrozenSet, Set, Tuple

from src.plan_denoising.typings import FluentLevelPatch


# A canonical key that uniquely identifies a fluent flip independent of polarity.
_PatchKey = Tuple[int, int, str, str]  # (obs_idx, comp_idx, state_type, normalized_fluent)


def _key(patch: FluentLevelPatch) -> _PatchKey:
    return (
        patch.observation_index,
        patch.component_index,
        patch.state_type,
        patch.normalized_fluent,
    )


@dataclass(frozen=True)
class RecoveryMetrics:
    """Precision / recall metrics for noise-flip recovery.

    Attributes:
        precision: Fraction of proposed patches that match a ground-truth flip.
        recall: Fraction of ground-truth flips that were proposed.
        f1: Harmonic mean of precision and recall.
        true_positives: Proposed patches that match a ground-truth flip.
        false_positives: Proposed patches with no matching ground-truth flip.
        false_negatives: Ground-truth flips that were not proposed.
    """
    precision: float
    recall: float
    f1: float
    true_positives: FrozenSet[FluentLevelPatch]
    false_positives: FrozenSet[FluentLevelPatch]
    false_negatives: FrozenSet[FluentLevelPatch]

    def __str__(self) -> str:
        return (
            f"RecoveryMetrics("
            f"precision={self.precision:.3f}, "
            f"recall={self.recall:.3f}, "
            f"f1={self.f1:.3f}, "
            f"|TP|={len(self.true_positives)}, "
            f"|FP|={len(self.false_positives)}, "
            f"|FN|={len(self.false_negatives)})"
        )


def evaluate_noise_recovery(
    proposed: Set[FluentLevelPatch],
    ground_truth: Set[FluentLevelPatch],
) -> RecoveryMetrics:
    """Compute precision, recall, and F1 for conflict-search noise recovery.

    Args:
        proposed: The set of ``FluentLevelPatch`` objects returned by the
            conflict search.
        ground_truth: The set of ``FluentLevelPatch`` objects returned by
            ``create_bounded_noisy_observation``.

    Returns:
        A ``RecoveryMetrics`` instance with precision, recall, F1, and the
        full TP / FP / FN patch sets.
    """
    proposed_keys = {_key(p): p for p in proposed}
    gt_keys = {_key(p): p for p in ground_truth}

    tp_keys = set(proposed_keys) & set(gt_keys)
    fp_keys = set(proposed_keys) - set(gt_keys)
    fn_keys = set(gt_keys) - set(proposed_keys)

    true_positives = frozenset(proposed_keys[k] for k in tp_keys)
    false_positives = frozenset(proposed_keys[k] for k in fp_keys)
    false_negatives = frozenset(gt_keys[k] for k in fn_keys)

    precision = len(tp_keys) / len(proposed_keys) if proposed_keys else 0.0
    recall = len(tp_keys) / len(gt_keys) if gt_keys else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return RecoveryMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )
