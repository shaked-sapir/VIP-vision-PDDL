"""Simulated conflict-search sanity check.

Workflow
--------
1. Load fully ground-truth trajectories (no classifier, no vision noise).
2. Inject bounded masking + polarity noise in memory via
   ``create_bounded_noisy_observation``, which also returns the exact
   ground-truth flip set.
3. Run the conflict-driven patch search on the degraded observations.
4. Compare the proposed ``FluentLevelPatch`` set to the ground truth with
   ``evaluate_noise_recovery`` and print precision / recall / F1.

Usage example
-------------
    python -m benchmark.simulated_version.run_simulated_experiment \\
        --domain  path/to/domain.pddl \\
        --trajectories path/to/traj0.trajectory path/to/traj1.trajectory \\
        --masking-strategy percentage --masking-p 0.4 \\
        --noising-strategy random    --noising-p 0.15
"""

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Set

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser
from utilities import NegativePreconditionPolicy

from benchmark.simulated_version.noise_evaluation import evaluate_noise_recovery
from benchmark.simulated_version.noise_injection import create_bounded_noisy_observation
from src.pi_sam.noisy_pisam.typings import FluentLevelPatch
from src.pi_sam.plan_denoising.conflict_search import ConflictDrivenPatchSearch
from src.pi_sam.plan_denoising.frontier import (
    ConflictGroupStrategy,
    NodeChoosingStrategy,
    SearchMode,
)
from src.pi_sam.predicate_masking import PredicateMasker
from src.pi_sam.predicate_noising import PredicateNoiser
from src.pi_sam.masking import MaskingType
from src.pi_sam.noising import NoisingType
from src.utils.pddl_state import ground_observation_completely


# ---------------------------------------------------------------------------
# Strategy → kwarg helpers
# ---------------------------------------------------------------------------

def _masking_kwargs(strategy: MaskingType, p: float) -> dict:
    if strategy == MaskingType.RANDOM:
        return {"masking_proba": p}
    return {"masking_ratio": p}


def _noising_kwargs(strategy: NoisingType, p: float) -> dict:
    if strategy == NoisingType.RANDOM:
        return {"noise_proba": p}
    return {"noise_ratio": p}


# ---------------------------------------------------------------------------
# GT observation loading
# ---------------------------------------------------------------------------

def load_gt_observation(trajectory_path: Path, domain):
    """Parse a trajectory file and ground it completely — no masking applied."""
    observation = TrajectoryParser(partial_domain=domain).parse_trajectory(trajectory_path)
    return ground_observation_completely(domain, observation)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_simulated_experiment(
    domain_path: Path,
    gt_trajectory_paths: List[Path],
    masking_strategy: MaskingType = MaskingType.PERCENTAGE,
    masking_p: float = 0.4,
    noising_strategy: NoisingType = NoisingType.PERCENTAGE,
    noising_p: float = 0.15,
    seed: int = 42,
    timeout_seconds: int = 60,
    max_search_nodes: int = None,
) -> None:
    """Run the full simulate → search → evaluate pipeline.

    Args:
        domain_path: Path to the PDDL domain file.
        gt_trajectory_paths: Paths to fully ground-truth ``.trajectory`` files.
        masking_strategy: Which masking strategy to use (RANDOM or PERCENTAGE).
        masking_p: The probability / ratio parameter for the masking strategy.
        noising_strategy: Which noising strategy to use (RANDOM or PERCENTAGE).
        noising_p: The probability / ratio parameter for the noising strategy.
        seed: Random seed passed to both masker and noiser.
        timeout_seconds: Conflict-search timeout per run.
        max_search_nodes: Optional cap on search nodes explored.
    """
    domain = DomainParser(domain_path, partial_parsing=True).parse_domain()

    masker = PredicateMasker(
        seed=seed,
        masking_strategy=masking_strategy,
        masking_kwargs=_masking_kwargs(masking_strategy, masking_p),
    )
    noiser = PredicateNoiser(
        seed=seed,
        noising_strategy=noising_strategy,
        noising_kwargs=_noising_kwargs(noising_strategy, noising_p),
    )

    noisy_observations = []
    ground_truth_flips: Set[FluentLevelPatch] = set()

    print(f"\n=== Noise injection "
          f"(masking={masking_strategy.value}, p={masking_p:.0%} | "
          f"noising={noising_strategy.value}, p={noising_p:.0%}) ===")
    for obs_idx, traj_path in enumerate(gt_trajectory_paths):
        gt_obs = load_gt_observation(traj_path, domain)
        noisy_obs, _masking_info, flips = create_bounded_noisy_observation(
            gt_obs, masker, noiser, obs_idx=obs_idx
        )
        noisy_observations.append(noisy_obs)
        ground_truth_flips |= flips
        print(f"  trajectory {obs_idx}: {len(flips)} fluents flipped")

    print(f"Total ground-truth flips: {len(ground_truth_flips)}")

    # Only the initial state (t=0) is trusted as GT — matches gtrate0 benchmark
    # behavior.  Synthetic masking/noise breaks all later states.
    gt_source_indices_by_obs: Dict[int, Set[int]] = {
        obs_idx: {0} for obs_idx in range(len(noisy_observations))
    }

    print("\n=== Running conflict-driven patch search ===")
    conflict_search = ConflictDrivenPatchSearch(
        partial_domain_template=deepcopy(domain),
        negative_preconditions_policy=NegativePreconditionPolicy.hard,
        seed=seed,
        search_mode=SearchMode.ANYTIME_DFS,
        node_choosing_strategy=NodeChoosingStrategy.MODEL_PATCH_FIRST,
        conflict_group_strategy=ConflictGroupStrategy.LARGEST,
    )

    _model, _conflicts, _model_constraints, proposed_patches, _cost, _report, _patched = (
        conflict_search.run(
            observations=noisy_observations,
            max_nodes=max_search_nodes,
            timeout_seconds=timeout_seconds,
            gt_source_indices_by_obs=gt_source_indices_by_obs,
        )
    )

    print(f"Proposed patches: {len(proposed_patches)}")

    print("\n=== Recovery evaluation ===")
    metrics = evaluate_noise_recovery(proposed_patches, ground_truth_flips)
    print(metrics)

    if metrics.false_negatives:
        print("\nMissed flips (false negatives):")
        for patch in sorted(metrics.false_negatives, key=str):
            print(f"  {patch}")

    if metrics.false_positives:
        print("\nSpurious proposals (false positives):")
        for patch in sorted(metrics.false_positives, key=str):
            print(f"  {patch}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulated bounded-noise conflict-search sanity check."
    )
    parser.add_argument("--domain", required=True, type=Path,
                        help="Path to the PDDL domain file.")
    parser.add_argument("--trajectories", required=True, nargs="+", type=Path,
                        help="Paths to fully ground-truth .trajectory files.")
    _masking_choices = [MaskingType.RANDOM, MaskingType.PERCENTAGE]
    parser.add_argument("--masking-strategy", type=MaskingType, default=MaskingType.PERCENTAGE,
                        choices=_masking_choices, metavar="STRATEGY",
                        help="Masking strategy: 'random' or 'percentage' (default: percentage).")
    parser.add_argument("--masking-p", type=float, default=0.4,
                        help="Probability / ratio for the masking strategy (default: 0.4).")
    parser.add_argument("--noising-strategy", type=NoisingType, default=NoisingType.PERCENTAGE,
                        choices=list(NoisingType), metavar="STRATEGY",
                        help="Noising strategy: 'random' or 'percentage' (default: percentage).")
    parser.add_argument("--noising-p", type=float, default=0.15,
                        help="Probability / ratio for the noising strategy (default: 0.15).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=60,
                        help="Conflict-search timeout in seconds (default: 60).")
    parser.add_argument("--max-nodes", type=int, default=None,
                        help="Optional cap on search nodes explored.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_simulated_experiment(
        domain_path=args.domain,
        gt_trajectory_paths=args.trajectories,
        masking_strategy=args.masking_strategy,
        masking_p=args.masking_p,
        noising_strategy=args.noising_strategy,
        noising_p=args.noising_p,
        seed=args.seed,
        timeout_seconds=args.timeout,
        max_search_nodes=args.max_nodes,
    )
