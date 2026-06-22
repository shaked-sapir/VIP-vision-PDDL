"""
Simulated data utilities — load GT trajectories and inject synthetic masking + noise.

Produces the same ``List[Observation]`` and ``gt_source_indices_by_obs`` that the
file-based pipeline produces, so the downstream conflict search + evaluation code
is identical for both data sources.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser
from pddl_plus_parser.models import Observation

from benchmark.simulated_version.noise_injection import create_bounded_noisy_observation
from src.pi_sam.masking import MaskingType
from src.pi_sam.noising import NoisingType
from src.pi_sam.predicate_masking import PredicateMasker
from src.pi_sam.predicate_noising import PredicateNoiser
from src.utils.pddl_state import ground_observation_completely


# ---------------------------------------------------------------------------
# Strategy → kwarg helpers (mirrored from run_simulated_experiment.py)
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

def load_gt_observation(trajectory_path: Path, domain) -> Observation:
    """Parse a trajectory file and ground it completely — no masking applied."""
    observation = TrajectoryParser(partial_domain=domain).parse_trajectory(trajectory_path)
    return ground_observation_completely(domain, observation)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prepare_simulated_observations(
    domain_path: Path,
    gt_trajectory_paths: List[Path],
    masking_strategy: MaskingType = MaskingType.PERCENTAGE,
    masking_p: float = 0.4,
    noising_strategy: NoisingType = NoisingType.PERCENTAGE,
    noising_p: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Observation], Dict[int, Set[int]]]:
    """Load GT trajectories and inject synthetic masking + noise.

    This is the simulated-data counterpart of ``prepare_fold_trajectories`` +
    file loading.  It returns pre-built Observation objects ready to be passed
    directly into ``learn_sam_pisam(..., pre_built_observations=...)``.

    Args:
        domain_path: Path to the PDDL domain file.
        gt_trajectory_paths: Paths to fully ground-truth ``.trajectory`` files.
        masking_strategy: Which masking strategy to use.
        masking_p: The probability / ratio parameter for the masking strategy.
        noising_strategy: Which noising strategy to use.
        noising_p: The probability / ratio parameter for the noising strategy.
        seed: Random seed passed to both masker and noiser.

    Returns:
        A two-tuple:
          - ``observations``: List of Observation objects with synthetic
            masking and noise applied.
          - ``gt_source_indices_by_obs``: Dict mapping each observation index
            to the set of component indices whose source state is ground truth.
            In simulated mode only the initial state (index 0) is GT.
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

    observations: List[Observation] = []

    print(f"\n=== Simulated noise injection "
          f"(masking={masking_strategy.value}, p={masking_p:.0%} | "
          f"noising={noising_strategy.value}, p={noising_p:.0%}) ===")

    for obs_idx, traj_path in enumerate(gt_trajectory_paths):
        gt_obs = load_gt_observation(traj_path, domain)
        noisy_obs, _masking_info, flips = create_bounded_noisy_observation(
            gt_obs, masker, noiser, obs_idx=obs_idx,
        )
        observations.append(noisy_obs)
        print(f"  trajectory {obs_idx} ({traj_path.name}): {len(flips)} fluents flipped")

    # Only the initial state (t=0) is trusted as GT in simulated mode
    gt_source_indices_by_obs: Dict[int, Set[int]] = {
        obs_idx: {0} for obs_idx in range(len(observations))
    }

    print(f"  Total observations: {len(observations)}")
    return observations, gt_source_indices_by_obs
