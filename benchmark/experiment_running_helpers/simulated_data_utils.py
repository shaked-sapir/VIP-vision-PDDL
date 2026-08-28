"""
Simulated data utilities — load GT trajectories and inject synthetic masking + noise.

Produces the same ``List[Observation]`` and ``gt_source_indices_by_obs`` that the
file-based pipeline produces, so the downstream conflict search + evaluation code
is identical for both data sources.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from pddl_plus_parser.lisp_parsers import (
    DomainParser,
    ProblemParser,
    TrajectoryParser,
)
from pddl_plus_parser.models import Observation

from benchmark.simulated_version.noise_injection import create_bounded_noisy_observation
from src.observation_degradation.masking import MaskingType
from src.observation_degradation.noising import NoisingType
from src.observation_degradation.predicate_masking import PredicateMasker
from src.observation_degradation.predicate_noising import PredicateNoiser
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

def load_gt_observation(
    trajectory_path: Path, domain, problem_path: Optional[Path] = None
) -> Observation:
    """Parse a trajectory file and ground it completely — no masking applied.

    Args:
        trajectory_path: The ``.trajectory`` to parse.
        domain: Parsed (partial) domain.
        problem_path: The window's problem file. Without it the parser can only
            infer objects from the trajectory text, which lists positive fluents
            only — so an object that goes unmentioned for a whole window is
            absent from the object table, and grounding then raises ``KeyError``
            on it. Measured on gripper: 44 of 800 windows never mention one of
            the two rooms, because the robot stayed in the other one.

    Returns:
        The fully grounded observation.
    """
    if problem_path is not None and Path(problem_path).exists():
        problem = ProblemParser(
            problem_path=Path(problem_path), domain=domain
        ).parse_problem()
        observation = TrajectoryParser(
            partial_domain=domain, problem=problem
        ).parse_trajectory(trajectory_path)
    else:
        observation = TrajectoryParser(partial_domain=domain).parse_trajectory(
            trajectory_path
        )
    return ground_observation_completely(domain, observation)


# ---------------------------------------------------------------------------
# CV-aligned trajectory selection
# ---------------------------------------------------------------------------

def build_gt_trajectory_lookup(gt_trajectory_paths: List[Path]) -> Dict[str, Path]:
    """Map problem directory name (e.g. ``problem3``) → GT trajectory path."""
    lookup: Dict[str, Path] = {}
    for path in gt_trajectory_paths:
        lookup[path.parent.name] = path
    return lookup


def select_simulated_gt_trajectories(
    selected_problem_dirs: List[Path],
    num_trajectories: int,
    gt_trajectory_lookup: Dict[str, Path],
) -> List[Path]:
    """Pick GT trajectories matching the fold's CV-sampled training pool."""
    selected: List[Path] = []
    for prob_dir in selected_problem_dirs[:num_trajectories]:
        gt_path = gt_trajectory_lookup.get(prob_dir.name)
        if gt_path is None:
            available = ", ".join(sorted(gt_trajectory_lookup))
            raise FileNotFoundError(
                f"No simulated GT trajectory for {prob_dir.name}. "
                f"Available: {available}"
            )
        selected.append(gt_path)
    return selected


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
    problem_paths: Optional[List[Path]] = None,
) -> Tuple[List[Observation], Dict[int, Set[int]]]:
    """Load GT trajectories and inject synthetic masking + noise.

    This is the simulated-data counterpart of ``prepare_fold_trajectories`` +
    file loading.  It returns pre-built Observation objects ready to be passed
    directly into ``learn_cdps(..., pre_built_observations=...)``.

    Args:
        domain_path: Path to the PDDL domain file.
        gt_trajectory_paths: Paths to fully ground-truth ``.trajectory`` files.
        masking_strategy: Which masking strategy to use.
        masking_p: The probability / ratio parameter for the masking strategy.
        noising_strategy: Which noising strategy to use.
        noising_p: The probability / ratio parameter for the noising strategy.
        seed: Random seed passed to both masker and noiser.
        problem_paths: The problem file for each trajectory, positionally
            aligned with ``gt_trajectory_paths``. Supplies the declared object
            table; see :func:`load_gt_observation` for what goes wrong without
            it. ``None`` falls back to inferring objects from the trajectory.

    Returns:
        A two-tuple:
          - ``observations``: List of Observation objects with synthetic
            masking and noise applied.
          - ``gt_source_indices_by_obs``: Dict mapping each observation index
            to the set of component indices whose source state is ground truth.
            In simulated mode only the initial state (index 0) is GT — it is
            also left unmasked and un-noised in the observation data.
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
        gt_obs = load_gt_observation(
            traj_path, domain,
            problem_paths[obs_idx] if problem_paths else None,
        )
        noisy_obs, _masking_info, flips = create_bounded_noisy_observation(
            gt_obs, masker, noiser, obs_idx=obs_idx,
        )
        observations.append(noisy_obs)
        print(f"  trajectory {obs_idx} ({traj_path.name}): {len(flips)} fluents flipped")

    # Only component 0's source state (t=0) is GT — preserved above, not corrupted.
    gt_source_indices_by_obs: Dict[int, Set[int]] = {
        obs_idx: {0} for obs_idx in range(len(observations))
    }

    print(f"  Total observations: {len(observations)}")
    return observations, gt_source_indices_by_obs
