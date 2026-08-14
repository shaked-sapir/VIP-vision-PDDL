"""
Data source abstraction for benchmark experiments.

Separates *where observations come from* from *which algorithm learns from them*.
Two concrete implementations:

  ImageDataSource     — reads pre-generated .trajectory / .masking_info files
                        produced by the real image pipeline (data_generator.py).
  SimulatedDataSource — builds Observation objects in-memory by injecting
                        synthetic masking + noise into ground-truth trajectories.

Both expose the same two-method interface so run_single_fold() is data-source-agnostic:

    data_source.setup(problem_dirs, domain_ref_path, gt_rate_percentages, frame_axiom_mode)
    prepared, pre_built_obs, gt_indices = data_source.prepare(
        selected_pool, num_trajectories, gt_rate, fold, output_dir
    )

Both guarantee the same contract: ``prepared_trajectories`` always point at real,
on-disk degraded (masked + flipped) files. ``ImageDataSource`` reads pre-generated
files; ``SimulatedDataSource`` persists its in-memory degraded observations to
``output_dir/original_observations/`` so the learner and the baselines consume
byte-identical data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from pddl_plus_parser.models import Observation

from benchmark.experiment_running_helpers.cleaned_trajectories import save_fold_observations
from benchmark.experiment_running_helpers.simulated_data_utils import (
    build_gt_trajectory_lookup,
    prepare_simulated_observations,
    select_simulated_gt_trajectories,
)
from benchmark.experiment_running_helpers.trajectory_utils import (
    pregenerate_all_gt_frame_axiom_files,
    prepare_fold_trajectories,
)
from src.observation_degradation.masking import MaskingType
from src.observation_degradation.noising import NoisingType

# Type alias: (trajectory_path, masking_path, problem_pddl_path, gt_state_indices)
PreparedTrajectory = Tuple[Path, Path, Path, Set[int]]

PrepareResult = Tuple[
    List[PreparedTrajectory],            # prepared_trajectories
    Optional[List[Observation]],         # pre_built_observations (None for file-based)
    Optional[Dict[int, Set[int]]],       # gt_source_indices (None for file-based)
]


class DataSource(ABC):
    """Abstract base for experiment data sources."""

    @abstractmethod
    def setup(
        self,
        problem_dirs: List[Path],
        domain_ref_path: Path,
        gt_rate_percentages: List[int],
        frame_axiom_mode: str,
    ) -> None:
        """One-time pre-experiment setup (called once before the fold loop)."""

    @abstractmethod
    def prepare(
        self,
        selected_pool: List[Path],
        num_trajectories: int,
        gt_rate: int,
        fold: int,
        output_dir: Path,
    ) -> PrepareResult:
        """Prepare observations for one fold.

        Args:
            output_dir: The fold's working directory. Used by sources that must
                persist their degraded data to disk (so ``prepared_trajectories``
                point at real files). File-based sources may ignore it.

        Returns:
            prepared_trajectories: List of (traj_path, masking_path, pddl_path, gt_indices),
                always pointing at real on-disk degraded files.
            pre_built_observations: Pre-built Observation objects, or None when
                observations should be loaded from prepared_trajectories by the
                learning helper.
            gt_source_indices: Explicit GT index map, or None to derive from
                prepared_trajectories in the learning helper.
        """


class ImageDataSource(DataSource):
    """Reads pre-generated trajectory files produced by the image pipeline.

    setup() pre-generates all GT + frame-axiom files on disk (once).
    prepare() loads the correct subset for a given fold/gt_rate.
    """

    def setup(
        self,
        problem_dirs: List[Path],
        domain_ref_path: Path,
        gt_rate_percentages: List[int],
        frame_axiom_mode: str,
    ) -> None:
        pregenerate_all_gt_frame_axiom_files(
            problem_dirs, domain_ref_path, gt_rate_percentages, frame_axiom_mode,
        )

    def prepare(
        self,
        selected_pool: List[Path],
        num_trajectories: int,
        gt_rate: int,
        fold: int,
        output_dir: Path,
    ) -> PrepareResult:
        # File-based sources already point at real on-disk degraded files.
        prepared = prepare_fold_trajectories(selected_pool, num_trajectories, gt_rate)
        return prepared, None, None


class SimulatedDataSource(DataSource):
    """Builds observations in-memory by injecting noise into GT trajectories.

    setup() is a no-op — no disk pre-generation is needed.
    prepare() loads GT trajectory files and applies synthetic masking + noise
    on the fly for each fold, using fold-offset seeds for reproducibility.
    """

    def __init__(
        self,
        gt_trajectory_paths: List[Path],
        masking_strategy: MaskingType = MaskingType.PERCENTAGE,
        masking_p: float = 0.4,
        noising_strategy: NoisingType = NoisingType.PERCENTAGE,
        noising_p: float = 0.15,
        seed: int = 42,
    ) -> None:
        self._gt_paths = gt_trajectory_paths
        self._masking_strategy = masking_strategy
        self._masking_p = masking_p
        self._noising_strategy = noising_strategy
        self._noising_p = noising_p
        self._seed = seed
        self._gt_lookup = build_gt_trajectory_lookup(gt_trajectory_paths)
        self._domain_ref_path: Optional[Path] = None

    # Read-only accessors so callers (e.g. run_params logging) don't reach into
    # private attributes.
    @property
    def masking_p(self) -> float:
        return self._masking_p

    @property
    def noising_p(self) -> float:
        return self._noising_p

    @property
    def masking_strategy(self) -> MaskingType:
        return self._masking_strategy

    @property
    def noising_strategy(self) -> NoisingType:
        return self._noising_strategy

    def setup(
        self,
        problem_dirs: List[Path],
        domain_ref_path: Path,
        gt_rate_percentages: List[int],
        frame_axiom_mode: str,
    ) -> None:
        """Store domain path for use in prepare(); no disk pre-generation needed."""
        self._domain_ref_path = domain_ref_path

    def prepare(
        self,
        selected_pool: List[Path],
        num_trajectories: int,
        gt_rate: int,
        fold: int,
        output_dir: Path,
    ) -> PrepareResult:
        if self._domain_ref_path is None:
            raise RuntimeError("SimulatedDataSource.setup() must be called before prepare().")

        if gt_rate > 0:
            # TODO: implement injection if desired to run with gt_rate>0
            raise NotImplementedError(
                "SimulatedDataSource.prepare() does not support gt_rate>0 yet "
                "(GT-state injection is not implemented for simulated runs); "
                f"got gt_rate={gt_rate}."
            )

        selected_dirs = selected_pool[:num_trajectories]
        selected_gt = select_simulated_gt_trajectories(
            selected_pool, num_trajectories, self._gt_lookup,
        )
        print(
            f"  [SIMULATED] Loading {len(selected_gt)} GT trajectories "
            f"({', '.join(p.parent.name for p in selected_gt)}) + injecting noise..."
        )
        observations, gt_source_indices = prepare_simulated_observations(
            domain_path=self._domain_ref_path,
            gt_trajectory_paths=selected_gt,
            masking_strategy=self._masking_strategy,
            masking_p=self._masking_p,
            noising_strategy=self._noising_strategy,
            noising_p=self._noising_p,
            seed=self._seed + fold,
        )

        # Persist the degraded (masked + flipped) observations to the cell so
        # every consumer — our conflict search and the baselines — reads the
        # identical on-disk data. The problem .pddl comes from the training-pool
        # dir, where it actually lives (gt_trajectories/ holds only .trajectory).
        original_obs_dir = output_dir / "original_observations"
        problem_pddls = [d / f"{d.name}.pddl" for d in selected_dirs]
        naming = [(None, None, pddl, {0}) for pddl in problem_pddls]
        save_fold_observations(observations, naming, original_obs_dir, "original_observation")

        prepared: List[PreparedTrajectory] = []
        for problem_dir, problem_pddl in zip(selected_dirs, problem_pddls):
            stem = f"original_observation_{problem_dir.name}"
            prepared.append((
                original_obs_dir / f"{stem}.trajectory",
                original_obs_dir / f"{stem}.masking_info",
                problem_pddl,
                {0},
            ))

        print(f"  ✓ Simulated {len(observations)} noisy observations "
              f"→ {original_obs_dir.name}/")
        return prepared, observations, gt_source_indices
