"""ROSAME baseline runner.

A single runner is used for every experiment cell. It always uses the ternary
(partially-observable) encoding — masked fluents are encoded as 0.5, unmasked
fluents as 1.0/0.0. When nothing is masked (e.g. mask=0 cells) the ternary
encoding degenerates to plain binary ROSAME, so one consistent "ROSAME" line
covers the whole grid.

Noise needs no special handling: flipped fluents already carry their wrong label
in the (degraded) trajectory, so the encoder reads the wrong value directly.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from benchmark.algorithm_adapters.anytime_snapshots import SnapshotWriter
from benchmark.algorithm_adapters.best_checkpoint import BestModelTracker
from benchmark.algorithm_adapters.po_rosame_runner import DEFAULT_BATCH_SIZE, PORosame_Runner
from benchmark.algorithm_adapters.seeding import seed_everything
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser

from benchmark.baselines.base_runner import BaselineRunner
from src.milp.loss_convergence import LossConvergenceRule
from src.utils.masking import load_masking_info, mask_observation
from src.utils.pddl import ground_observation_completely

TRAINING_SERIES_DIRNAME = "rosame_training"


class LearningBudget:
    """The fold's wall-clock learning budget, started when constructed."""

    def __init__(self, timeout_seconds: Optional[float]) -> None:
        self.timeout_seconds = timeout_seconds
        self._start = time.perf_counter()

    def seconds_left(self) -> float:
        """Seconds remaining; ``inf`` when no budget was given."""
        if self.timeout_seconds is None:
            return float("inf")
        return self.timeout_seconds - (time.perf_counter() - self._start)

    def exhausted(self) -> bool:
        """Whether the budget is spent."""
        return self.seconds_left() <= 0


def write_training_series(work_dir: Path, arm_name: str, payload: Mapping[str, object]) -> Path:
    """Write one arm's per-epoch loss series to ``<work_dir>/rosame_training/<arm>.json``."""
    out_dir = Path(work_dir) / TRAINING_SERIES_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{arm_name}.json"
    path.write_text(json.dumps(payload))
    return path


def _setup_rosame_workspace(
    prepared_trajectories: List[Tuple[Path, Path, Path]],
    work_dir: Path,
) -> List[str]:
    """Copy trajectory/problem/masking files into a ROSAME-specific workspace.

    ROSAME expects each trajectory in its own subdirectory named after the
    problem, alongside the ``.pddl`` problem file and (when present) a
    ``.masking_info`` file.

    Returns:
        List of trajectory file paths inside the workspace.
    """
    workspace_dir = work_dir / "temp_rosame_workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    traj_paths: List[str] = []

    for traj_path, masking_path, problem_pddl_path, *_ in prepared_trajectories:
        problem_name = problem_pddl_path.stem

        if not traj_path.exists() or traj_path.stat().st_size == 0:
            print(f"  Warning: Trajectory missing/empty: {traj_path}")
            continue
        if not problem_pddl_path.exists() or problem_pddl_path.stat().st_size == 0:
            print(f"  Warning: Problem PDDL missing/empty: {problem_pddl_path}")
            continue

        problem_dir = workspace_dir / problem_name
        problem_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(traj_path, problem_dir / f"{problem_name}.trajectory")
        shutil.copy(problem_pddl_path, problem_dir / f"{problem_name}.pddl")
        if masking_path is not None and masking_path.exists():
            shutil.copy(masking_path, problem_dir / f"{problem_name}.masking_info")

        traj_paths.append(str(problem_dir / f"{problem_name}.trajectory"))

    return traj_paths


class RosameBaselineRunner(BaselineRunner):
    """ROSAME baseline — always ternary (0.5 for masked, degenerates to binary).

    Two training schedules are selectable via ``train_per_trajectory`` (default
    ``True`` = the historical continual loop):
      - per-trajectory: fully train each trace before the next (ordering bias);
      - pooled: one persistent optimizer, all traces interleaved per epoch.
    Pooled needs no shared object universe (plain ROSAME has no CV head).

    ``snapshot_interval`` turns on anytime instrumentation: every Nth epoch the
    current model is written under ``anytime_snapshots/<name>/``. It is off by
    default because ROSAME otherwise emits a single model, and every existing
    result was produced that way — a run that silently started writing hundreds
    of extra files per fold would be a surprise, not a feature.
    """

    def __init__(
        self,
        train_per_trajectory: bool = True,
        snapshot_interval: Optional[int] = None,
        batch_size: Optional[int] = None,
        rosame_seed: Optional[int] = 8800,
        epochs: int = 100,
        rosame_convergence: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.train_per_trajectory = train_per_trajectory
        self.snapshot_interval = snapshot_interval
        self.batch_size = batch_size
        self.rosame_seed = rosame_seed
        self.epochs = epochs
        self.convergence_rule = LossConvergenceRule.from_config(rosame_convergence)

    @property
    def effective_batch_size(self) -> int:
        """Transitions per pooled optimizer step; 0 means one step per trace."""
        size = DEFAULT_BATCH_SIZE if self.batch_size is None else self.batch_size
        return size if size and size > 0 else 0

    def run_params(self) -> Dict[str, object]:
        params: Dict[str, object] = {
            "train_per_trajectory": self.train_per_trajectory,
            "batch_size": self.effective_batch_size,
            "rosame_seed": self.rosame_seed,
        }
        if self.convergence_rule is not None:
            params["epochs"] = self.epochs
            params["rosame_convergence"] = self.convergence_rule.as_stats()
        return params

    def _stop_check(self) -> Optional[Callable[[List[float]], bool]]:
        """The loop's stop hook: the plateau rule, or ``None`` when it is off."""
        return None if self.convergence_rule is None else self.convergence_rule.converged

    def _tracker(self) -> BestModelTracker:
        """Best-loss checkpointing, active only when the rule is on."""
        return BestModelTracker(active=self.convergence_rule is not None)

    @property
    def name(self) -> str:
        return "ROSAME_24"

    @property
    def display_name(self) -> str:
        return "ROSAME (24)"

    @property
    def input_kind(self) -> str:
        return "symbolic"

    @property
    def paper(self) -> str:
        return "24"

    @property
    def uses_milp(self) -> bool:
        return False

    @property
    def color(self) -> str:
        return "#e8710a"

    def _build_prepared(
        self, traj_paths: List[str], partial_domain
    ) -> List[Tuple[object, object]]:
        """Parse each workspace trajectory into a ``(problem, masked_obs)`` pair.

        Applies ternary masking (0.5) when a ``.masking_info`` file is present;
        otherwise the observation is fully observed and ROSAME degenerates to
        binary.
        """
        prepared: List[Tuple[object, object]] = []
        for traj_path_str in traj_paths:
            traj_path = Path(traj_path_str)
            problem_name = traj_path.stem
            problem_path = traj_path.parent / f"{problem_name}.pddl"
            masking_info_path = traj_path.parent / f"{problem_name}.masking_info"

            problem = ProblemParser(problem_path, partial_domain).parse_problem()
            observation = TrajectoryParser(partial_domain, problem).parse_trajectory(traj_path)
            grounded = ground_observation_completely(partial_domain, observation)

            if masking_info_path.exists():
                masking_info = load_masking_info(masking_info_path, partial_domain)
                learn_obs = mask_observation(grounded, masking_info)
            else:
                learn_obs = grounded

            prepared.append((problem, learn_obs))
        return prepared

    def learn(
        self,
        domain_path: Path,
        prepared_trajectories: List[Tuple[Path, Path, Path]],
        work_dir: Path,
        timeout_seconds: int = 60,
    ) -> Tuple[Optional[str], Dict]:
        traj_paths = _setup_rosame_workspace(prepared_trajectories, work_dir)
        if not traj_paths:
            print("  [ROSAME] No valid trajectories, skipping")
            return None, {}

        extra_info: Dict = dict(self.run_params())
        seed_everything(self.rosame_seed)
        budget = LearningBudget(timeout_seconds)
        if self.convergence_rule is not None and self.train_per_trajectory:
            print("  [ROSAME] rosame_convergence is set but the per-trajectory schedule "
                  "has no single loss curve; the rule is not applied")
        snapshot = None
        if self.snapshot_interval is not None:
            snapshot = SnapshotWriter(
                work_dir / "anytime_snapshots" / self.name,
                interval=self.snapshot_interval,
            )
            extra_info["snapshot_interval"] = self.snapshot_interval

        try:
            partial_domain = DomainParser(domain_path, partial_parsing=True).parse_domain()
            rosame = PORosame_Runner(str(domain_path))

            prepared = self._build_prepared(traj_paths, partial_domain)
            learn_kwargs = {}
            if self.batch_size is not None:
                learn_kwargs["batch_size"] = self.batch_size
            model, report = rosame.learn_full_with_report(
                prepared,
                train_per_trajectory=self.train_per_trajectory,
                epochs=self.epochs,
                snapshot=snapshot,
                stop_check=self._stop_check(),
                timeout_check=budget.exhausted,
                tracker=self._tracker(),
                **learn_kwargs,
            )
            extra_info.update(report.as_stats())
            if report.losses:
                write_training_series(work_dir, self.name, {
                    **report.as_stats(), "losses": report.losses,
                })
            if model and ":action" in model:
                return model, extra_info
            raise ValueError("Invalid ROSAME model")

        except Exception as e:
            print(f"  Warning: ROSAME learning failed: {e}")
            return None, extra_info
