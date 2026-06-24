"""ROSAME baseline runners (fully-observable and partially-observable)."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from benchmark.baselines.base_runner import BaselineRunner


def _record_timing(profiler, category, step_name, elapsed, traj_idx, problem_name):
    """Helper to record timing with consistent metadata."""
    if profiler:
        profiler.add_detailed_timing(
            category, step_name, elapsed,
            {"trajectory_index": traj_idx, "problem_name": problem_name},
        )


def _setup_rosame_workspace(
    prepared_trajectories: List[Tuple[Path, Path, Path]],
    workspace_name: str,
    work_dir: Path,
    copy_masking: bool,
) -> List[str]:
    """Copy trajectory/problem/masking files into a ROSAME-specific workspace.

    ROSAME expects each trajectory in its own subdirectory named after the
    problem, alongside the ``.pddl`` problem file (and optionally a
    ``.masking_info`` file for PO-ROSAME).

    Returns:
        List of trajectory file paths inside the workspace.
    """
    workspace_dir = work_dir / f"temp_{workspace_name}_workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    traj_paths: List[str] = []

    for traj_path, masking_path, problem_pddl_path, *_ in prepared_trajectories:
        problem_name = problem_pddl_path.stem

        # Validate inputs
        if not traj_path.exists():
            print(f"  Warning: Trajectory file does not exist: {traj_path}")
            continue
        if traj_path.stat().st_size == 0:
            print(f"  Warning: Trajectory file is EMPTY: {traj_path}")
            continue
        if not problem_pddl_path.exists():
            print(f"  Warning: Problem PDDL does not exist: {problem_pddl_path}")
            continue
        if problem_pddl_path.stat().st_size == 0:
            print(f"  Warning: Problem PDDL is EMPTY: {problem_pddl_path}")
            continue

        problem_dir = workspace_dir / problem_name
        problem_dir.mkdir(parents=True, exist_ok=True)

        dest_traj_path = problem_dir / f"{problem_name}.trajectory"
        shutil.copy(traj_path, dest_traj_path)
        shutil.copy(problem_pddl_path, problem_dir / f"{problem_name}.pddl")

        if copy_masking and masking_path.exists():
            shutil.copy(masking_path, problem_dir / f"{problem_name}.masking_info")

        traj_paths.append(str(dest_traj_path))

    return traj_paths


class RosameBaselineRunner(BaselineRunner):
    """Fully-observable ROSAME baseline (binary state encoding)."""

    @property
    def name(self) -> str:
        return "ROSAME"

    @property
    def display_name(self) -> str:
        return "ROSAME"

    @property
    def color(self) -> str:
        return "#e8710a"

    def supports_mode(self, mode: str) -> bool:
        return mode == "fullyobs"

    def learn(
        self,
        mode: str,
        domain_path: Path,
        prepared_trajectories: List[Tuple[Path, Path, Path]],
        work_dir: Path,
        timeout_seconds: int = 60,
        profiler=None,
    ) -> Tuple[Optional[str], Dict]:
        from benchmark.amlgym_models.po_rosame_runner import PORosame_Runner
        from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser

        traj_paths = _setup_rosame_workspace(
            prepared_trajectories, "rosame", work_dir, copy_masking=False,
        )
        if not traj_paths:
            print("  [ROSAME] No valid trajectories, skipping")
            return None, {}

        try:
            partial_domain = DomainParser(domain_path, partial_parsing=True).parse_domain()
            rosame = PORosame_Runner(str(domain_path))

            for traj_idx, traj_path_str in enumerate(traj_paths):
                traj_path = Path(traj_path_str)
                problem_path = traj_path.parent / f"{traj_path.parent.stem}.pddl"
                problem_name = traj_path.stem
                category = "rosame_trajectory_processing"

                problem = ProblemParser(problem_path, partial_domain).parse_problem()
                rosame.add_problem(problem)

                def _time_step(step_name, func):
                    start = time.perf_counter() if profiler else None
                    result = func()
                    _record_timing(
                        profiler, category, step_name,
                        time.perf_counter() - start if profiler else 0,
                        traj_idx, problem_name,
                    )
                    return result

                observation = _time_step(
                    "parse_trajectory",
                    lambda: TrajectoryParser(partial_domain).parse_trajectory(traj_path),
                )
                rosame.ground_new_trajectory()
                _time_step("learn_rosame_single", lambda: rosame.learn_rosame(observation))

            model = rosame.rosame_to_pddl()
            if model and ":action" in model:
                return model, {}
            raise ValueError("Invalid ROSAME model")

        except Exception as e:
            print(f"  Warning: ROSAME learning failed: {e}")
            return None, {}


class PORosameBaselineRunner(BaselineRunner):
    """Partially-observable ROSAME baseline (ternary state encoding, 0.5 for masked)."""

    @property
    def name(self) -> str:
        return "PO_ROSAME"

    @property
    def display_name(self) -> str:
        return "PO-ROSAME"

    @property
    def color(self) -> str:
        return "#e8710a"

    def supports_mode(self, mode: str) -> bool:
        return mode == "masked"

    def learn(
        self,
        mode: str,
        domain_path: Path,
        prepared_trajectories: List[Tuple[Path, Path, Path]],
        work_dir: Path,
        timeout_seconds: int = 60,
        profiler=None,
    ) -> Tuple[Optional[str], Dict]:
        from benchmark.amlgym_models.po_rosame_runner import PORosame_Runner
        from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
        from src.utils.masking import load_masking_info, mask_observation
        from src.utils.pddl import ground_observation_completely

        traj_paths = _setup_rosame_workspace(
            prepared_trajectories, "po_rosame", work_dir, copy_masking=True,
        )
        if not traj_paths:
            print("  [PO_ROSAME] No valid trajectories, skipping")
            return None, {}

        try:
            partial_domain = DomainParser(domain_path, partial_parsing=True).parse_domain()
            pi_rosame = PORosame_Runner(str(domain_path))

            for traj_idx, traj_path_str in enumerate(traj_paths):
                traj_path = Path(traj_path_str)
                problem_path = traj_path.parent / f"{traj_path.parent.stem}.pddl"
                masking_info_path = traj_path.parent / f"{traj_path.parent.stem}.masking_info"
                problem_name = traj_path.stem
                category = "rosame_trajectory_processing"

                problem = ProblemParser(problem_path, partial_domain).parse_problem()
                pi_rosame.add_problem(problem)

                def _time_step(step_name, func):
                    start = time.perf_counter() if profiler else None
                    result = func()
                    _record_timing(
                        profiler, category, step_name,
                        time.perf_counter() - start if profiler else 0,
                        traj_idx, problem_name,
                    )
                    return result

                observation = _time_step(
                    "parse_trajectory",
                    lambda: TrajectoryParser(partial_domain).parse_trajectory(traj_path),
                )
                grounded_observation = _time_step(
                    "ground_observation_completely",
                    lambda: ground_observation_completely(partial_domain, observation),
                )
                masking_info = _time_step(
                    "load_masking_info",
                    lambda: load_masking_info(masking_info_path, partial_domain),
                )
                masked_observation = _time_step(
                    "mask_observation",
                    lambda: mask_observation(grounded_observation, masking_info),
                )

                pi_rosame.ground_new_trajectory()
                _time_step("learn_rosame_single", lambda: pi_rosame.learn_rosame(masked_observation))

            model = pi_rosame.rosame_to_pddl()
            if model and ":action" in model:
                return model, {}
            raise ValueError("Invalid PO_ROSAME model")

        except Exception as e:
            print(f"  Warning: PO_ROSAME learning failed: {e}")
            return None, {}
