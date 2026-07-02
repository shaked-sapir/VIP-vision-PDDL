"""
Data Generator for Benchmark System

Generates multi-problem training trajectories for all supported domains.
Uses a domain registry to avoid per-domain boilerplate — adding a new domain
means adding one entry to _DOMAIN_REGISTRY.

Unified problem layout (all domains):
    src/domains/<domain>/problems/
        ├── problem0/
        │   ├── problem0.pddl
        │   ├── (state_*.png + .trajectory — external domains only)
        │   └── ...
        └── problem1/
            └── ...

Output structure:
    benchmark/data/<domain>/<experiment>/training/trajectories/
        ├── problem0/
        │   ├── problem0.pddl
        │   ├── problem0.trajectory      (inferred)
        │   └── problem0.masking_info     (inferred)
        └── ...
"""

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trajectory_handlers.llm_blocks_trajectory_handler import LLMBlocksImageTrajectoryHandler
from src.trajectory_handlers.llm_npuzzle_trajectory_handler import LLMNpuzzleImageTrajectoryHandler
from src.trajectory_handlers.llm_hanoi_trajectory_handler import LLMHanoiImageTrajectoryHandler
from src.trajectory_handlers.llm_hiking_trajectory_handler import LLMHikingImageTrajectoryHandler
from src.trajectory_handlers.llm_maze_trajectory_handler import LLMMazeImageTrajectoryHandler
from src.trajectory_handlers.llm_depot_trajectory_handler import LLMDepotImageTrajectoryHandler
from src.trajectory_handlers.llm_gripper_trajectory_handler import LLMGripperImageTrajectoryHandler
from src.utils.config import load_config


# ── Problem-file transforms (PDDLGym → AMLGym) ───────────────────────────

def _transform_blocks_problem(problem_file_path: Path) -> None:
    """Remove robot type and handfull predicate from a blocksworld problem file."""
    content = problem_file_path.read_text()
    content = (content
               .replace('robot - robot', '')
               .replace('(handempty robot)', '(handempty)')
               .replace('(handfull robot)', '')
               .replace('(handfull)', ''))
    problem_file_path.write_text(content)


def _transform_npuzzle_problem(problem_file_path: Path) -> None:
    """Replace an n-puzzle problem file with the eval-compatible version."""
    eval_source = project_root / "benchmark" / "domains" / "n_puzzle" / "eight01x_eval.pddl"
    content = eval_source.read_text()
    problem_file_path.write_text(content)


def _apply_transform(transform_fn: Optional[Callable], problem_dir: Path) -> None:
    """Apply a per-file transform to all .pddl files in a problem directory."""
    if transform_fn is None:
        return
    for pddl_file in problem_dir.glob("*.pddl"):
        transform_fn(pddl_file)


# ── External domain cleanup ─────────────────────────────────────────────

def _cleanup_external_source_files(problem_dir: Path) -> None:
    """Remove source images from an external problem dir after inference.

    Keeps:
        - {problem}.trajectory — the inferred trajectory (overwrites GT during pipeline)
        - {problem}.masking_info — inference masking info
        - {problem}.pddl — problem file
        - {problem}_trajectory.json — GT trajectory JSON (needed by trajectory_utils
          for GT injection at various rates during evaluation)

    Removes:
        - state_*.png — source images (no longer needed after inference)
        - plan.txt — source plan file
    """
    for img in problem_dir.glob("state_*.png"):
        img.unlink()
    for plan_file in problem_dir.glob("plan.txt"):
        plan_file.unlink()


# ── Domain registry ──────────────────────────────────────────────────────

_DOMAIN_REGISTRY = {
    "blocksworld": {
        "display_name": "BLOCKSWORLD",
        "config_key": "blocks",
        "handler_class": LLMBlocksImageTrajectoryHandler,
        "transform_fn": _transform_blocks_problem,
        "is_external": False,
    },
    "npuzzle": {
        "display_name": "N-PUZZLE",
        "config_key": "n_puzzle",
        "handler_class": LLMNpuzzleImageTrajectoryHandler,
        "transform_fn": _transform_npuzzle_problem,
        "is_external": False,
    },
    "hanoi": {
        "display_name": "HANOI",
        "config_key": "hanoi",
        "handler_class": LLMHanoiImageTrajectoryHandler,
        "transform_fn": None,
        "is_external": False,
    },
    "hiking": {
        "display_name": "HIKING",
        "config_key": "hiking",
        "handler_class": LLMHikingImageTrajectoryHandler,
        "transform_fn": None,
        "is_external": False,
    },
    "maze": {
        "display_name": "MAZE",
        "config_key": "maze",
        "handler_class": LLMMazeImageTrajectoryHandler,
        "transform_fn": None,
        "is_external": False,
    },
    "depot": {
        "display_name": "DEPOT",
        "config_key": "depot",
        "handler_class": LLMDepotImageTrajectoryHandler,
        "transform_fn": None,
        "is_external": True,
    },
    "gripper": {
        "display_name": "GRIPPER",
        "config_key": "gripper",
        "handler_class": LLMGripperImageTrajectoryHandler,
        "transform_fn": None,
        "is_external": True,
    },
}


def _natural_sort_key(path: Path):
    """Sort key that orders problem1, problem2, ..., problem10 numerically."""
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', path.name)]


def _collect_problem_dirs(problems_dir: Path) -> List[Path]:
    """Collect problem subdirectories from the unified layout.

    Each problem is a subdirectory of problems_dir containing at minimum
    a .pddl file (and for external domains, images + GT trajectory).
    """
    return sorted(
        [d for d in problems_dir.iterdir() if d.is_dir() and not d.name.startswith(('.', '_'))],
        key=_natural_sort_key,
    )


# ── Main generation function ─────────────────────────────────────────────

def generate_trajectories(
    domain: str,
    output_base_dir: Path,
    num_steps: int = 100,
    vendor: str = "openai",
    start_index: int = 0,
    planner: Optional[str] = None,
    problem_start: Optional[int] = None,
    problem_end: Optional[int] = None,
) -> Path:
    """Generate trajectories for all problems in a domain.

    Works for both PDDLGym and external domains via handler.run_pipeline.

    PDDLGym flow:
        1. Create empty output dir.
        2. run_pipeline generates images + GT there, then infers.
        3. Copy problem .pddl from source, apply transform.

    External flow:
        1. Copy entire source problem dir (images + GT + .pddl) to output.
        2. run_pipeline reads images and GT from the copy, infers in place.
        3. Clean up: remove images, GT trajectory, and JSON — keep only
           inferred .trajectory + .masking_info + .pddl.

    Args:
        domain: Domain key from _DOMAIN_REGISTRY.
        output_base_dir: Base directory for benchmark data (e.g. benchmark/data/).
        num_steps: Max trajectory steps (gym domains only, default 100).
        vendor: LLM vendor — "openai" or "google" (default "openai").
        start_index: Advance gym env N steps before generating (gym only, default 0).
        planner: Planner for gym trajectory — "ff", "fd", or None for random (default None).
        problem_start: First problem index to process (0-based, inclusive, default None = all).
        problem_end: Last problem index to process (0-based, inclusive, default None = all).

    Returns:
        Path to the trajectories directory.
    """
    registry = _DOMAIN_REGISTRY[domain]
    config_key = registry["config_key"]
    is_external = registry["is_external"]

    # Load configuration
    config = load_config()
    domain_config = config["domains"][config_key]
    domain_file = Path(domain_config["domain_file"])
    problems_dir = Path(domain_config["problems_dir"])
    domain_name = domain_config.get("gym_domain_name", config_key)

    # Model name for experiment naming
    model_name = config[vendor].get("fluent_classification_model", {}).get("model_name", vendor)

    # Build experiment name
    timestamp = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
    experiment_name = f"multi_problem_{timestamp}__model={model_name}__steps={num_steps}"
    if start_index > 0:
        experiment_name += f"__start={start_index}"
    if planner:
        experiment_name += f"__planner={planner}"

    print("=" * 80)
    print(f"GENERATING {registry['display_name']} TRAJECTORIES")
    print(f"Experiment: {experiment_name}")
    print(f"Mode: {'External images' if is_external else (planner.upper() + ' planner' if planner else 'Random actions')}")
    if start_index > 0:
        print(f"Starting from state index: {start_index}")
    print("=" * 80)
    print()

    # Setup output directories (use CLI domain name, not config_key, to match existing data dirs)
    experiment_dir = output_base_dir / domain / experiment_name
    trajectories_dir = experiment_dir / "training" / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    # Collect and filter problem subdirectories
    problem_dirs = _collect_problem_dirs(problems_dir)
    print(f"Found {len(problem_dirs)} problems in {problems_dir}")

    if problem_start is not None or problem_end is not None:
        start = problem_start if problem_start is not None else 0
        end = (problem_end + 1) if problem_end is not None else len(problem_dirs)
        problem_dirs = problem_dirs[start:end]
        print(f"Filtered to {len(problem_dirs)} problems (indices {start}–{end - 1})")

    print(f"Domain name: {domain_name}")
    print()

    # Build kwargs for run_pipeline (gym-specific ones are harmlessly ignored by external handlers)
    pipeline_kwargs = {}
    if num_steps != 100:
        pipeline_kwargs["num_steps"] = num_steps
    if start_index > 0:
        pipeline_kwargs["start_index"] = start_index
    if planner:
        pipeline_kwargs["planner"] = planner

    # Process each problem
    for problem_idx, source_problem_dir in enumerate(problem_dirs):
        problem_name = source_problem_dir.name
        print(f"[{problem_idx + 1}/{len(problem_dirs)}] Processing {problem_name}...")

        output_problem_dir = trajectories_dir / problem_name
        output_problem_dir.mkdir(exist_ok=True)

        try:
            handler = registry["handler_class"](
                domain_name=domain_name,
                pddl_domain_file=domain_file,
                vendor=vendor,
            )

            if is_external:
                # Copy source data (images + GT + .pddl) to output dir for inference
                shutil.copytree(source_problem_dir, output_problem_dir, dirs_exist_ok=True)

                handler.run_pipeline(
                    problem_name=problem_name,
                    images_path=output_problem_dir,
                    **pipeline_kwargs,
                )

                # Clean up: remove source images and GT, keep inferred output + .pddl
                _cleanup_external_source_files(output_problem_dir)

            else:
                # PDDLGym: handler generates images + GT into output dir, then infers
                handler.run_pipeline(
                    problem_name=problem_name,
                    images_path=output_problem_dir,
                    **pipeline_kwargs,
                )

                # Copy and transform problem file from source
                source_pddl = source_problem_dir / f"{problem_name}.pddl"
                shutil.copy(source_pddl, output_problem_dir)
                _apply_transform(registry["transform_fn"], output_problem_dir)

            print(f"  ✓ {problem_name}")
            print()

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            print()
            continue

    print()
    print("=" * 80)
    print("TRAJECTORY GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nExperiment saved to: {experiment_dir}")
    print(f"  Trajectories: {trajectories_dir}")
    print(f"  Problems processed: {len(list(trajectories_dir.iterdir()))}")
    print()

    return trajectories_dir


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate training trajectories for benchmark experiments"
    )
    parser.add_argument(
        "--domain", type=str, required=True,
        choices=list(_DOMAIN_REGISTRY.keys()),
        help="Domain to generate data for",
    )
    parser.add_argument(
        "--num-steps", type=int, default=100,
        help="Max trajectory steps per problem (gym domains only, default: 100)",
    )
    parser.add_argument(
        "--start-index", type=int, default=0,
        help="Advance gym env N steps before generating (gym only, default: 0)",
    )
    parser.add_argument(
        "--planner", type=str, default=None, choices=["ff", "fd"],
        help="Planner to use (ff or fd). Omit for random actions.",
    )
    parser.add_argument(
        "--problem-start", type=int, default=None,
        help="First problem index to process (0-based, inclusive)",
    )
    parser.add_argument(
        "--problem-end", type=int, default=None,
        help="Last problem index to process (0-based, inclusive)",
    )
    parser.add_argument(
        "--vendor", type=str, default="openai", choices=["openai", "google"],
        help="LLM vendor for the vision pipeline (default: openai)",
    )

    args = parser.parse_args()

    generate_trajectories(
        domain=args.domain,
        output_base_dir=Path(__file__).parent / "data",
        num_steps=args.num_steps,
        vendor=args.vendor,
        start_index=args.start_index,
        planner=args.planner,
        problem_start=args.problem_start,
        problem_end=args.problem_end,
    )
