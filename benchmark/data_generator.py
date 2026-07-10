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
from src.trajectory_handlers.pddlgym_problem_generator import PDDLGymProblemGenerator
from src.utils.config import load_config
from src.utils.pddl_trajectory import ensure_trajectory_json, extract_actions_from_trajectory_json


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


def _apply_transform(transform_fn: Optional[Callable], problem_dir: Path) -> None:
    """Apply a per-file transform to all .pddl files in a problem directory."""
    if transform_fn is None:
        return
    for pddl_file in problem_dir.glob("*.pddl"):
        transform_fn(pddl_file)


# ── External domain cleanup ─────────────────────────────────────────────

def _cleanup_external_source_files(problem_dir: Path) -> None:
    """Remove leftover source files from a problem dir after inference.

    Keeps:
        - state_*.png — rendered images (retained for inspection/debugging)
        - {problem}.trajectory — the inferred trajectory (overwrites GT during pipeline)
        - {problem}.masking_info — inference masking info
        - {problem}.pddl — problem file
        - {problem}_trajectory.json — GT trajectory JSON (needed by trajectory_utils
          for GT injection at various rates during evaluation)

    Removes:
        - plan.txt — source plan file (no longer needed after inference)
    """
    for plan_file in problem_dir.glob("plan.txt"):
        plan_file.unlink()


# ── Generation-mode helpers (PDDLGym domains) ────────────────────────────

def _run_generation_inference(handler, problem_name: str, problem_dir: Path) -> None:
    """Run external-style inference over a freshly-generated PDDLGym problem folder.

    The generated folder already contains GT (_trajectory.json) + images + .pddl,
    so we treat it exactly like an external (depot/gripper) problem: extract the
    ground actions from the GT JSON and run the LLM classification pipeline.

    The generator writes the .pddl and _trajectory.json in the raw gym schema
    (its _GymWalkHandler has no schema-translation hook). Inference produces a
    .trajectory already in the eval schema (via _rename_ground_action + the
    classifiers). So after inference we translate the .pddl to the eval schema
    and rebuild _trajectory.json from the translated .trajectory + .pddl, leaving
    all three files consistently in the eval schema.
    """
    ensure_trajectory_json(problem_dir)
    actions = extract_actions_from_trajectory_json(problem_dir)
    handler._set_seq_idx_format(problem_dir)
    handler.create_trajectory_and_masks(problem_name, actions, problem_dir)
    _translate_problem_and_rebuild_json(handler, problem_dir)


def _translate_problem_and_rebuild_json(handler, problem_dir: Path) -> None:
    """Translate the problem .pddl to the eval schema and rebuild its GT JSON.

    Uses the domain handler's translate_problem_pddl (no-op for domains whose
    gym schema already matches the eval schema). After translation the raw GT
    _trajectory.json is stale, so it is deleted and rebuilt from the eval-schema
    .trajectory + .pddl via ensure_trajectory_json.
    """
    for pddl_file in problem_dir.glob("*.pddl"):
        handler.translate_problem_pddl(pddl_file)
    for json_file in problem_dir.glob("*_trajectory.json"):
        json_file.unlink()
    ensure_trajectory_json(problem_dir)


def _resolve_problem_index(domain_config: dict, problem_index: Optional[int]) -> int:
    """Resolve the 0-based problem position (natural order) from an explicit value or config default.

    Position 0 selects problem1, position 1 selects problem2, etc. — matching
    legacy mode's ordering (see PDDLGymProblemGenerator.problem_index).
    """
    if problem_index is not None:
        return problem_index
    gen_cfg = domain_config.get("generation", {})
    return gen_cfg.get("default_problem_index", 0)


# ── Domain registry ──────────────────────────────────────────────────────

_DOMAIN_REGISTRY = {
    "blocksworld": {
        "display_name": "BLOCKSWORLD",
        "config_key": "blocksworld",
        "handler_class": LLMBlocksImageTrajectoryHandler,
        "transform_fn": _transform_blocks_problem,
        "is_external": False,
        "supports_generation": True,
    },
    "npuzzle": {
        "display_name": "N-PUZZLE",
        "config_key": "npuzzle",
        "handler_class": LLMNpuzzleImageTrajectoryHandler,
        "transform_fn": None,
        "is_external": False,
        "supports_generation": True,
    },
    "hanoi": {
        "display_name": "HANOI",
        "config_key": "hanoi",
        "handler_class": LLMHanoiImageTrajectoryHandler,
        "transform_fn": None,
        "is_external": False,
        "supports_generation": True,
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


# ── Generation-mode entry point (generate problems + images, then infer) ──

def generate_trajectories_via_generation(
    domain: str,
    output_base_dir: Path,
    vendor: str = "openai",
    problem_index: Optional[int] = None,
    num_problems: Optional[int] = None,
    length_min: Optional[int] = None,
    length_max: Optional[int] = None,
    skip: Optional[int] = None,
    seed: Optional[int] = None,
    run_inference: bool = True,
    output_dir_name: Optional[str] = None,
) -> Path:
    """Generate brand-new PDDLGym problems (init/goal/images/plan), then infer.

    Unlike the legacy path (which walks pre-authored problem files), this
    (ROSAME-style):
        1. Selects a bundled PDDLGym problem by index (fixes object count) and
           runs one long random walk from it.
        2. Cuts the walk into distinct, solvable sub-problems — each written as a
           depot/gripper-style folder (.pddl + .trajectory + _trajectory.json +
           plan.txt + state_*.png).
        3. Runs the same external-style LLM inference over each folder to produce
           the inferred .trajectory + .masking_info.

    Args:
        domain: Domain key from _DOMAIN_REGISTRY (must support generation).
        output_base_dir: Base directory for benchmark data (e.g. benchmark/data/).
        vendor: LLM vendor — "openai" or "google".
        problem_index: Bundled PDDLGym problem index to walk from. Defaults to config.
        num_problems: Number of problems to generate (default from config).
        length_min: Min window length in steps (default from config).
        length_max: Max window length in steps (default from config).
        skip: States discarded between windows (default from config).
        seed: RNG seed for reproducibility.
        run_inference: If False, only generate folders (skip LLM). Useful for tests.
        output_dir_name: Override for the output directory name. If None, a name is
            auto-generated as ``multi_problem_<timestamp>__model=<model>__gen__...``.

    Returns:
        Path to the trajectories directory.
    """
    registry = _DOMAIN_REGISTRY[domain]
    if not registry.get("supports_generation"):
        raise ValueError(f"Domain '{domain}' does not support generation mode.")

    config = load_config()
    domain_config = config["domains"][registry["config_key"]]
    domain_file = Path(domain_config["domain_file"])
    gym_domain_name = domain_config["gym_domain_name"]
    problem_prefix = domain_config.get("problem_prefix", "problem")
    gen_cfg = domain_config.get("generation", {})

    problem_index = _resolve_problem_index(domain_config, problem_index)
    num_problems = num_problems if num_problems is not None else gen_cfg.get("num_problems", 10)
    length_min = length_min if length_min is not None else gen_cfg.get("length_min", 9)
    length_max = length_max if length_max is not None else gen_cfg.get("length_max", 20)
    skip = skip if skip is not None else gen_cfg.get("skip", 1)
    cursor_steps_limit = config.get("trajectory", {}).get(
        "generation_cursor_steps_limit", 1000)

    model_name = config[vendor].get("fluent_classification_model", {}).get("model_name", vendor)
    timestamp = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
    auto_name = (f"multi_problem_{timestamp}__model={model_name}"
                 f"__gen__prob={problem_index}__len={length_min}-{length_max}")
    experiment_name = output_dir_name if output_dir_name is not None else auto_name

    print("=" * 80)
    print(f"GENERATING {registry['display_name']} PROBLEMS (generation mode)")
    print(f"Experiment: {experiment_name}")
    print(f"Env: {gym_domain_name} | problem index: {problem_index}")
    print(f"Problems: {num_problems} | length {length_min}-{length_max} | skip {skip}")
    print("=" * 80)
    print()

    experiment_dir = output_base_dir / domain / experiment_name
    trajectories_dir = experiment_dir / "training" / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate the problem folders (images + GT + pddl + plan).
    generator = PDDLGymProblemGenerator(gym_domain_name, problem_index=problem_index)
    generated_dirs = generator.generate(
        output_dir=trajectories_dir,
        num_problems=num_problems,
        length_range=(length_min, length_max),
        skip=skip,
        seed=seed,
        problem_prefix=problem_prefix,
        cursor_steps_limit=cursor_steps_limit,
    )
    print(f"Generated {len(generated_dirs)} problem folders.")
    print()

    if not run_inference:
        print("run_inference=False — skipping LLM inference.")
        return trajectories_dir

    # 2. Run external-style inference on each generated folder.
    for problem_idx, problem_dir in enumerate(generated_dirs):
        problem_name = problem_dir.name
        print(f"[{problem_idx + 1}/{len(generated_dirs)}] Inferring {problem_name}...")
        try:
            handler = registry["handler_class"](
                domain_name=gym_domain_name,
                pddl_domain_file=domain_file,
                vendor=vendor,
            )
            _run_generation_inference(handler, problem_name, problem_dir)
            _apply_transform(registry["transform_fn"], problem_dir)
            _cleanup_external_source_files(problem_dir)
            print(f"  ✓ {problem_name}")
            print()
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            print()
            continue

    # Export GT .trajectory files (one per problem) from the retained GT JSONs.
    # Lazy import avoids a circular import (the GT script imports helpers from here).
    from benchmark.generate_gt_trajectories import generate_gt_trajectories
    gt_root = generate_gt_trajectories(trajectories_dir, experiment_dir)

    print("=" * 80)
    print("GENERATION-MODE TRAJECTORY GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nExperiment saved to: {experiment_dir}")
    print(f"  Trajectories: {trajectories_dir}")
    print(f"  GT trajectories: {gt_root}")
    print()
    return trajectories_dir


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
    output_dir_name: Optional[str] = None,
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
        output_dir_name: Override for the output directory name. If None, a name is
            auto-generated as ``multi_problem_<timestamp>__model=<model>__steps=<n>``.

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
    auto_name = f"multi_problem_{timestamp}__model={model_name}__steps={num_steps}"
    if start_index > 0:
        auto_name += f"__start={start_index}"
    if planner:
        auto_name += f"__planner={planner}"
    experiment_name = output_dir_name if output_dir_name is not None else auto_name

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

                # Copy and transform problem file from source. run_pipeline already
                # wrote a translated .trajectory + _trajectory.json (via the domain
                # handler), but the copied .pddl is still in the raw gym schema, so
                # translate it to the eval schema to match.
                source_pddl = source_problem_dir / f"{problem_name}.pddl"
                shutil.copy(source_pddl, output_problem_dir)
                _apply_transform(registry["transform_fn"], output_problem_dir)
                for pddl_file in output_problem_dir.glob("*.pddl"):
                    handler.translate_problem_pddl(pddl_file)

            print(f"  ✓ {problem_name}")
            print()

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            print()
            continue

    # Export GT .trajectory files (one per problem) from the retained GT JSONs.
    # Lazy import avoids a circular import (the GT script imports helpers from here).
    from benchmark.generate_gt_trajectories import generate_gt_trajectories
    gt_root = generate_gt_trajectories(trajectories_dir, experiment_dir)

    print()
    print("=" * 80)
    print("TRAJECTORY GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nExperiment saved to: {experiment_dir}")
    print(f"  Trajectories: {trajectories_dir}")
    print(f"  GT trajectories: {gt_root}")
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

    # ── Generation-mode args (--gen-mode) ────────────────────────────────
    parser.add_argument(
        "--gen-mode", type=str, default="predefined", choices=["predefined", "generate"],
        help="'predefined' runs the LLM pipeline over pre-authored problem files; "
             "'generate' creates new problems + images from a bundled PDDLGym problem, "
             "then infers (default: predefined)",
    )
    parser.add_argument(
        "--problem-index", type=int, default=None,
        help="0-based problem position to walk from in natural (numeric) order "
             "(generate mode only): 0 -> problem1, 1 -> problem2, etc. — matches "
             "legacy ordering. Defaults to config default_problem_index.",
    )
    parser.add_argument(
        "--num-problems", type=int, default=None,
        help="Number of problems to generate (generate mode only, default from config)",
    )
    parser.add_argument(
        "--length-min", type=int, default=None,
        help="Min window length in steps (generate mode only, default from config)",
    )
    parser.add_argument(
        "--length-max", type=int, default=None,
        help="Max window length in steps (generate mode only, default from config)",
    )
    parser.add_argument(
        "--skip", type=int, default=None,
        help="States discarded between windows (generate mode only, default from config)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for reproducibility (generate mode only)",
    )
    parser.add_argument(
        "--no-inference", action="store_true",
        help="Generate folders only, skip LLM inference (generate mode only)",
    )
    parser.add_argument(
        "--output-dir-name", type=str, default=None,
        help="Custom name for the output directory. Overrides the auto-generated "
             "'multi_problem_<timestamp>__model=<model>...' name.",
    )

    args = parser.parse_args()

    if args.gen_mode == "generate":
        generate_trajectories_via_generation(
            domain=args.domain,
            output_base_dir=Path(__file__).parent / "data",
            vendor=args.vendor,
            problem_index=args.problem_index,
            num_problems=args.num_problems,
            length_min=args.length_min,
            length_max=args.length_max,
            skip=args.skip,
            seed=args.seed,
            run_inference=not args.no_inference,
            output_dir_name=args.output_dir_name,
        )
    else:
        generate_trajectories(
            domain=args.domain,
            output_base_dir=Path(__file__).parent / "data",
            num_steps=args.num_steps,
            vendor=args.vendor,
            start_index=args.start_index,
            planner=args.planner,
            problem_start=args.problem_start,
            problem_end=args.problem_end,
            output_dir_name=args.output_dir_name,
        )
