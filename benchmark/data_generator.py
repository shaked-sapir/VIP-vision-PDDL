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
from typing import TYPE_CHECKING, Callable, List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trajectory_handlers.frame_classification import (
    ConcurrentFrameClassifier,
    SequentialFrameClassifier,
)
from src.trajectory_handlers.llm_blocks_trajectory_handler import LLMBlocksImageTrajectoryHandler
from src.trajectory_handlers.llm_npuzzle_trajectory_handler import LLMNpuzzleImageTrajectoryHandler
from src.trajectory_handlers.llm_hanoi_trajectory_handler import LLMHanoiImageTrajectoryHandler
from src.trajectory_handlers.llm_hiking_trajectory_handler import LLMHikingImageTrajectoryHandler
from src.trajectory_handlers.llm_maze_trajectory_handler import LLMMazeImageTrajectoryHandler
from src.trajectory_handlers.llm_depot_trajectory_handler import LLMDepotImageTrajectoryHandler
from src.trajectory_handlers.llm_gripper_trajectory_handler import LLMGripperImageTrajectoryHandler
from src.trajectory_handlers.pddlgym_problem_generator import PDDLGymProblemGenerator
from src.trace_generation.sources import (
    DEFAULT_MAX_PLANNING_TIME,
    DEFAULT_MAX_RANDOM_TRIALS,
    DEFAULT_MAX_REPLANNING_TIME,
    DEFAULT_MAX_STEPS,
    WalkConfig,
)
from src.utils.config import load_config
from src.utils.pddl_trajectory import ensure_trajectory_json, extract_actions_from_trajectory_json

if TYPE_CHECKING:
    from src.trace_generation.corpus import Corpus


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

def _run_generation_inference(handler, problem_name: str, problem_dir: Path,
                              gt_root: Path) -> None:
    """Run external-style inference over a freshly-generated PDDLGym problem folder.

    The generated folder already contains GT (_trajectory.json) + images + .pddl,
    so we treat it exactly like an external (depot/gripper) problem: extract the
    ground actions from the GT JSON and run the LLM classification pipeline.

    The generator writes the .pddl and _trajectory.json in the raw gym schema
    (its _GymWalkHandler has no schema-translation hook). Inference produces a
    .trajectory already in the eval schema (via _rename_ground_action + the
    classifiers).

    Before we translate the .pddl and rebuild the (noisy) _trajectory.json from
    the classifier .trajectory, we export the *true* GT — schema-translated to
    the eval schema — into ``gt_root/{problem}/``. This preserves ground truth as
    a first-class parallel artifact; without it the raw GT JSON would be deleted
    and gt_trajectories/ would silently contain classifier output.
    """
    from benchmark.experiment_running_helpers.gt_builder import export_gt_from_problem_dir

    ensure_trajectory_json(problem_dir)
    actions = extract_actions_from_trajectory_json(problem_dir)
    handler._set_seq_idx_format(problem_dir)
    handler.create_trajectory_and_masks(problem_name, actions, problem_dir)

    # Capture GT (raw gym schema -> eval schema) BEFORE the rebuild overwrites it.
    export_gt_from_problem_dir(
        problem_dir, gt_root, problem_name,
        handler=handler, needs_schema_translation=True,
    )

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

    Raises:
        KeyError: No explicit index and no ``generation.from_pddlgym.default_problem_index``.
    """
    if problem_index is not None:
        return problem_index
    pddlgym_cfg = domain_config.get("generation", {}).get("from_pddlgym", {})
    if "default_problem_index" not in pddlgym_cfg:
        raise KeyError(
            "No problem index given and 'default_problem_index' is not set under "
            "domains.<domain>.generation.from_pddlgym in config.yaml. Pass "
            "--problem-index or add the key.")
    return pddlgym_cfg["default_problem_index"]


# ── Domain registry ──────────────────────────────────────────────────────

def build_frame_classifier(inference_workers: int):
    """The frame-classification strategy for a run.

    Args:
        inference_workers: ``1`` keeps one VLM call in flight at a time, which
            is the original behaviour and the default. Higher issues a window's
            frames together.

    Returns:
        A :class:`FrameClassifier`.
    """
    if inference_workers <= 1:
        return SequentialFrameClassifier()
    return ConcurrentFrameClassifier(max_workers=inference_workers)


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

_REGISTERED_DOMAINS = sorted(_DOMAIN_REGISTRY)


def _natural_sort_key(path: Path):
    """Sort key that orders problem1, problem2, ..., problem10 numerically."""
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', path.name)]


def _collect_problem_dirs(problems_dir: Path) -> List[Path]:
    """Collect problem subdirectories from the unified layout.

    Each problem is a subdirectory of problems_dir containing at minimum
    a .pddl file (and for external domains, images + GT trajectory).

    Raises:
        FileNotFoundError: ``problems_dir`` does not exist, or holds no problem
            subdirectory.
    """
    if not problems_dir.is_dir():
        raise FileNotFoundError(
            f"--gen-mode predefined needs a problems directory at "
            f"{problems_dir.resolve()}, which does not exist. Create it with one "
            f"subdirectory per problem (problem0/problem0.pddl, ...), point "
            f"domains.<domain>.problems_dir in config.yaml at an existing "
            f"directory, or use --gen-mode generate/trace instead.")

    found = sorted(
        [d for d in problems_dir.iterdir() if d.is_dir() and not d.name.startswith(('.', '_'))],
        key=_natural_sort_key,
    )
    if not found:
        raise FileNotFoundError(
            f"--gen-mode predefined found no problem subdirectory in "
            f"{problems_dir.resolve()}. Each problem is its own subdirectory "
            f"holding at least a .pddl file (problem0/problem0.pddl, ...).")
    return found


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
    inference_workers: int = 1,
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
    gt_root = experiment_dir / "gt_trajectories"
    for problem_idx, problem_dir in enumerate(generated_dirs):
        problem_name = problem_dir.name
        print(f"[{problem_idx + 1}/{len(generated_dirs)}] Inferring {problem_name}...")
        try:
            handler = registry["handler_class"](
                domain_name=gym_domain_name,
                pddl_domain_file=domain_file,
                vendor=vendor,
                frame_classifier=build_frame_classifier(inference_workers),
            )
            _run_generation_inference(handler, problem_name, problem_dir, gt_root)
            _apply_transform(registry["transform_fn"], problem_dir)
            _cleanup_external_source_files(problem_dir)
            print(f"  ✓ {problem_name}")
            print()
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            print()
            continue

    # GT trajectories are exported per-problem inside _run_generation_inference
    # (before the noisy rebuild), so no separate post-pass is needed.

    print("=" * 80)
    print("GENERATION-MODE TRAJECTORY GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nExperiment saved to: {experiment_dir}")
    print(f"  Trajectories: {trajectories_dir}")
    print(f"  GT trajectories: {gt_root}")
    print()
    return trajectories_dir


# ── Trace-mode entry point (symbolic corpus, no images and no LLM) ───────

def _resolve_trace_domain_file(domain: Optional[str],
                               domain_file: Optional[Path]) -> Path:
    """The domain PDDL to trace against, given explicitly or via the registry.

    Args:
        domain: A key of ``_DOMAIN_REGISTRY``, or ``None``.
        domain_file: An explicit path, which bypasses the registry entirely.

    Raises:
        ValueError: Neither was given, or ``domain`` is not registered.
        FileNotFoundError: The resolved path does not exist, whichever branch
            produced it.
    """
    if domain_file is None:
        if domain is None:
            raise ValueError(
                "Trace mode needs a domain: pass --domain-file for any PDDL domain, "
                f"or --domain to use one of {_REGISTERED_DOMAINS}.")
        if domain not in _DOMAIN_REGISTRY:
            raise ValueError(
                f"Domain {domain!r} is not registered, so its domain file cannot be "
                f"looked up. Pass --domain-file to point at the PDDL directly, or "
                f"use one of {_REGISTERED_DOMAINS}.")
        domain_file = (load_config()["domains"]
                       [_DOMAIN_REGISTRY[domain]["config_key"]]["domain_file"])

    domain_file = Path(domain_file)
    if not domain_file.is_file():
        raise FileNotFoundError(f"No domain file at {domain_file}.")
    return domain_file


def _describe_cut(mode_name: str, asked: Optional[int], skip: int,
                  length_range: Optional[tuple]) -> str:
    """One banner line for the cut settings the mode actually reads."""
    if length_range is None:
        return f"{mode_name} | the whole trace as one problem"
    return (f"{mode_name} | problems {asked} | skip {skip} | "
            f"length {length_range[0]}-{length_range[1]}")


def _describe_problem_count(produced: int, asked: Optional[int]) -> str:
    """One banner phrase for the produced problem count, naming any shortfall."""
    if asked is None or produced >= asked:
        return str(produced)
    return f"{produced}  *** SHORT: asked for {asked} ***"


def _trace_dir_name(timestamp: str, source_kind: str, source_file: Path,
                    mode_name: str, asked: Optional[int]) -> str:
    """The corpus directory name, carrying only the settings the mode reads."""
    count_tag = f"__n={asked}" if asked is not None else ""
    return (f"trace_{timestamp}__{source_kind}={Path(source_file).stem}"
            f"__cut={mode_name}{count_tag}")


def _print_trace_header(label: str, corpus_root: Path, source_kind: str,
                        source_file: Path, cut_description: str) -> None:
    """Announce what is about to be generated and where it will land."""
    print("=" * 80)
    print(f"GENERATING {label.upper()} PROBLEMS (trace mode)")
    print(f"Corpus: {corpus_root}")
    print(f"Source: {source_kind} | {source_file}")
    print(f"Cut: {cut_description}")
    print("=" * 80)
    print()


def _print_trace_summary(corpus: "Corpus") -> None:
    """Report what was written, naming any shortfall against the request."""
    print("=" * 80)
    print("TRACE-MODE TRAJECTORY GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nCorpus saved to: {corpus.root}")
    print("  Problems: "
          f"{_describe_problem_count(corpus.num_problems, corpus.problems_requested)}")
    print(f"  Trajectories: {corpus.trajectories_dir}")
    print(f"  GT trajectories: {corpus.root / 'gt_trajectories'}")
    print(f"  Manifest: {corpus.info_file}")
    print()


def generate_trajectories_via_trace(
    output_base_dir: Path,
    source_kind: str,
    problem_file: Path,
    domain: Optional[str] = None,
    domain_file: Optional[Path] = None,
    trajectory_file: Optional[Path] = None,
    walk: Optional[WalkConfig] = None,
    render: bool = False,
    cut_mode: str = "uniform",
    num_problems: int = 10,
    length_min: int = 9,
    length_max: int = 20,
    skip: int = 1,
    cut_seed: Optional[int] = None,
    problem_prefix: str = "problem",
    output_dir_name: Optional[str] = None,
) -> Path:
    """Build a symbolic corpus by cutting one trace into problem folders.

    Walks a PDDL problem or replays an existing ``.trajectory``, cuts the result
    into windows and writes each as a problem folder plus its ground truth. No
    images and no LLM are involved, so the states are exact.

    Every setting is a plain argument with a default; nothing is read from a
    ``generation`` block in ``config.yaml``. The corpus records what it was given
    in its own ``generation_info.json``. ``config.yaml`` is consulted only to look
    up ``domain``'s PDDL file, and not at all when ``domain_file`` is passed.

    Args:
        output_base_dir: Base directory for benchmark data (e.g. benchmark/data/).
        source_kind: "problem" to walk, "trajectory" to replay.
        problem_file: The problem to walk, or the replayed trajectory's problem.
        domain: Registry key naming the domain PDDL, and the output subdirectory.
            Optional when ``domain_file`` is given, in which case the subdirectory
            falls back to the domain's own name.
        domain_file: The domain PDDL. Bypasses the registry, so any PDDL domain
            works whether or not this repo knows about it.
        trajectory_file: The trajectory to replay ("trajectory" kind only).
        walk: The walk's settings ("problem" kind only). Defaults to an unseeded
            random walk.
        render: Copy each window's frames out as state_*.png.
        cut_mode: "none" or "uniform".
        num_problems: Cap on the number of problem folders.
        length_min: Min window length, for cut_mode="uniform".
        length_max: Max window length, for cut_mode="uniform".
        skip: States discarded between windows.
        cut_seed: Window-length RNG seed. Defaults to the walk's seed.
        problem_prefix: Problem folder name prefix; folders are numbered from 0.
        output_dir_name: Override for the output directory name.

    Returns:
        Path to the trajectories directory.
    """
    from src.trace_generation.corpus import SourceKind, build_source, generate_corpus
    from src.trace_generation.cutter import CutMode

    kind = SourceKind(source_kind)
    mode = CutMode(cut_mode)
    walk = walk if walk is not None else WalkConfig()

    source = build_source(
        kind,
        domain_file=_resolve_trace_domain_file(domain, domain_file),
        problem_file=Path(problem_file),
        trajectory_file=Path(trajectory_file) if trajectory_file else None,
        walk=walk,
        attach_frames=render and kind is SourceKind.TRAJECTORY,
    )

    # NONE takes the whole trace as one window, so it reads neither setting.
    length_range = (length_min, length_max) if mode is CutMode.UNIFORM else None
    asked = num_problems if mode is not CutMode.NONE else None
    label = domain or source.domain_name

    source_file = trajectory_file if kind is SourceKind.TRAJECTORY else problem_file
    timestamp = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
    auto_name = _trace_dir_name(timestamp, kind.value, Path(source_file),
                                mode.value, asked)
    corpus_root = output_base_dir / label / (output_dir_name or auto_name)

    _print_trace_header(label, corpus_root, kind.value, source_file,
                        _describe_cut(mode.value, asked, skip, length_range))

    corpus = generate_corpus(
        source,
        corpus_root=corpus_root,
        cut_mode=mode,
        length_range=length_range,
        skip=skip,
        num_problems=asked,
        seed=cut_seed if cut_seed is not None else walk.seed,
        problem_prefix=problem_prefix,
        render=render,
        extra_info={"domain": label},
    )

    _print_trace_summary(corpus)
    return corpus.trajectories_dir


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
    inference_workers: int = 1,
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
    gt_root = experiment_dir / "gt_trajectories"
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
                frame_classifier=build_frame_classifier(inference_workers),
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

            # Export GT to gt_trajectories/ from the retained (eval-schema) GT
            # JSON. For predefined/external the JSON is already GT + eval-schema
            # (never rebuilt from the classifier), so no schema translation.
            from benchmark.experiment_running_helpers.gt_builder import export_gt_from_problem_dir
            export_gt_from_problem_dir(
                output_problem_dir, gt_root, problem_name,
                needs_schema_translation=False,
            )

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
        "--domain", type=str, default=None,
        help="Domain to generate data for. Required, and one of "
             f"{_REGISTERED_DOMAINS}, for --gen-mode predefined/generate. "
             "In trace mode it is optional and only names the output "
             "subdirectory; pass --domain-file to trace a domain this repo does "
             "not know about.",
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
        "--gen-mode", type=str, default="predefined",
        choices=["predefined", "generate", "trace"],
        help="'predefined' runs the LLM pipeline over pre-authored problem files; "
             "'generate' creates new problems + images from a bundled PDDLGym problem, "
             "then infers; 'trace' cuts one symbolic trace into problems, with no "
             "images and no LLM (default: predefined)",
    )
    parser.add_argument(
        "--problem-index", type=int, default=None,
        help="0-based problem position to walk from in natural (numeric) order "
             "(generate mode only): 0 -> problem1, 1 -> problem2, etc. — matches "
             "legacy ordering. Defaults to config generation.from_pddlgym."
             "default_problem_index.",
    )
    parser.add_argument(
        "--num-problems", type=int, default=None,
        help="Number of problems to generate (generate/trace modes; default from "
             "config in generate mode, 10 in trace mode)",
    )
    parser.add_argument(
        "--length-min", type=int, default=None,
        help="Min window length in steps (generate/trace modes; default from "
             "config in generate mode, 9 in trace mode)",
    )
    parser.add_argument(
        "--length-max", type=int, default=None,
        help="Max window length in steps (generate/trace modes; default from "
             "config in generate mode, 20 in trace mode)",
    )
    parser.add_argument(
        "--skip", type=int, default=None,
        help="States discarded between windows (generate/trace modes; default "
             "from config in generate mode, 1 in trace mode)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Walk RNG seed for reproducibility (generate/trace modes)",
    )
    parser.add_argument(
        "--inference-workers", type=int, default=1, metavar="N",
        help="VLM calls in flight per window. 1 (the default) is one at a time, "
             "the original behaviour. Higher issues a window's frames together; "
             "measured ~7x faster at 12, with results still assembled in frame "
             "order. A frame that fails every attempt abandons its window.",
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

    # ── Trace-mode args (--gen-mode trace) ───────────────────────────────
    #
    # Trace mode reads no 'generation' block from config.yaml: every setting is
    # a flag with a default, and generation_info.json records what was used.
    parser.add_argument(
        "--domain-file", type=Path, default=None,
        help="The domain PDDL to trace against (trace mode only). Bypasses the "
             "domain registry and config.yaml, so any PDDL domain works.",
    )
    parser.add_argument(
        "--source-kind", type=str, default=None, choices=["problem", "trajectory"],
        help="'problem' walks a PDDL problem, 'trajectory' replays an existing "
             ".trajectory (trace mode only, required)",
    )
    parser.add_argument(
        "--problem-file", type=Path, default=None,
        help="The problem to walk, or the replayed trajectory's problem "
             "(trace mode only, required)",
    )
    parser.add_argument(
        "--trajectory-file", type=Path, default=None,
        help="The .trajectory to replay (trace mode with --source-kind trajectory)",
    )
    parser.add_argument(
        "--backend", type=str, default="native", choices=["native", "pddlgym"],
        help="Walk backend (trace mode with --source-kind problem, default: native)",
    )
    parser.add_argument(
        "--p-rnd", type=float, default=1.0,
        help="Probability of substituting a random applicable action for the planned "
             "one; 1.0 skips the planner entirely (trace mode, default: 1.0)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS,
        help=f"Cap on walked transitions (trace mode, default: {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument(
        "--preserve-solvability", action="store_true",
        help="Keep only random actions that leave the problem solvable "
             "(trace mode with --source-kind problem)",
    )
    parser.add_argument(
        "--stop-at-goal", action="store_true",
        help="Stop the walk as soon as the plan completes, instead of replanning "
             "(trace mode with --source-kind problem)",
    )
    parser.add_argument(
        "--max-planning-time", type=int, default=DEFAULT_MAX_PLANNING_TIME,
        help="Planner timeout in seconds (trace mode, default: "
             f"{DEFAULT_MAX_PLANNING_TIME})",
    )
    parser.add_argument(
        "--max-replanning-time", type=int, default=DEFAULT_MAX_REPLANNING_TIME,
        help="Solvability-check timeout in seconds, used by "
             f"--preserve-solvability (trace mode, default: {DEFAULT_MAX_REPLANNING_TIME})",
    )
    parser.add_argument(
        "--max-random-trials", type=int, default=DEFAULT_MAX_RANDOM_TRIALS,
        help="Random actions tried per substitution under --preserve-solvability "
             f"(trace mode, default: {DEFAULT_MAX_RANDOM_TRIALS})",
    )
    parser.add_argument(
        "--cut-mode", type=str, default="uniform", choices=["none", "uniform"],
        help="How the trace is split into problems (trace mode, default: uniform)",
    )
    parser.add_argument(
        "--cut-seed", type=int, default=None,
        help="Window-length RNG seed, kept separate from the walk's. "
             "Defaults to --seed (trace mode only)",
    )
    parser.add_argument(
        "--problem-prefix", type=str, default="problem",
        help="Problem folder name prefix; folders are numbered from 0 "
             "(trace mode, default: problem)",
    )
    parser.add_argument(
        "--render", action=argparse.BooleanOptionalAction, default=False,
        help="Copy each window's frames out as state_*.png (trace mode, "
             "default: off)",
    )

    args = parser.parse_args()

    if args.gen_mode != "trace" and args.domain not in _DOMAIN_REGISTRY:
        parser.error(f"--gen-mode {args.gen_mode} requires --domain, one of "
                     f"{_REGISTERED_DOMAINS}.")

    if args.gen_mode == "trace":
        if args.source_kind is None:
            parser.error("--gen-mode trace requires --source-kind.")
        if args.problem_file is None:
            parser.error("--gen-mode trace requires --problem-file.")
        # These four are shared with generate mode, which needs None to mean
        # "fall back to config"; trace mode uses its own defaults instead.
        cut_args = {}
        if args.num_problems is not None:
            cut_args["num_problems"] = args.num_problems
        if args.length_min is not None:
            cut_args["length_min"] = args.length_min
        if args.length_max is not None:
            cut_args["length_max"] = args.length_max
        if args.skip is not None:
            cut_args["skip"] = args.skip

        generate_trajectories_via_trace(
            output_base_dir=Path(__file__).parent / "data",
            source_kind=args.source_kind,
            problem_file=args.problem_file,
            domain=args.domain,
            domain_file=args.domain_file,
            trajectory_file=args.trajectory_file,
            walk=WalkConfig(
                backend=args.backend,
                p_rnd=args.p_rnd,
                seed=args.seed,
                max_steps=args.max_steps,
                preserve_solvability=args.preserve_solvability,
                stop_at_goal=args.stop_at_goal,
                max_planning_time=args.max_planning_time,
                max_replanning_time=args.max_replanning_time,
                max_random_trials=args.max_random_trials,
            ),
            render=args.render,
            cut_mode=args.cut_mode,
            cut_seed=args.cut_seed,
            problem_prefix=args.problem_prefix,
            output_dir_name=args.output_dir_name,
            **cut_args,
        )
    elif args.gen_mode == "generate":
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
            inference_workers=args.inference_workers,
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
            inference_workers=args.inference_workers,
        )
