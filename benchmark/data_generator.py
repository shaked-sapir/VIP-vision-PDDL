"""
Data Generator for Benchmark System

One generation path for every domain: a trace (a walk of a PDDL problem, or a
replayed ``.trajectory``) is cut into windows and emitted as problem folders;
``--render`` wraps the source in a renderer so each folder also gets its
``state_*.png`` frames; VLM inference then runs over the emitted folders exactly
as it does over externally-sourced ones.

Modes:
    trace       cut one trace into problems; symbolic unless --render
    generate    trace mode with a random walk, --render and inference (alias)
    predefined  one walk per pre-authored problem folder, rendered and inferred;
                external domains copy their shipped frames instead of walking

Output structure (``src/utils/corpus_layout.py``):
    benchmark/data/<domain>/<experiment>/training/trajectories/problemN/
        problemN.pddl, plan.txt, problemN_trajectory.json          (symbolic pool)
    benchmark/data/<domain>/<experiment>/training/image_trajectories/problemN/
        problemN.pddl, problemN_trajectory.json, state_*.png,      (image pool)
        problemN.trajectory + .masking_info   (inferred, after inference)
    benchmark/data/<domain>/<experiment>/gt_trajectories/problemN/
    benchmark/data/<domain>/<experiment>/generation_info.json
"""

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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
from src.trace_generation.sources import (
    DEFAULT_MAX_PLANNING_TIME,
    DEFAULT_MAX_RANDOM_TRIALS,
    DEFAULT_MAX_REPLANNING_TIME,
    DEFAULT_MAX_STEPS,
    TraceSource,
    WalkConfig,
)
from src.utils.config import load_config
from src.utils.corpus_layout import GT_SUBPATH, IMAGE_POOL_SUBPATH, POOL_SUBPATH

if TYPE_CHECKING:
    from src.trace_generation.corpus import Corpus


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
        "is_external": False,
    },
    "npuzzle": {
        "display_name": "N-PUZZLE",
        "config_key": "npuzzle",
        "handler_class": LLMNpuzzleImageTrajectoryHandler,
        "is_external": False,
    },
    "hanoi": {
        "display_name": "HANOI",
        "config_key": "hanoi",
        "handler_class": LLMHanoiImageTrajectoryHandler,
        "is_external": False,
    },
    "hiking": {
        "display_name": "HIKING",
        "config_key": "hiking",
        "handler_class": LLMHikingImageTrajectoryHandler,
        "is_external": False,
    },
    "maze": {
        "display_name": "MAZE",
        "config_key": "maze",
        "handler_class": LLMMazeImageTrajectoryHandler,
        "is_external": False,
    },
    "depot": {
        "display_name": "DEPOT",
        "config_key": "depot",
        "handler_class": LLMDepotImageTrajectoryHandler,
        "is_external": True,
    },
    "gripper": {
        "display_name": "GRIPPER",
        "config_key": "gripper",
        "handler_class": LLMGripperImageTrajectoryHandler,
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


# ── Rendering ────────────────────────────────────────────────────────────

def _rendered(source: TraceSource, domain: Optional[str],
              visualizer_root: Optional[Path]) -> TraceSource:
    """Wrap ``source`` in the renderer registered for ``domain``."""
    from src.trace_generation.frame_renderer import RenderedSource
    from src.trace_generation.renderer_registry import RENDERABLE_DOMAINS, build_renderer

    if domain is None:
        raise ValueError(
            f"--render needs --domain to pick a renderer, one of {RENDERABLE_DOMAINS}.")
    return RenderedSource(source, build_renderer(domain, visualizer_root=visualizer_root))


# ── Inference over emitted problem folders ───────────────────────────────

def _build_handler(domain: str, vendor: str, inference_workers: int):
    """The domain's inference handler, configured from ``config.yaml``."""
    registry = _DOMAIN_REGISTRY[domain]
    domain_config = load_config()["domains"][registry["config_key"]]
    return registry["handler_class"](
        domain_name=domain_config.get("gym_domain_name", registry["config_key"]),
        pddl_domain_file=Path(domain_config["domain_file"]),
        vendor=vendor,
        frame_classifier=build_frame_classifier(inference_workers),
    )


def run_inference(domain: str, image_trajectories_dir: Path, *, vendor: str = "openai",
                  inference_workers: int = 1) -> List[Path]:
    """Run VLM inference over every ``problemN/`` folder of an image pool.

    Each folder must hold ``state_*.png``, its ``.pddl`` and its GT
    ``_trajectory.json``, which is what the emitter writes with ``render=True``.
    The inferred ``.trajectory`` and ``.masking_info`` land in the same folder.
    A folder whose inference fails is reported and skipped.

    Returns:
        The folders inferred successfully.
    """
    problem_dirs = _collect_problem_dirs(image_trajectories_dir)
    done: List[Path] = []
    for index, problem_dir in enumerate(problem_dirs):
        name = problem_dir.name
        print(f"[{index + 1}/{len(problem_dirs)}] Inferring {name}...")
        try:
            handler = _build_handler(domain, vendor, inference_workers)
            handler.run_pipeline(problem_name=name, images_path=problem_dir)
            done.append(problem_dir)
            print(f"  ✓ {name}\n")
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")
    return done


# ── Trace-mode entry point ───────────────────────────────────────────────

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
    if corpus.image_trajectories_dir is not None:
        print(f"  Image trajectories: {corpus.image_trajectories_dir}")
    print(f"  GT trajectories: {corpus.root / GT_SUBPATH}")
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
    visualizer_root: Optional[Path] = None,
    cut_mode: str = "uniform",
    num_problems: int = 10,
    length_min: int = 9,
    length_max: int = 20,
    skip: int = 1,
    cut_seed: Optional[int] = None,
    problem_prefix: str = "problem",
    output_dir_name: Optional[str] = None,
) -> "Corpus":
    """Build a corpus by cutting one trace into problem folders.

    Walks a PDDL problem or replays an existing ``.trajectory``, cuts the result
    into windows and writes each as a problem folder plus its ground truth. With
    ``render`` the source is wrapped in the domain's :class:`FrameRenderer`, so
    every folder also holds ``state_*.png``; no LLM is involved either way.

    Every setting is a plain argument with a default; nothing is read from a
    ``generation`` block in ``config.yaml``. The corpus records what it was given
    in its own ``generation_info.json``. ``config.yaml`` is consulted only to look
    up ``domain``'s PDDL file, and not at all when ``domain_file`` is passed.

    Args:
        output_base_dir: Base directory for benchmark data (e.g. benchmark/data/).
        source_kind: "problem" to walk, "trajectory" to replay.
        problem_file: The problem to walk, or the replayed trajectory's problem.
        domain: Registry key naming the domain PDDL, the renderer and the output
            subdirectory. Optional when ``domain_file`` is given and ``render``
            is off, in which case the subdirectory falls back to the domain's
            own name.
        domain_file: The domain PDDL. Bypasses the registry, so any PDDL domain
            works whether or not this repo knows about it.
        trajectory_file: The trajectory to replay ("trajectory" kind only).
        walk: The walk's settings ("problem" kind only). Defaults to an unseeded
            random walk.
        render: Render each state and write the frames as state_*.png.
        visualizer_root: The planning-visualizer checkout, for depot and gripper
            frames. Read from ``config.yaml`` when omitted.
        cut_mode: "none" or "uniform".
        num_problems: Cap on the number of problem folders.
        length_min: Min window length, for cut_mode="uniform".
        length_max: Max window length, for cut_mode="uniform".
        skip: States discarded between windows.
        cut_seed: Window-length RNG seed. Defaults to the walk's seed.
        problem_prefix: Problem folder name prefix; folders are numbered from 0.
        output_dir_name: Override for the output directory name.

    Returns:
        The written corpus; ``image_trajectories_dir`` is set when rendering.
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
    )
    if render:
        source = _rendered(source, domain, visualizer_root)

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
    return corpus


# ── Generate mode: a rendered random walk, then inference ────────────────

def generate_trajectories_via_generation(
    domain: str,
    output_base_dir: Path,
    problem_file: Path,
    vendor: str = "openai",
    num_problems: int = 10,
    length_min: int = 9,
    length_max: int = 20,
    skip: int = 1,
    seed: Optional[int] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    run_inference_after: bool = True,
    output_dir_name: Optional[str] = None,
    inference_workers: int = 1,
    visualizer_root: Optional[Path] = None,
) -> Path:
    """Random-walk ``problem_file``, cut, render, then infer: trace mode plus VLM.

    Args:
        domain: Domain key from _DOMAIN_REGISTRY.
        output_base_dir: Base directory for benchmark data (e.g. benchmark/data/).
        problem_file: The eval-schema problem whose object universe the walk uses.
        vendor: LLM vendor — "openai" or "google".
        num_problems: Number of problems to generate.
        length_min: Min window length in steps.
        length_max: Max window length in steps.
        skip: States discarded between windows.
        seed: Walk and cut RNG seed.
        max_steps: Cap on walked transitions.
        run_inference_after: If False, only generate folders (skip LLM).
        output_dir_name: Override for the output directory name.
        inference_workers: VLM calls in flight per window.
        visualizer_root: The planning-visualizer checkout, for depot and gripper.

    Returns:
        Path to the image pool the inference ran over.
    """
    model_name = load_config()[vendor].get("fluent_classification_model", {}).get(
        "model_name", vendor)
    timestamp = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
    auto_name = (f"multi_problem_{timestamp}__model={model_name}"
                 f"__gen__prob={Path(problem_file).stem}__len={length_min}-{length_max}")

    corpus = generate_trajectories_via_trace(
        output_base_dir=output_base_dir,
        source_kind="problem",
        problem_file=Path(problem_file),
        domain=domain,
        walk=WalkConfig(p_rnd=1.0, seed=seed, max_steps=max_steps),
        render=True,
        visualizer_root=visualizer_root,
        cut_mode="uniform",
        num_problems=num_problems,
        length_min=length_min,
        length_max=length_max,
        skip=skip,
        output_dir_name=output_dir_name or auto_name,
    )
    image_dir = corpus.image_trajectories_dir
    if image_dir is None:
        raise ValueError(f"{corpus.root} was written without frames; nothing to infer.")
    if not run_inference_after:
        print("--no-inference: folders generated, LLM inference skipped.")
        return image_dir

    run_inference(domain, image_dir, vendor=vendor, inference_workers=inference_workers)
    print(f"Experiment saved to: {corpus.root}")
    return image_dir


# ── Predefined mode: one walk (or copy) per pre-authored problem ─────────

def _walk_config(num_steps: int, planner: Optional[str], seed: Optional[int]) -> WalkConfig:
    """A random walk of ``num_steps``, or a Fast Downward plan that stops at the goal."""
    if planner:
        return WalkConfig(p_rnd=0.0, seed=seed, max_steps=num_steps, stop_at_goal=True)
    return WalkConfig(p_rnd=1.0, seed=seed, max_steps=num_steps)


def _emit_walked_problem(problem_name: str, index: int, problem_pddl: Path,
                         domain_file: Path, walk: WalkConfig, experiment_dir: Path,
                         domain: str, visualizer_root: Optional[Path]):
    """Walk one problem, render it and write it as ``problemN/`` plus its GT."""
    from src.trace_generation.cutter import CutMode, cut
    from src.trace_generation.emitter import emit_window
    from src.trace_generation.sources import ProblemWalkSource

    source = _rendered(ProblemWalkSource(problem_pddl, domain_file, walk=walk),
                       domain, visualizer_root)
    windows = cut(source.steps(), mode=CutMode.NONE)
    if not windows:
        raise ValueError(f"{problem_pddl} yielded no transition to emit.")
    return emit_window(windows[0], problem_name=problem_name, index=index,
                       domain_name=source.domain_name, corpus_root=experiment_dir,
                       render=True, source=str(problem_pddl))


def _copy_external_problem(source_problem_dir: Path, experiment_dir: Path) -> Path:
    """Copy a shipped problem folder (frames, GT, pddl) into both pools and export its GT."""
    from benchmark.experiment_running_helpers.gt_builder import export_gt_from_problem_dir
    from src.trace_generation.emitter import copy_problem_files
    from src.utils.pddl_trajectory import ensure_trajectory_json

    name = source_problem_dir.name
    image_dir = experiment_dir / IMAGE_POOL_SUBPATH / name
    shutil.copytree(source_problem_dir, image_dir, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("plan.txt"))
    ensure_trajectory_json(image_dir)
    export_gt_from_problem_dir(image_dir, experiment_dir / GT_SUBPATH,
                               name, needs_schema_translation=False)

    problem_dir = experiment_dir / POOL_SUBPATH / name
    problem_dir.mkdir(parents=True, exist_ok=True)
    copy_problem_files(image_dir, problem_dir)
    plan = source_problem_dir / "plan.txt"
    if plan.exists():
        shutil.copy(plan, problem_dir / plan.name)
    return image_dir


def generate_trajectories(
    domain: str,
    output_base_dir: Path,
    num_steps: int = 100,
    vendor: str = "openai",
    planner: Optional[str] = None,
    seed: Optional[int] = None,
    problem_start: Optional[int] = None,
    problem_end: Optional[int] = None,
    output_dir_name: Optional[str] = None,
    inference_workers: int = 1,
    visualizer_root: Optional[Path] = None,
    run_inference_after: bool = True,
) -> Path:
    """Generate trajectories for every pre-authored problem of a domain.

    PDDLGym-rendered domains walk each ``problems_dir/problemN/problemN.pddl``
    (random, or a Fast Downward plan with ``planner``), render the states and
    write the folder; external domains copy their shipped frames and GT. Then
    inference runs over every folder.

    Args:
        domain: Domain key from _DOMAIN_REGISTRY.
        output_base_dir: Base directory for benchmark data (e.g. benchmark/data/).
        num_steps: Max trajectory steps (walked domains only, default 100).
        vendor: LLM vendor — "openai" or "google" (default "openai").
        planner: "ff" or "fd" to follow a Fast Downward plan instead of a random
            walk (the walk's planner is always Fast Downward).
        seed: Walk RNG seed.
        problem_start: First problem index to process (0-based, inclusive).
        problem_end: Last problem index to process (0-based, inclusive).
        output_dir_name: Override for the output directory name.
        inference_workers: VLM calls in flight per window.
        visualizer_root: The planning-visualizer checkout, for depot and gripper.
        run_inference_after: If False, only generate folders (skip LLM).

    Returns:
        Path to the image pool.
    """
    from src.trace_generation.emitter import build_generation_info, write_generation_info

    registry = _DOMAIN_REGISTRY[domain]
    is_external = registry["is_external"]

    config = load_config()
    domain_config = config["domains"][registry["config_key"]]
    domain_file = Path(domain_config["domain_file"])
    problems_dir = Path(domain_config["problems_dir"])
    model_name = config[vendor].get("fluent_classification_model", {}).get("model_name", vendor)

    timestamp = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
    auto_name = f"multi_problem_{timestamp}__model={model_name}__steps={num_steps}"
    if planner:
        auto_name += f"__planner={planner}"
    experiment_name = output_dir_name if output_dir_name is not None else auto_name

    print("=" * 80)
    print(f"GENERATING {registry['display_name']} TRAJECTORIES")
    print(f"Experiment: {experiment_name}")
    print(f"Mode: {'External images' if is_external else ('planned walk' if planner else 'random walk')}")
    print("=" * 80)
    print()

    experiment_dir = output_base_dir / domain / experiment_name
    trajectories_dir = experiment_dir / POOL_SUBPATH
    image_trajectories_dir = experiment_dir / IMAGE_POOL_SUBPATH
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    image_trajectories_dir.mkdir(parents=True, exist_ok=True)

    problem_dirs = _collect_problem_dirs(problems_dir)
    print(f"Found {len(problem_dirs)} problems in {problems_dir}")
    if problem_start is not None or problem_end is not None:
        start = problem_start if problem_start is not None else 0
        end = (problem_end + 1) if problem_end is not None else len(problem_dirs)
        problem_dirs = problem_dirs[start:end]
        print(f"Filtered to {len(problem_dirs)} problems (indices {start}–{end - 1})")
    print()

    walk = _walk_config(num_steps, planner, seed)
    emitted = []
    for index, source_problem_dir in enumerate(problem_dirs):
        name = source_problem_dir.name
        print(f"[{index + 1}/{len(problem_dirs)}] Preparing {name}...")
        try:
            if is_external:
                _copy_external_problem(source_problem_dir, experiment_dir)
            else:
                emitted.append(_emit_walked_problem(
                    name, index, source_problem_dir / f"{name}.pddl", domain_file,
                    walk, experiment_dir, domain, visualizer_root))
            print(f"  ✓ {name}\n")
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")

    if emitted:
        info = build_generation_info(emitted, {
            "source_kind": "problem", "domain": domain, "domain_file": str(domain_file),
            "problems_dir": str(problems_dir), "render": True, **walk.describe()})
        write_generation_info(experiment_dir, info)

    if run_inference_after:
        run_inference(domain, image_trajectories_dir, vendor=vendor,
                      inference_workers=inference_workers)

    print()
    print("=" * 80)
    print("TRAJECTORY GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nExperiment saved to: {experiment_dir}")
    print(f"  Trajectories: {trajectories_dir}")
    print(f"  Image trajectories: {image_trajectories_dir}")
    print(f"  GT trajectories: {experiment_dir / GT_SUBPATH}")
    print(f"  Problems processed: {len(list(image_trajectories_dir.iterdir()))}")
    print()

    return image_trajectories_dir


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate training trajectories for benchmark experiments"
    )
    parser.add_argument(
        "--domain", type=str, default=None,
        help="Domain to generate data for. Required, and one of "
             f"{_REGISTERED_DOMAINS}, for --gen-mode predefined/generate and for "
             "--render. In trace mode it is otherwise optional and only names the "
             "output subdirectory; pass --domain-file to trace a domain this repo "
             "does not know about.",
    )
    parser.add_argument(
        "--num-steps", type=int, default=100,
        help="Max trajectory steps per problem (predefined mode, default: 100)",
    )
    parser.add_argument(
        "--planner", type=str, default=None, choices=["ff", "fd"],
        help="Follow a Fast Downward plan instead of a random walk (predefined "
             "mode). Both names run Fast Downward.",
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

    parser.add_argument(
        "--gen-mode", type=str, default="predefined",
        choices=["predefined", "generate", "trace"],
        help="'predefined' walks (or copies) every pre-authored problem, renders "
             "and infers; 'generate' random-walks --problem-file, cuts it into "
             "problems, renders and infers; 'trace' cuts one trace into problems "
             "with no LLM, rendering only with --render (default: predefined)",
    )
    parser.add_argument(
        "--num-problems", type=int, default=None,
        help="Number of problems to generate (generate/trace modes, default 10)",
    )
    parser.add_argument(
        "--length-min", type=int, default=None,
        help="Min window length in steps (generate/trace modes, default 9)",
    )
    parser.add_argument(
        "--length-max", type=int, default=None,
        help="Max window length in steps (generate/trace modes, default 20)",
    )
    parser.add_argument(
        "--skip", type=int, default=None,
        help="States discarded between windows (generate/trace modes, default 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Walk RNG seed for reproducibility (all modes)",
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
        help="Generate and render folders only, skip LLM inference "
             "(predefined/generate modes)",
    )
    parser.add_argument(
        "--output-dir-name", type=str, default=None,
        help="Custom name for the output directory. Overrides the auto-generated "
             "'multi_problem_<timestamp>__model=<model>...' name.",
    )
    parser.add_argument(
        "--visualizer-root", type=Path, default=None,
        help="The planning-visualizer checkout that renders depot and gripper "
             "(default: rendering.planning_visualizer_dir in config.yaml, else "
             "../planning-visualizer)",
    )

    # ── Trace-mode args (--gen-mode trace, and the source of generate mode) ──
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
             "(trace and generate modes, required)",
    )
    parser.add_argument(
        "--trajectory-file", type=Path, default=None,
        help="The .trajectory to replay (trace mode with --source-kind trajectory)",
    )
    parser.add_argument(
        "--p-rnd", type=float, default=1.0,
        help="Probability of substituting a random applicable action for the planned "
             "one; 1.0 skips the planner entirely (trace mode, default: 1.0)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS,
        help=f"Cap on walked transitions (trace/generate modes, default: {DEFAULT_MAX_STEPS})",
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
        help="Render every state and write the frames as state_*.png (trace "
             "mode, default: off; needs --domain)",
    )

    args = parser.parse_args()

    if args.gen_mode != "trace" and args.domain not in _DOMAIN_REGISTRY:
        parser.error(f"--gen-mode {args.gen_mode} requires --domain, one of "
                     f"{_REGISTERED_DOMAINS}.")

    # The four cut settings default per mode; None means "use the default".
    cut_args = {}
    if args.num_problems is not None:
        cut_args["num_problems"] = args.num_problems
    if args.length_min is not None:
        cut_args["length_min"] = args.length_min
    if args.length_max is not None:
        cut_args["length_max"] = args.length_max
    if args.skip is not None:
        cut_args["skip"] = args.skip

    if args.gen_mode == "trace":
        if args.source_kind is None:
            parser.error("--gen-mode trace requires --source-kind.")
        if args.problem_file is None:
            parser.error("--gen-mode trace requires --problem-file.")

        generate_trajectories_via_trace(
            output_base_dir=Path(__file__).parent / "data",
            source_kind=args.source_kind,
            problem_file=args.problem_file,
            domain=args.domain,
            domain_file=args.domain_file,
            trajectory_file=args.trajectory_file,
            walk=WalkConfig(
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
            visualizer_root=args.visualizer_root,
            cut_mode=args.cut_mode,
            cut_seed=args.cut_seed,
            problem_prefix=args.problem_prefix,
            output_dir_name=args.output_dir_name,
            **cut_args,
        )
    elif args.gen_mode == "generate":
        if args.problem_file is None:
            parser.error("--gen-mode generate requires --problem-file, the "
                         "eval-schema problem whose object universe the walk uses.")
        generate_trajectories_via_generation(
            domain=args.domain,
            output_base_dir=Path(__file__).parent / "data",
            problem_file=args.problem_file,
            vendor=args.vendor,
            seed=args.seed,
            max_steps=args.max_steps,
            run_inference_after=not args.no_inference,
            output_dir_name=args.output_dir_name,
            inference_workers=args.inference_workers,
            visualizer_root=args.visualizer_root,
            **cut_args,
        )
    else:
        generate_trajectories(
            domain=args.domain,
            output_base_dir=Path(__file__).parent / "data",
            num_steps=args.num_steps,
            vendor=args.vendor,
            planner=args.planner,
            seed=args.seed,
            problem_start=args.problem_start,
            problem_end=args.problem_end,
            output_dir_name=args.output_dir_name,
            inference_workers=args.inference_workers,
            visualizer_root=args.visualizer_root,
            run_inference_after=not args.no_inference,
        )
