import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import pandas as pd
from amlgym.algorithms import *
from amlgym.benchmarks import *
from amlgym.metrics import *
from pddl_plus_parser.lisp_parsers import TrajectoryParser, DomainParser
from pddl_plus_parser.models import Observation

from benchmark.amlgym_models.NOISY_PISAM import NOISY_PISAM
from benchmark.amlgym_models.NOISY_SAM import NOISY_SAM
from benchmark.amlgym_models.PISAM import PISAM
from benchmark.amlgym_models.PO_ROSAME import PO_ROSAME
from benchmark.amlgym_models.ROSAME import ROSAME
from benchmark.amlgym_models.SAM import SAM
from src.utils.masking import load_masking_info, save_masking_info
from src.utils.pddl import observation_to_trajectory_file

# =============================================================================
# CONFIG & PATHS
# =============================================================================

benchmark_path = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark")
print_metrics()

# Global lock for thread-safe evaluation (AMLGym SimpleDomainReader is not thread-safe)
evaluation_lock = Lock()

experiment_data_dirs_masked = {
    "blocksworld": ["multi_problem_04-12-2025T12:00:44__model=gpt-5.1__steps=50__planner"],
    "hanoi": ["multi_problem_06-12-2025T13:58:24__model=gpt-5.1__steps=100__planner"],
    "n_puzzle_typed": ["multi_problem_06-12-2025T13:32:59__model=gpt-5.1__steps=100__planner"],
    "maze": ["experiment_07-12-2025T16:16:54__model=gpt-5.1__steps=100__planner"]
}

experiment_data_dirs_fullyobs = {
    "blocksworld": ["multi_problem_07-12-2025T17:27:33__model=gpt-5.1__steps=100__planner__NO_MASK"],
    "hanoi": ["multi_problem_07-12-2025T17:30:57__model=gpt-5.1__steps=100__planner__NO_MASK"],
    "n_puzzle_typed": ["multi_problem_06-12-2025T13:32:59__model=gpt-5.1__steps=100__planner"],
    "maze": ["multi_problem_07-12-2025T17:37:10__model=gpt-5.1__steps=100__planner__NO_MASK"]
}

domain_name_mappings = {
    'n_puzzle_typed': 'npuzzle',
    'blocksworld': 'blocksworld',
    'hanoi': 'hanoi',
    'maze': 'maze',
}

domain_properties = {
    'blocksworld': {
        "domain_path": benchmark_path / 'domains' / 'blocksworld' / 'blocksworld.pddl',
    },
    'hanoi': {
        "domain_path": benchmark_path / 'domains' / 'hanoi' / 'hanoi.pddl',
    },
    'n_puzzle_typed': {
        "domain_path": benchmark_path / 'domains' / 'n_puzzle' / 'n_puzzle.pddl',
    },
    'maze': {
        "domain_path": benchmark_path / 'domains' / 'maze' / 'maze.pddl',
    },
}

N_FOLDS = 5
TRAJECTORY_SIZES = [1, 3, 5, 7, 10]
# TRAJECTORY_SIZES = [1, 3, 5, 7, 10, 20, 30]
# TRAJECTORY_SIZES    = [3]
NUM_TRAJECTORIES = 5  # Always use 5 trajectories

metric_cols = [
    "precision_precs_pos", "precision_precs_neg", "precision_eff_pos", "precision_eff_neg", "precision_overall",
    "recall_precs_pos", "recall_precs_neg", "recall_eff_pos", "recall_eff_neg", "recall_overall",
    "problems_count", "solving_ratio", "false_plans_ratio", "unsolvable_ratio", "timed_out",
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def pre_generate_truncated_trajectories(problem_dirs: List[Path], domain_path: Path,
                                       trajectory_sizes: List[int]) -> None:
    """Pre-generate all truncated trajectories and masking files for all sizes."""
    print(f"\nPre-generating truncated trajectories for sizes: {trajectory_sizes}")

    for prob_dir in problem_dirs:
        traj_files = [f for f in prob_dir.glob("*.trajectory")
                     if 'truncated' not in f.stem and 'final' not in f.stem and 'frame_axioms' not in f.stem]

        for traj_path in traj_files:
            for size in trajectory_sizes:
                truncate_trajectory(traj_path, domain_path, size)

    print(f"✓ Pre-generated all truncated trajectories")


def truncate_trajectory(traj_path: Path, domain_path: Path, max_steps: int) -> Path:
    """Truncate trajectory to max_steps. Skip if already exists."""
    output_path = traj_path.parent / f"{traj_path.stem}_truncated_{max_steps}.trajectory"

    output_masking_path = traj_path.parent / f"{traj_path.stem}_truncated_{max_steps}.masking_info"

    domain = DomainParser(domain_path).parse_domain()
    parser = TrajectoryParser(domain)
    observation = parser.parse_trajectory(traj_path)

    if len(observation.components) > max_steps:
        observation.components = observation.components[:max_steps]

    # Save truncated trajectory if it doesn't exist
    if not output_path.exists():
        observation_to_trajectory_file(observation, output_path)

    # Truncate and save masking_info if it doesn't exist
    if not output_masking_path.exists():
        problem_name = traj_path.stem.split('_truncated_')[0].split('_final')[0].split('_frame_axioms')[0]
        masking_info_path = traj_path.parent / f"{problem_name}.masking_info"

        if masking_info_path.exists():
            masking_info = load_masking_info(masking_info_path, domain)
            truncated_masking_info = masking_info[:max_steps + 1]
            save_masking_info(output_path.parent, output_path.stem, truncated_masking_info)

    return output_path


def save_learning_metrics(output_dir: Path, report: dict, trajectory_mapping: Dict[str, str] = None):
    """Save learning metrics to JSON file."""
    metrics = {
        "learning_time_seconds": report.get("total_time_seconds", None),
        "max_depth": report.get("max_depth", None),
        "nodes_expanded": report.get("nodes_expanded", None),
    }

    # Add trajectory mapping if provided
    if trajectory_mapping:
        metrics["trajectory_mapping"] = trajectory_mapping

    with open(output_dir / "learning_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)


def run_noisy_pisam_trial(domain_path: Path, trajectories: List[Path], testing_dir: Path,
                         fold: int, traj_size: int) -> Tuple[str, List[Observation], dict]:
    """Run NOISY_PISAM and save results."""
    noisy_pisam = NOISY_PISAM()
    model, final_obs, report = noisy_pisam.learn(str(domain_path), [str(t) for t in trajectories])

    # Create output directory
    timestamp = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
    output_dir = testing_dir / f"{timestamp}__fold={fold}__traj-size={traj_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create trajectory mapping (final_observation_X -> original trajectory name)
    trajectory_mapping = {}
    for i, (obs, traj_path) in enumerate(zip(final_obs, trajectories)):
        obs_path = output_dir / f"final_observation_{i}.trajectory"
        observation_to_trajectory_file(obs, obs_path)
        # Store original trajectory name (without path)
        trajectory_mapping[f"final_observation_{i}"] = traj_path.name

    # Save learning metrics with trajectory mapping
    save_learning_metrics(output_dir, report, trajectory_mapping)

    # Save learned model
    with open(output_dir / "learned_model.pddl", 'w') as f:
        f.write(model)

    return model, final_obs, report


def run_noisy_sam_trial(domain_path: Path, trajectories: List[Path], testing_dir: Path,
                        fold: int, traj_size: int) -> Tuple[str, List[Observation], dict]:
    """Run NOISY_SAM and save results."""
    noisy_sam = NOISY_SAM()
    model, final_obs, report = noisy_sam.learn(str(domain_path), [str(t) for t in trajectories])

    # Create output directory
    timestamp = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
    output_dir = testing_dir / f"{timestamp}__fold={fold}__traj-size={traj_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create trajectory mapping (final_observation_X -> original trajectory name)
    trajectory_mapping = {}
    for i, (obs, traj_path) in enumerate(zip(final_obs, trajectories)):
        obs_path = output_dir / f"final_observation_{i}.trajectory"
        observation_to_trajectory_file(obs, obs_path)
        # Store original trajectory name (without path)
        trajectory_mapping[f"final_observation_{i}"] = traj_path.name

    # Save learning metrics with trajectory mapping
    save_learning_metrics(output_dir, report, trajectory_mapping)

    # Save learned model
    with open(output_dir / "learned_model.pddl", 'w') as f:
        f.write(model)

    return model, final_obs, report


def evaluate_model(model_path: str, domain_ref_path: Path, test_problems: List[str]) -> dict:
    """Evaluate a learned model. Handles AMLGym SimpleDomainReader race conditions."""
    # NOTE: evaluation_lock is a threading lock, but we use ProcessPoolExecutor,
    # so it doesn't prevent race conditions across processes.

    import time
    import random

    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Add small random delay to reduce collision probability
            if attempt > 0:
                time.sleep(random.uniform(0.1, 0.5))

            # Run all evaluations together - if any fail, retry all
            precision = syntactic_precision(model_path, str(domain_ref_path))
            recall = syntactic_recall(model_path, str(domain_ref_path))
            problem_solving_result = problem_solving(model_path, str(domain_ref_path), test_problems, timeout=60)

            # Success - break out of retry loop
            break

        except (FileNotFoundError, ValueError, IndexError) as e:
            # Race condition with SimpleDomainReader
            if attempt < max_retries - 1:
                # Clean up potentially corrupted _clean file
                clean_file = f"{domain_ref_path}_clean"
                try:
                    if Path(clean_file).exists():
                        Path(clean_file).unlink()
                except:
                    pass
                continue
            else:
                # Final attempt failed - return null metrics
                print(f"Warning: Evaluation failed after {max_retries} attempts: {e}")
                precision = None
                recall = None
                problem_solving_result = None

    return {
        'precision_precs_pos': precision.get('precs_pos') if isinstance(precision, dict) else None,
        'precision_precs_neg': precision.get('precs_neg') if isinstance(precision, dict) else None,
        'precision_eff_pos': precision.get('eff_pos') if isinstance(precision, dict) else None,
        'precision_eff_neg': precision.get('eff_neg') if isinstance(precision, dict) else None,
        'precision_overall': precision.get('mean') if isinstance(precision, dict) else precision,
        'recall_precs_pos': recall.get('precs_pos') if isinstance(recall, dict) else None,
        'recall_precs_neg': recall.get('precs_neg') if isinstance(recall, dict) else None,
        'recall_eff_pos': recall.get('eff_pos') if isinstance(recall, dict) else None,
        'recall_eff_neg': recall.get('eff_neg') if isinstance(recall, dict) else None,
        'recall_overall': recall.get('mean') if isinstance(recall, dict) else recall,
        'solving_ratio': problem_solving_result.get('solving_ratio') if isinstance(problem_solving_result, dict) else None,
        'false_plans_ratio': problem_solving_result.get('false_plans_ratio') if isinstance(problem_solving_result, dict) else None,
        'unsolvable_ratio': problem_solving_result.get('unsolvable_ratio') if isinstance(problem_solving_result, dict) else None,
        'timed_out': problem_solving_result.get('timed_out') if isinstance(problem_solving_result, dict) else None,
    }


def format_mean_std(mean_val, std_val) -> str:
    """Format value as mean±std."""
    if mean_val is None or pd.isna(mean_val):
        return ""

    # Handle non-numeric values
    if isinstance(mean_val, str):
        return mean_val

    if std_val is None or pd.isna(std_val):
        return f"{mean_val:.3f}"

    # Handle non-numeric std_val
    if isinstance(std_val, str):
        return f"{mean_val:.3f}"

    return f"{mean_val:.3f}±{std_val:.3f}"


def run_single_fold(fold: int, problem_dirs: List[Path], n_problems: int, traj_size: int,
                    domain_ref_path: Path, testing_dir: Path, bench_name: str, mode: str = 'masked') -> List[dict]:
    """
    Run a single fold experiment and return 4 results.

    Args:
        mode: Either 'masked' (PISAM/PO_ROSAME with masking) or 'fullyobs' (SAM/ROSAME without masking)

    Returns:
        List of 4 dicts with results for: unclean SAM/PISAM, unclean ROSAME, cleaned SAM/PISAM, cleaned ROSAME
    """
    print(f"[PID {os.getpid()}] Fold {fold+1}/{N_FOLDS}, size={traj_size}, mode={mode}")

    fold_work_dir = testing_dir / f"work_fold{fold}_size{traj_size}"
    fold_work_dir.mkdir(parents=True, exist_ok=True)
    original_cwd = os.getcwd()
    os.chdir(fold_work_dir)

    try:
        # CV split
        indices = list(range(n_problems))
        random.seed(42 + fold)
        random.shuffle(indices)
        n_train = max(1, min(int(0.8 * n_problems), n_problems - 1))
        train_idx, test_idx = indices[:n_train], indices[n_train:]

        train_problem_dirs = [problem_dirs[i] for i in train_idx]
        test_problem_dirs = [problem_dirs[i] for i in test_idx]

        # Select and use pre-generated truncated trajectories
        random.seed(42 + fold)
        selected_dirs = random.sample(train_problem_dirs, min(NUM_TRAJECTORIES, len(train_problem_dirs)))

        truncated_trajs = []
        for prob_dir in selected_dirs:
            truncated_file = prob_dir / f"{prob_dir.name}_truncated_{traj_size}.trajectory"
            if truncated_file.exists():
                truncated_trajs.append(truncated_file)

        test_problem_paths = [str(list(d.glob("*.pddl"))[0]) for d in test_problem_dirs if list(d.glob("*.pddl"))]

        null_metrics = {k: None for k in ['precision_precs_pos', 'precision_precs_neg',
                        'precision_eff_pos', 'precision_eff_neg', 'precision_overall',
                        'recall_precs_pos', 'recall_precs_neg', 'recall_eff_pos',
                        'recall_eff_neg', 'recall_overall', 'solving_ratio',
                        'false_plans_ratio', 'unsolvable_ratio', 'timed_out']}

        # ========================================================================
        # PHASE 1: UNCLEAN (learning on original truncated trajectories)
        # ========================================================================
        print(f"  Phase 1: Learning on unclean trajectories...")

        # Choose algorithms based on mode
        if mode == 'masked':
            # Run base PISAM (no denoising) on truncated trajectories - PISAM handles trajectories directly
            sam_unclean = PISAM()
            sam_unclean_model = sam_unclean.learn(str(domain_ref_path), [str(t) for t in truncated_trajs])
            algo_name = 'PISAM'
        else:  # fullyobs
            # For SAM, we need to create problem files alongside trajectories
            temp_sam_unclean_dir = testing_dir / f"temp_sam_unclean_fold{fold}_size{traj_size}"
            temp_sam_unclean_dir.mkdir(parents=True, exist_ok=True)

            print(f"  DEBUG SAM unclean: Number of truncated_trajs={len(truncated_trajs)}")
            sam_unclean_traj_paths = []
            for truncated_traj in truncated_trajs:
                problem_name = truncated_traj.stem.split('_truncated_')[0]
                problem_file = truncated_traj.parent / f"{truncated_traj.parent.name}.pddl"

                print(f"  DEBUG SAM unclean: truncated_traj={truncated_traj}")
                print(f"  DEBUG SAM unclean: problem_file={problem_file}, exists={problem_file.exists()}")

                if not problem_file.exists():
                    print(f"  DEBUG SAM unclean: SKIPPING - problem file not found")
                    continue

                problem_dir = temp_sam_unclean_dir / problem_name
                problem_dir.mkdir(parents=True, exist_ok=True)

                # Copy trajectory and problem file
                traj_path = problem_dir / f"{problem_name}.trajectory"
                shutil.copy(truncated_traj, traj_path)
                shutil.copy(problem_file, problem_dir / f"{problem_name}.pddl")
                sam_unclean_traj_paths.append(str(traj_path))
                print(f"  DEBUG SAM unclean: Successfully copied to {traj_path}")

            print(f"  DEBUG SAM unclean: Final count={len(sam_unclean_traj_paths)} trajectories")
            print(f"  DEBUG SAM unclean: Calling SAM.learn with {len(sam_unclean_traj_paths)} trajectories")
            sam_unclean = SAM()
            sam_unclean_model = sam_unclean.learn(str(domain_ref_path), sam_unclean_traj_paths, use_problems=False)
            algo_name = 'SAM'
            print(f"  DEBUG SAM unclean: SAM.learn completed, model length={len(sam_unclean_model)}")

        temp_sam_unclean_path = testing_dir / f'{algo_name}_unclean_{bench_name}_fold{fold}_size{traj_size}.pddl'
        temp_sam_unclean_path.write_text(sam_unclean_model)

        sam_unclean_metrics = evaluate_model(str(temp_sam_unclean_path), domain_ref_path, test_problem_paths)
        unclean_sam_result = {
            'domain': bench_name, 'algorithm': algo_name, 'fold': fold,
            'traj_size': traj_size, 'problems_count': len(test_problem_paths),
            '_internal_phase': 'unclean',
            'fold_data_creation_timedout': 0,
            **sam_unclean_metrics
        }
        temp_sam_unclean_path.unlink()

        # Run ROSAME on original truncated trajectories
        temp_rosame_unclean_dir = testing_dir / f"temp_rosame_unclean_fold{fold}_size{traj_size}"
        temp_rosame_unclean_dir.mkdir(parents=True, exist_ok=True)

        rosame_unclean_traj_paths = []
        for truncated_traj in truncated_trajs:
            problem_name = truncated_traj.stem.split('_truncated_')[0]
            problem_file = truncated_traj.parent / f"{truncated_traj.parent.name}.pddl"

            if not problem_file.exists():
                continue

            problem_dir = temp_rosame_unclean_dir / problem_name
            problem_dir.mkdir(parents=True, exist_ok=True)

            # Copy the truncated trajectory itself
            traj_path = problem_dir / f"{problem_name}.trajectory"
            shutil.copy(truncated_traj, traj_path)
            shutil.copy(problem_file, problem_dir / f"{problem_name}.pddl")

            # For masked mode, also copy masking_info
            if mode == 'masked':
                masking_file = truncated_traj.parent / f"{truncated_traj.stem}.masking_info"
                if masking_file.exists():
                    shutil.copy(masking_file, problem_dir / f"{problem_name}.masking_info")
                else:
                    continue  # Skip if masking file missing in masked mode

            rosame_unclean_traj_paths.append(str(traj_path))

        if rosame_unclean_traj_paths:
            try:
                if mode == 'masked':
                    rosame_unclean_model = PO_ROSAME().learn(str(domain_ref_path), rosame_unclean_traj_paths, use_problems=False)
                else:  # fullyobs
                    rosame_unclean_model = ROSAME().learn(str(domain_ref_path), rosame_unclean_traj_paths, use_problems=False)

                if rosame_unclean_model and ":action" in rosame_unclean_model:
                    temp_rosame_unclean_path = testing_dir / f'ROSAME_unclean_{bench_name}_fold{fold}_size{traj_size}.pddl'
                    temp_rosame_unclean_path.write_text(rosame_unclean_model)
                    rosame_unclean_metrics = evaluate_model(str(temp_rosame_unclean_path), domain_ref_path, test_problem_paths)
                    temp_rosame_unclean_path.unlink()
                else:
                    raise ValueError("Invalid ROSAME model")
            except Exception as e:
                print(f"  Warning: ROSAME (unclean) failed: {e}")
                rosame_unclean_metrics = null_metrics
        else:
            rosame_unclean_metrics = null_metrics

        unclean_rosame_result = {
            'domain': bench_name, 'algorithm': 'ROSAME', 'fold': fold,
            'traj_size': traj_size, 'problems_count': len(test_problem_paths),
            '_internal_phase': 'unclean',
            'fold_data_creation_timedout': 0,
            **rosame_unclean_metrics
        }

        # ========================================================================
        # PHASE 2: CLEANED (learning on denoised trajectories)
        # ========================================================================
        print(f"  Phase 2: Learning on cleaned trajectories...")

        # Run NOISY_PISAM or NOISY_SAM based on mode
        if mode == 'masked':
            sam_model, final_obs, report = run_noisy_pisam_trial(
                domain_ref_path, truncated_trajs, testing_dir, fold, traj_size)
            algo_name = 'PISAM'
        else:  # fullyobs
            # For NOISY_SAM, we need to create problem files alongside trajectories
            temp_noisy_sam_dir = testing_dir / f"temp_noisy_sam_fold{fold}_size{traj_size}"
            temp_noisy_sam_dir.mkdir(parents=True, exist_ok=True)

            noisy_sam_traj_paths = []
            for truncated_traj in truncated_trajs:
                problem_name = truncated_traj.stem.split('_truncated_')[0]
                problem_file = truncated_traj.parent / f"{truncated_traj.parent.name}.pddl"

                if not problem_file.exists():
                    continue

                problem_dir = temp_noisy_sam_dir / problem_name
                problem_dir.mkdir(parents=True, exist_ok=True)

                # Copy trajectory and problem file
                traj_path = problem_dir / f"{problem_name}.trajectory"
                shutil.copy(truncated_traj, traj_path)
                shutil.copy(problem_file, problem_dir / f"{problem_name}.pddl")
                noisy_sam_traj_paths.append(traj_path)

            sam_model, final_obs, report = run_noisy_sam_trial(
                domain_ref_path, noisy_sam_traj_paths, testing_dir, fold, traj_size)
            algo_name = 'SAM'

        # Determine if fold data creation timed out
        fold_timedout = 0 if report.get('terminated_by') == 'solution_found' else 1

        temp_sam_path = testing_dir / f'{algo_name}_{bench_name}_fold{fold}_size{traj_size}.pddl'
        temp_sam_path.write_text(sam_model)

        sam_metrics = evaluate_model(str(temp_sam_path), domain_ref_path, test_problem_paths)
        cleaned_sam_result = {
            'domain': bench_name, 'algorithm': algo_name, 'fold': fold,
            'traj_size': traj_size, 'problems_count': len(test_problem_paths),
            '_internal_phase': 'cleaned',
            'fold_data_creation_timedout': fold_timedout,
            **sam_metrics
        }

        # Run ROSAME with final observations
        temp_rosame_dir = testing_dir / f"temp_rosame_fold{fold}_size{traj_size}"
        temp_rosame_dir.mkdir(parents=True, exist_ok=True)

        rosame_traj_paths = []
        for obs, truncated_traj in zip(final_obs, truncated_trajs):
            problem_name = truncated_traj.stem.split('_truncated_')[0]
            problem_file = truncated_traj.parent / f"{truncated_traj.parent.name}.pddl"

            if not problem_file.exists():
                continue

            problem_dir = temp_rosame_dir / problem_name
            problem_dir.mkdir(parents=True, exist_ok=True)

            traj_path = problem_dir / f"{problem_name}.trajectory"
            observation_to_trajectory_file(obs, traj_path)
            shutil.copy(problem_file, problem_dir / f"{problem_name}.pddl")

            # Handle masking info only in masked mode
            if mode == 'masked':
                masking_file = truncated_traj.parent / f"{truncated_traj.stem}.masking_info"
                if not masking_file.exists():
                    continue

                # Load original masking info and adjust to final observation length
                domain = DomainParser(domain_ref_path).parse_domain()
                original_masking_info = load_masking_info(masking_file, domain)

                # Final observation may have different length than truncated trajectory
                final_obs_length = len(obs.components)

                # Adjust masking info to match final observation length
                if len(original_masking_info) - 1 > final_obs_length:
                    # Truncate if final obs is shorter
                    adjusted_masking_info = original_masking_info[:final_obs_length + 1]
                elif len(original_masking_info) - 1 < final_obs_length:
                    # Extend with last masking state if final obs is longer
                    adjusted_masking_info = original_masking_info + [original_masking_info[-1]] * (final_obs_length - (len(original_masking_info) - 1))
                else:
                    adjusted_masking_info = original_masking_info

                # Save adjusted masking info
                save_masking_info(problem_dir, problem_name, adjusted_masking_info)

            rosame_traj_paths.append(str(traj_path))

        if rosame_traj_paths:
            try:
                if mode == 'masked':
                    rosame_model = PO_ROSAME().learn(str(domain_ref_path), rosame_traj_paths, use_problems=False)
                else:  # fullyobs
                    rosame_model = ROSAME().learn(str(domain_ref_path), rosame_traj_paths, use_problems=False)

                if rosame_model and ":action" in rosame_model:
                    temp_rosame_path = testing_dir / f'ROSAME_{bench_name}_fold{fold}_size{traj_size}.pddl'
                    temp_rosame_path.write_text(rosame_model)
                    rosame_metrics = evaluate_model(str(temp_rosame_path), domain_ref_path, test_problem_paths)
                    temp_rosame_path.unlink()
                else:
                    raise ValueError("Invalid ROSAME model")
            except Exception as e:
                print(f"  Warning: ROSAME (cleaned) failed: {e}")
                rosame_metrics = null_metrics
        else:
            rosame_metrics = null_metrics

        cleaned_rosame_result = {
            'domain': bench_name, 'algorithm': 'ROSAME', 'fold': fold,
            'traj_size': traj_size, 'problems_count': len(test_problem_paths),
            '_internal_phase': 'cleaned',
            'fold_data_creation_timedout': fold_timedout,
            **rosame_metrics
        }

        # Cleanup
        if temp_sam_path.exists():
            temp_sam_path.unlink()

        return [unclean_sam_result, unclean_rosame_result, cleaned_sam_result, cleaned_rosame_result]
    finally:
        # Always restore working directory
        os.chdir(original_cwd)


def generate_excel_report(unclean_results: List[dict], cleaned_results: List[dict], output_path: Path):
    """Generate Excel report with aggregated results for both unclean and cleaned trajectories."""
    if not unclean_results and not cleaned_results:
        return

    # Define metric groups for Excel table structure
    precision_metrics = ["precision_precs_pos", "precision_precs_neg", "precision_eff_pos", "precision_eff_neg", "precision_overall"]
    recall_metrics = ["recall_precs_pos", "recall_precs_neg", "recall_eff_pos", "recall_eff_neg", "recall_overall"]
    problem_metrics = ["problems_count", "solving_ratio", "false_plans_ratio", "unsolvable_ratio", "timed_out"]

    # Process both result sets and combine with phase labels
    all_results_with_phase = []

    if unclean_results:
        df_unclean = pd.DataFrame(unclean_results)
        grouped_unclean = df_unclean.groupby(["domain", "algorithm", "traj_size"])[metric_cols].agg(["mean", "std"]).reset_index()

        flat_cols = []
        for col in grouped_unclean.columns:
            if isinstance(col, tuple):
                base, stat = col
                flat_cols.append(base if stat == "" else f"{base}_{stat}")
            else:
                flat_cols.append(col)
        grouped_unclean.columns = flat_cols

        df_avg_unclean = grouped_unclean[["domain", "algorithm", "traj_size"]].copy()
        for m in metric_cols:
            df_avg_unclean[m] = grouped_unclean[f"{m}_mean"]
            df_avg_unclean[f"{m}_std"] = grouped_unclean[f"{m}_std"]
        df_avg_unclean["_phase"] = "unclean"

        all_results_with_phase.append(df_avg_unclean)

    if cleaned_results:
        df_cleaned = pd.DataFrame(cleaned_results)
        grouped_cleaned = df_cleaned.groupby(["domain", "algorithm", "traj_size"])[metric_cols].agg(["mean", "std"]).reset_index()

        flat_cols = []
        for col in grouped_cleaned.columns:
            if isinstance(col, tuple):
                base, stat = col
                flat_cols.append(base if stat == "" else f"{base}_{stat}")
            else:
                flat_cols.append(col)
        grouped_cleaned.columns = flat_cols

        df_avg_cleaned = grouped_cleaned[["domain", "algorithm", "traj_size"]].copy()
        for m in metric_cols:
            df_avg_cleaned[m] = grouped_cleaned[f"{m}_mean"]
            df_avg_cleaned[f"{m}_std"] = grouped_cleaned[f"{m}_std"]
        df_avg_cleaned["_phase"] = "cleaned"

        all_results_with_phase.append(df_avg_cleaned)

    df_avg = pd.concat(all_results_with_phase, ignore_index=True)

    # Group by (trajectory size, phase)
    by_size_phase = defaultdict(list)
    for _, row in df_avg.iterrows():
        phase = row["_phase"]
        size_key = f"{int(row['traj_size'])}__{'unclean' if phase == 'unclean' else ''}"
        by_size_phase[size_key].append(row.to_dict())

    def clean_excel_value(v):
        if v is None or pd.isna(v):
            return ""
        if isinstance(v, float) and (v == float("inf") or v == float("-inf")):
            return ""
        return v

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        workbook = writer.book
        thin_border = workbook.add_format({"border": 1})
        thick_left = workbook.add_format({"border": 1, "left": 2})
        thick_right = workbook.add_format({"border": 1, "right": 2})

        # Sort sheet names: size=1__unclean, size=1, size=3__unclean, size=3, ...
        def sort_key(key):
            parts = key.split('__')
            size = int(parts[0])
            phase = 0 if len(parts) > 1 and parts[1] == 'unclean' else 1
            return (size, phase)

        for size_phase_key in sorted(by_size_phase.keys(), key=sort_key):
            results = by_size_phase[size_phase_key]
            # Sheet name: "size=1__unclean" or "size=1" (for cleaned)
            if size_phase_key.endswith('__unclean'):
                sheet_name = f"size={size_phase_key}"
            else:
                sheet_name = f"size={size_phase_key.split('__')[0]}"  # Remove trailing "__"

            sheet = workbook.add_worksheet(sheet_name)
            writer.sheets[sheet_name] = sheet

            domains = sorted({r["domain"] for r in results})
            algorithms = sorted({r["algorithm"] for r in results})

            # Map (domain, algorithm) -> result dict
            res_map = {(r["domain"], r["algorithm"]): r for r in results}

            def write_syn_table(start_row):
                """Write syntactic P/R table with mean±std values."""
                row0, row1, row2 = start_row, start_row + 1, start_row + 2

                sheet.write(row0, 0, "", thin_border)
                sheet.write(row1, 0, "", thin_border)
                sheet.write(row2, 0, "Domain", thin_border)

                col = 1
                type_spans, metric_spans = {}, {}

                for t, metrics in [("Precision", precision_metrics), ("Recall", recall_metrics)]:
                    type_start = col
                    for m in metrics:
                        metric_start = col
                        for _alg in algorithms:
                            col += 1
                        metric_end = col - 1
                        metric_spans[(t, m)] = (metric_start, metric_end)
                    type_end = col - 1
                    type_spans[t] = (type_start, type_end)

                # Write merged headers
                for t, (c_start, c_end) in type_spans.items():
                    sheet.merge_range(row0, c_start, row0, c_end, t, thin_border)
                for (t, m), (c_start, c_end) in metric_spans.items():
                    sheet.merge_range(row1, c_start, row1, c_end, m, thin_border)

                # Write algorithm names
                col_ptr = 1
                for t, metrics in [("Precision", precision_metrics), ("Recall", recall_metrics)]:
                    for m in metrics:
                        for alg in algorithms:
                            sheet.write(row2, col_ptr, alg, thin_border)
                            col_ptr += 1

                # Write data rows with mean±std format
                for i, dom in enumerate(domains):
                    r_idx = row2 + 1 + i
                    sheet.write(r_idx, 0, dom, thin_border)
                    c = 1
                    for t, metrics in [("Precision", precision_metrics), ("Recall", recall_metrics)]:
                        for m in metrics:
                            for alg in algorithms:
                                res = res_map.get((dom, alg), {})
                                mean_val = clean_excel_value(res.get(m))
                                std_val = clean_excel_value(res.get(f"{m}_std"))
                                formatted = format_mean_std(mean_val, std_val)
                                sheet.write(r_idx, c, formatted, thin_border)
                                c += 1

                first_row, last_row, last_col = row0, row2 + len(domains), col - 1

                # Thick borders between metric groups
                for (_t, _m), (start_c, end_c) in metric_spans.items():
                    sheet.conditional_format(
                        first_row, start_c, last_row, start_c,
                        {"type": "formula", "criteria": "TRUE", "format": thick_left},
                    )
                    sheet.conditional_format(
                        first_row, end_c, last_row, end_c,
                        {"type": "formula", "criteria": "TRUE", "format": thick_right},
                    )

                return first_row, last_row, last_col

            def write_prob_table(start_row):
                """Write problem-solving table with mean±std values."""
                row0, row1, row2 = start_row, start_row + 1, start_row + 2

                sheet.write(row0, 0, "", thin_border)
                sheet.write(row1, 0, "", thin_border)
                sheet.write(row2, 0, "Domain", thin_border)

                col = 1
                metric_spans = {}
                group_start = col

                for m in problem_metrics:
                    metric_start = col
                    for _alg in algorithms:
                        col += 1
                    metric_end = col - 1
                    metric_spans[m] = (metric_start, metric_end)
                group_end = col - 1

                sheet.merge_range(row0, group_start, row0, group_end, "ProblemSolving", thin_border)

                for m, (c_start, c_end) in metric_spans.items():
                    sheet.merge_range(row1, c_start, row1, c_end, m, thin_border)

                col_ptr = 1
                for m in problem_metrics:
                    for alg in algorithms:
                        sheet.write(row2, col_ptr, alg, thin_border)
                        col_ptr += 1

                # Write data rows with mean±std format
                for i, dom in enumerate(domains):
                    r_idx = row2 + 1 + i
                    sheet.write(r_idx, 0, dom, thin_border)
                    c = 1
                    for m in problem_metrics:
                        for alg in algorithms:
                            res = res_map.get((dom, alg), {})
                            mean_val = clean_excel_value(res.get(m))
                            std_val = clean_excel_value(res.get(f"{m}_std"))
                            formatted = format_mean_std(mean_val, std_val)
                            sheet.write(r_idx, c, formatted, thin_border)
                            c += 1

                first_row, last_row, last_col = row0, row2 + len(domains), col - 1

                # Thick borders between metrics
                for _m, (start_c, end_c) in metric_spans.items():
                    sheet.conditional_format(
                        first_row, start_c, last_row, start_c,
                        {"type": "formula", "criteria": "TRUE", "format": thick_left},
                    )
                    sheet.conditional_format(
                        first_row, end_c, last_row, end_c,
                        {"type": "formula", "criteria": "TRUE", "format": thick_right},
                    )

                return first_row, last_row, last_col

            # Generate tables
            syn_first, syn_last, syn_last_col = write_syn_table(start_row=0)
            gap = 5
            prob_start = syn_last + 1 + gap
            prob_first, prob_last, prob_last_col = write_prob_table(start_row=prob_start)


def generate_plots(unclean_results: List[dict], cleaned_results: List[dict], plots_dir: Path):
    """Generate plots per domain comparing unclean vs cleaned trajectories."""
    plots_dir.mkdir(exist_ok=True)

    def plot_metric_vs_size(df, metric_key, metric_title, save_path, phase_label, domain_label):
        """Plot metric vs trajectory size with error bars."""
        if df.empty:
            return

        plt.figure(figsize=(8, 5))

        algorithms = sorted(df["algorithm"].unique())
        for algo in algorithms:
            sub = df[df["algorithm"] == algo].sort_values("traj_size")
            x = sub["traj_size"]
            y = sub[metric_key]
            yerr = sub[f"{metric_key}_std"] if f"{metric_key}_std" in sub.columns else None

            plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4, label=algo)

        plt.title(f"{metric_title} vs Trajectory Size ({phase_label} - {domain_label})")
        plt.xlabel("Trajectory Size (steps)")
        plt.ylabel(metric_title)

        # Set x-axis ticks: bins of 5 (0, 5, 10, ..., 30)
        plt.xticks([0, 5, 10, 15, 20, 25, 30])

        # Set y-axis ticks: bins of 0.1 (0, 0.1, ..., 1.0)
        plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # Get all unique domains
    all_domains = set()
    if unclean_results:
        all_domains.update(r['domain'] for r in unclean_results)
    if cleaned_results:
        all_domains.update(r['domain'] for r in cleaned_results)

    # Generate plots for each domain
    for domain in sorted(all_domains):
        domain_upper = domain.upper()

        # Process unclean results for this domain
        if unclean_results:
            domain_unclean = [r for r in unclean_results if r['domain'] == domain]
            if domain_unclean:
                df_unclean = pd.DataFrame(domain_unclean)
                grouped_unclean = df_unclean.groupby(["algorithm", "traj_size"])[metric_cols].agg(["mean", "std"]).reset_index()

                flat_cols = []
                for col in grouped_unclean.columns:
                    if isinstance(col, tuple):
                        base, stat = col
                        flat_cols.append(base if stat == "" else f"{base}_{stat}")
                    else:
                        flat_cols.append(col)
                grouped_unclean.columns = flat_cols

                df_avg_unclean = grouped_unclean[["algorithm", "traj_size"]].copy()
                for m in metric_cols:
                    df_avg_unclean[m] = grouped_unclean[f"{m}_mean"]
                    df_avg_unclean[f"{m}_std"] = grouped_unclean[f"{m}_std"]

                plot_metric_vs_size(df_avg_unclean, "solving_ratio", "Solving Ratio",
                                   plots_dir / f"solving_ratio_vs_traj_size__unclean_({domain_upper}).png",
                                   "Unclean", domain_upper)
                plot_metric_vs_size(df_avg_unclean, "false_plans_ratio", "False Plan Ratio",
                                   plots_dir / f"false_plans_ratio_vs_traj_size__unclean_({domain_upper}).png",
                                   "Unclean", domain_upper)
                plot_metric_vs_size(df_avg_unclean, "unsolvable_ratio", "Unsolvable Ratio",
                                   plots_dir / f"unsolvable_ratio_vs_traj_size__unclean_({domain_upper}).png",
                                   "Unclean", domain_upper)

        # Process cleaned results for this domain
        if cleaned_results:
            domain_cleaned = [r for r in cleaned_results if r['domain'] == domain]
            if domain_cleaned:
                df_cleaned = pd.DataFrame(domain_cleaned)
                grouped_cleaned = df_cleaned.groupby(["algorithm", "traj_size"])[metric_cols].agg(["mean", "std"]).reset_index()

                flat_cols = []
                for col in grouped_cleaned.columns:
                    if isinstance(col, tuple):
                        base, stat = col
                        flat_cols.append(base if stat == "" else f"{base}_{stat}")
                    else:
                        flat_cols.append(col)
                grouped_cleaned.columns = flat_cols

                df_avg_cleaned = grouped_cleaned[["algorithm", "traj_size"]].copy()
                for m in metric_cols:
                    df_avg_cleaned[m] = grouped_cleaned[f"{m}_mean"]
                    df_avg_cleaned[f"{m}_std"] = grouped_cleaned[f"{m}_std"]

                plot_metric_vs_size(df_avg_cleaned, "solving_ratio", "Solving Ratio",
                                   plots_dir / f"solving_ratio_vs_traj_size_({domain_upper}).png",
                                   "Cleaned", domain_upper)
                plot_metric_vs_size(df_avg_cleaned, "false_plans_ratio", "False Plan Ratio",
                                   plots_dir / f"false_plans_ratio_vs_traj_size_({domain_upper}).png",
                                   "Cleaned", domain_upper)
                plot_metric_vs_size(df_avg_cleaned, "unsolvable_ratio", "Unsolvable Ratio",
                                   plots_dir / f"unsolvable_ratio_vs_traj_size_({domain_upper}).png",
                                   "Cleaned", domain_upper)


# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================
def main(selected_domains: List[str] = None, mode: str = 'masked'):
    """
    Run benchmark experiments.

    Args:
        selected_domains: List of domain names to run, or None for all domains in domain_name_mappings
        mode: Either 'masked' or 'fullyobs'
    """
    unclean_results = []
    cleaned_results = []

    # Create evaluation results directory
    evaluation_results_dir = benchmark_path / 'data' / 'evaluation_results'
    evaluation_results_dir.mkdir(parents=True, exist_ok=True)

    # Select appropriate experiment data directories based on mode
    experiment_data_dirs = experiment_data_dirs_masked if mode == 'masked' else experiment_data_dirs_fullyobs

    # Filter domains if specific domains are requested
    if selected_domains:
        domains_to_run = {k: v for k, v in domain_name_mappings.items() if k in selected_domains}
    else:
        domains_to_run = domain_name_mappings

    print(f"\n{'='*80}")
    print(f"RUNNING BENCHMARK IN {mode.upper()} MODE")
    print(f"Domains: {list(domains_to_run.keys())}")
    print(f"{'='*80}\n")

    for domain_name, bench_name in domains_to_run.items():
        domain_ref_path = domain_properties[domain_name]["domain_path"]

        for dir_name in experiment_data_dirs[domain_name]:
            data_dir = benchmark_path / 'data' / domain_name / dir_name
            trajectories_dir = data_dir / 'training' / 'trajectories'
            testing_dir = data_dir / 'testing'
            testing_dir.mkdir(parents=True, exist_ok=True)

            problem_dirs = sorted([d for d in trajectories_dir.iterdir() if d.is_dir()])
            n_problems = len(problem_dirs)

            if n_problems < 2:
                raise ValueError(f"Domain {bench_name} has too few problems ({n_problems}) for 80/20 CV.")

            print(f"\n{'=' * 80}")
            print(f"Domain: {bench_name} | data dir: {dir_name}")
            print(f"Total problems: {n_problems}")
            print(f"Trajectory sizes: {TRAJECTORY_SIZES}")
            print(f"CV folds: {N_FOLDS}")
            print(f"{'=' * 80}\n")

            # Pre-generate all truncated trajectories before parallel execution
            pre_generate_truncated_trajectories(problem_dirs, domain_ref_path, TRAJECTORY_SIZES)

            # IMPORTANT: Size first, then folds (so cheap computations finish first)
            for traj_size in TRAJECTORY_SIZES:
                print(f"\n{'='*60}\nTRAJECTORY SIZE = {traj_size}\n{'='*60}")

                # Run all folds in parallel for this trajectory size
                # with ThreadPoolExecutor(max_workers=N_FOLDS) as executor:
                with ProcessPoolExecutor(max_workers=N_FOLDS) as executor:
                    futures = []
                    for fold in range(N_FOLDS):
                        future = executor.submit(
                            run_single_fold,
                            fold, problem_dirs, n_problems, traj_size,
                            domain_ref_path, testing_dir, bench_name, mode
                        )
                        futures.append(future)

                    # Wait for all folds to complete and collect results
                    for future in as_completed(futures):
                        try:
                            results_list = future.result()  # Returns 4 results: [unclean_pisam, unclean_rosame, cleaned_pisam, cleaned_rosame]

                            # Separate by phase and remove internal marker
                            for result in results_list:
                                phase = result.pop('_internal_phase')
                                if phase == 'unclean':
                                    unclean_results.append(result)
                                else:  # phase == 'cleaned'
                                    cleaned_results.append(result)
                        except Exception as e:
                            print(f"ERROR in fold: {e}")
                            import traceback
                            traceback.print_exc()

                # Write TWO separate CSV files after all folds complete
                csv_unclean = evaluation_results_dir / f"results_{bench_name}_unclean.csv"
                csv_cleaned = evaluation_results_dir / f"results_{bench_name}.csv"

                pd.DataFrame(unclean_results).to_csv(csv_unclean, index=False)
                pd.DataFrame(cleaned_results).to_csv(csv_cleaned, index=False)

                # Create combined CSV (unclean + cleaned with phase column)
                csv_combined = evaluation_results_dir / f"results_{bench_name}_combined.csv"

                # Filter results for this domain and add phase column
                domain_unclean = [dict(r, phase='unclean') for r in unclean_results if r['domain'] == bench_name]
                domain_cleaned = [dict(r, phase='cleaned') for r in cleaned_results if r['domain'] == bench_name]
                combined_data = domain_unclean + domain_cleaned
                pd.DataFrame(combined_data).to_csv(csv_combined, index=False)

                print(f"\n✓ All folds for traj_size={traj_size} completed")
                print(f"✓ Unclean results written to {csv_unclean}")
                print(f"✓ Cleaned results written to {csv_cleaned}")
                print(f"✓ Combined results written to {csv_combined}")

                # Generate Excel report after each trajectory size completes
                print(f"\n{'='*60}")
                print(f"GENERATING AGGREGATED REPORT FOR TRAJECTORY SIZE = {traj_size}")
                print(f"{'='*60}")

                timestamp = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
                xlsx_path = evaluation_results_dir / f"benchmark_results_{timestamp}.xlsx"
                generate_excel_report(unclean_results, cleaned_results, xlsx_path)
                print(f"✓ Excel report saved to: {xlsx_path}")
                completed_sizes = sorted(set(r['traj_size'] for r in unclean_results))
                print(f"  Sheets completed so far: {[f'{s}__unclean' for s in completed_sizes] + [str(s) for s in completed_sizes]}")

                # Generate plots after each trajectory size
                plots_dir = evaluation_results_dir / "plots"
                generate_plots(unclean_results, cleaned_results, plots_dir)
                print(f"✓ Plots updated with results up to size={traj_size}")

    # =============================================================================
    # FINAL SUMMARY
    # =============================================================================

    # Create all-domains combined CSV file
    csv_all_combined = evaluation_results_dir / "results_all_domains_combined.csv"
    all_unclean = [dict(r, phase='unclean') for r in unclean_results]
    all_cleaned = [dict(r, phase='cleaned') for r in cleaned_results]
    all_combined_data = all_unclean + all_cleaned
    pd.DataFrame(all_combined_data).to_csv(csv_all_combined, index=False)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)
    print(f"\nTotal unclean results: {len(unclean_results)}")
    print(f"Total cleaned results: {len(cleaned_results)}")
    print(f"\nAll evaluation results saved to: {evaluation_results_dir}")
    print(f"  - Plots: {evaluation_results_dir / 'plots'}")
    print(f"  - Excel report: {xlsx_path}")
    print(f"  - Per-domain CSVs: {csv_unclean}, {csv_cleaned}, {csv_combined}")
    print(f"  - All-domains combined CSV: {csv_all_combined}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run PDDL action model learning benchmark')
    parser.add_argument('--domain', type=str, default='all',
                       help='Domain to run (blocksworld, hanoi, n_puzzle_typed, maze, or "all" for all domains)')
    parser.add_argument('--mode', type=str, default='fullyobs', choices=['masked', 'fullyobs'],
                       help='Mode to run: "masked" (PISAM/PO_ROSAME) or "fullyobs" (SAM/ROSAME)')

    args = parser.parse_args()

    # Determine which domains to run
    if args.domain == 'all':
        selected_domains = None  # Run all domains in domain_name_mappings
    else:
        # Validate domain name
        if args.domain not in domain_properties:
            print(f"Error: Unknown domain '{args.domain}'")
            print(f"Available domains: {list(domain_properties.keys())}")
            exit(1)
        selected_domains = [args.domain]

    main(selected_domains=selected_domains, mode=args.mode)