"""
Statistics and metrics calculation utilities for AMLGym experiments.
"""

import json
from pathlib import Path
from typing import List, Tuple


def count_transitions_in_trajectory(trajectory_path: Path) -> int:
    """
    Count the number of transitions (operators) in a trajectory file.
    
    Args:
        trajectory_path: Path to .trajectory file
        
    Returns:
        Number of transitions (operators) in the trajectory
    """
    if not trajectory_path.exists():
        return 0
    
    with open(trajectory_path, 'r') as f:
        content = f.read()
        # Count operators - each operator represents one transition
        num_transitions = content.count("(operator:")
    
    return num_transitions


def count_total_transitions_and_gt(
    prepared_trajectories: List[Tuple[Path, Path, Path, set]],
) -> Tuple[int, int]:
    """
    Count total transitions and GT states across all prepared trajectories.

    Args:
        prepared_trajectories: List of
            (trajectory_path, masking_path, problem_pddl_path, gt_indices) tuples;
            GT states are read directly from each trajectory's gt_indices set.

    Returns:
        Tuple of (total_transitions, total_gt_states)
    """
    total_transitions = 0
    total_gt_states = 0
    
    for traj_path, _masking_path, _problem_pddl_path, gt_indices, *_ in prepared_trajectories:
        total_transitions += count_transitions_in_trajectory(traj_path)
        total_gt_states += len(gt_indices)

    return total_transitions, total_gt_states


def load_learning_metrics(fold_work_dir: Path) -> dict:
    """
    Load learning metrics from learning_metrics.json in the fold directory.

    Args:
        fold_work_dir: Directory where fold results are saved

    Returns:
        Dictionary with learning metrics, or empty dict if not found
    """
    metrics_file = fold_work_dir / "learning_metrics.json"
    
    if not metrics_file.exists():
        return {}
    
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Extract relevant fields
        result = {
            'learning_time_seconds': metrics.get('learning_time_seconds', None),
            'timeout_during_learning': metrics.get('terminated_by') == 'timeout_exceeded' if metrics.get('terminated_by') else None,
            'nodes_expanded': metrics.get('nodes_expanded', None),  # This is the cleaning tree nodes for denoising
            'actual_timeout_seconds': metrics.get('actual_timeout_seconds', None),  # Actual timeout used (includes defaults)
        }
        
        return result
    except Exception as e:
        print(f"  Warning: Failed to load learning metrics: {e}")
        return {}

