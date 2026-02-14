"""
Statistics and metrics calculation utilities for AMLGym experiments.
"""

import json
import math
from pathlib import Path
from typing import List, Tuple

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser


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


def count_gt_states_in_trajectory(trajectory_path: Path, domain_path: Path, gt_rate: int) -> int:
    """
    Count the number of GT (ground truth) states in a trajectory based on GT injection rate.
    GT states are calculated based on the gt_rate percentage and trajectory length,
    following the same logic as inject_gt_states_by_percentage.
    
    Args:
        trajectory_path: Path to .trajectory file
        domain_path: Path to domain PDDL file
        gt_rate: Percentage of states to inject as GT (0-100)
        
    Returns:
        Number of GT states in the trajectory
    """
    if not trajectory_path.exists():
        return 0
    
    try:
        # Parse trajectory to get number of states
        domain = DomainParser(domain_path).parse_domain()
        parser = TrajectoryParser(domain)
        current_observation = parser.parse_trajectory(trajectory_path)
        num_steps = len(current_observation.components)
        num_states = num_steps + 1  # Including initial state
        
        # Calculate GT states using same logic as inject_gt_states_by_percentage
        gt_state_indices = set()
        gt_state_indices.add(0)  # Initial state is always GT
        
        if gt_rate > 0:
            # Calculate total number of GT states needed
            num_gt_states = max(1, math.ceil(num_states * gt_rate / 100.0))
            
            if num_gt_states > 1:
                # Calculate interval for even spacing
                interval = num_states / num_gt_states
                
                # Generate evenly spaced GT state indices
                for i in range(num_gt_states):
                    idx = int(i * interval)
                    if idx < num_states:
                        gt_state_indices.add(idx)
        
        return len(gt_state_indices)
    except Exception as e:
        print(f"  Warning: Failed to count GT states: {e}")
        return 0


def count_total_transitions_and_gt(
    prepared_trajectories: List[Tuple[Path, Path, Path]],
    domain_path: Path,
    gt_rate: int
) -> Tuple[int, int]:
    """
    Count total transitions and GT transitions across all prepared trajectories.
    
    Args:
        prepared_trajectories: List of (trajectory_path, masking_path, problem_pddl_path)
        domain_path: Path to domain PDDL file
        gt_rate: Percentage of states to inject as GT (0-100)
        
    Returns:
        Tuple of (total_transitions, total_gt_states)
    """
    total_transitions = 0
    total_gt_states = 0
    
    for traj_path, masking_path, problem_pddl_path in prepared_trajectories:
        transitions = count_transitions_in_trajectory(traj_path)
        total_transitions += transitions
        
        gt_states = count_gt_states_in_trajectory(traj_path, domain_path, gt_rate)
        total_gt_states += gt_states
    
    return total_transitions, total_gt_states


def load_learning_metrics(fold_work_dir: Path, phase: str, algorithm_name: str) -> dict:
    """
    Load learning metrics from JSON file.
    
    Args:
        fold_work_dir: Directory where fold results are saved
        phase: 'unclean' or 'cleaned'
        algorithm_name: Algorithm name (e.g., 'PISAM', 'SAM', 'PO_ROSAME', 'ROSAME')
        
    Returns:
        Dictionary with learning metrics, or empty dict if not found
    """
    # For cleaned phase with denoising, the metrics might be in a specific file
    # For now, we'll look for learning_metrics.json in the fold directory
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

