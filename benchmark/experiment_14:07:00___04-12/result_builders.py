"""
Result building and evaluation utilities for AMLGym experiments.
"""

from pathlib import Path
from typing import List


def evaluate_and_build_result(
    model: str,
    model_name: str,
    bench_name: str,
    fold: int,
    num_trajectories: int,
    gt_rate: int,
    test_problem_paths: List[str],
    phase: str,
    domain_ref_path: Path,
    testing_dir: Path,
    evaluate_model_func,
    null_metrics: dict,
    fold_work_dir: Path = None,
    total_transitions: int = None,
    total_gt_transitions: int = None,
    learning_metrics: dict = None
) -> dict:
    """
    Evaluate model and build result dictionary.
    
    Args:
        model: PDDL model string or None
        model_name: Name of the algorithm
        bench_name: Benchmark domain name
        fold: Fold number
        num_trajectories: Number of trajectories used
        gt_rate: GT injection rate
        test_problem_paths: List of test problem paths
        phase: 'unclean' or 'cleaned'
        domain_ref_path: Path to reference domain
        testing_dir: Testing directory
        evaluate_model_func: Function to evaluate model
        null_metrics: Dictionary of null metrics
        fold_work_dir: Optional fold working directory
        total_transitions: Total number of transitions used for learning
        total_gt_transitions: Total number of GT transitions used for learning
        learning_metrics: Dictionary with learning metrics (timeouts, runtimes, etc.)
        
    Returns:
        Result dictionary with all metrics
    """
    if model:
        print(f"  [DEBUG] Evaluating {model_name} ({phase})...")
        temp_path = testing_dir / f'{model_name}_{phase}_{bench_name}_fold{fold}.pddl'
        temp_path.write_text(model)

        # Save learned domain to fold directory
        if fold_work_dir:
            domain_save_path = fold_work_dir / f'learned_domain_{model_name}_{phase}.pddl'
            domain_save_path.write_text(model)
            print(f"  [DEBUG] Saved learned domain to {domain_save_path.name}")

        metrics = evaluate_model_func(str(temp_path), domain_ref_path, test_problem_paths)
        temp_path.unlink()
        print(f"  [DEBUG] Evaluation complete for {model_name}")
    else:
        print(f"  [DEBUG] No model for {model_name}, using null metrics")
        metrics = null_metrics

    # Extract timeout information from metrics
    timeout_during_planning = metrics.get('planning_timed_out_ratio', None) if metrics else None
    
    # Extract learning metrics if provided
    learning_time_seconds = None
    timeout_during_learning = None
    timeout_during_cleaning = None
    runtime_of_cleaning = None
    nodes_in_cleaning_tree = None
    
    if learning_metrics:
        learning_time_seconds = learning_metrics.get('learning_time_seconds', None)
        timeout_during_learning = learning_metrics.get('timeout_during_learning', None)
        # For cleaned phase, nodes_expanded represents cleaning tree nodes
        if phase == 'cleaned':
            nodes_in_cleaning_tree = learning_metrics.get('nodes_expanded', None)
            # If timeout during learning in cleaned phase, it's actually cleaning timeout
            if timeout_during_learning:
                timeout_during_cleaning = timeout_during_learning
                timeout_during_learning = None
            # Runtime of cleaning is the learning time for cleaned phase
            runtime_of_cleaning = learning_time_seconds

    result = {
        'domain': bench_name,
        'algorithm': model_name,
        'fold': fold,
        'num_trajectories': num_trajectories,
        'gt_rate': gt_rate,
        'problems_count': len(test_problem_paths),
        '_internal_phase': phase,
        'total_transitions': total_transitions,
        'total_gt_transitions': total_gt_transitions,
        'timeout_during_learning': timeout_during_learning,
        'timeout_during_cleaning': timeout_during_cleaning,
        'timeout_during_planning': timeout_during_planning,
        'runtime_of_learning_seconds': learning_time_seconds if phase == 'unclean' else None,
        'runtime_of_cleaning_seconds': runtime_of_cleaning,
        'nodes_in_cleaning_tree': nodes_in_cleaning_tree,
        **metrics
    }
    
    return result

