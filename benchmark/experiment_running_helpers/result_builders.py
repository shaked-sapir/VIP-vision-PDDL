"""
Result building and evaluation utilities for AMLGym experiments.

Every algorithm produces the same base record (identity + data context + AMLGym
metrics) plus a nested ``algorithm_specific`` dict it fills with its own extras.
See ``result_schema`` for the field lists.
"""

from pathlib import Path
from typing import List, Optional

from benchmark.experiment_running_helpers.result_schema import ALGORITHM_SPECIFIC_KEY


def evaluate_and_build_result(
    model: str,
    model_name: str,
    bench_name: str,
    fold: int,
    num_trajectories: int,
    gt_rate: int,
    test_problem_paths: List[str],
    domain_ref_path: Path,
    testing_dir: Path,
    evaluate_model_func,
    null_metrics: dict,
    fold_work_dir: Path = None,
    total_transitions: int = None,
    total_gt_transitions: int = None,
    learning_time_seconds: Optional[float] = None,
    algorithm_specific: Optional[dict] = None,
    planning_timeout: int = 60,
    test_states_path: str = None,
) -> dict:
    """Evaluate a learned model and build its base result record.

    Args:
        model: PDDL model string or None.
        model_name: Algorithm name (e.g. ``CDPS``, ``ROSAME``) — used as the row's
            ``algorithm`` and the ``learned_domain_<name>.pddl`` filename.
        learning_time_seconds: Wall-clock learning time for this algorithm (base).
        algorithm_specific: Algorithm-owned extras (nested under
            ``algorithm_specific``); CDPS passes its search stats, baselines their
            own report dict.
        (other args as before)

    Returns:
        A base result dict + nested ``algorithm_specific``.
    """
    if model:
        print(f"  [DEBUG] Evaluating {model_name}...")
        temp_dir = fold_work_dir if fold_work_dir else testing_dir
        temp_path = temp_dir / f'{model_name}_{bench_name}_fold{fold}.pddl'
        temp_path.write_text(model)

        if fold_work_dir:
            domain_save_path = fold_work_dir / f'learned_domain_{model_name}.pddl'
            domain_save_path.write_text(model)
            print(f"  [DEBUG] Saved learned domain to {domain_save_path.name}")

        metrics = evaluate_model_func(
            str(temp_path), domain_ref_path, test_problem_paths,
            planning_timeout=planning_timeout,
            test_states_path=test_states_path,
        )
        temp_path.unlink()
        print(f"  [DEBUG] Evaluation complete for {model_name}")
    else:
        print(f"  [DEBUG] No model for {model_name}, using null metrics")
        metrics = null_metrics

    result = {
        'domain': bench_name,
        'algorithm': model_name,
        'fold': fold,
        'num_trajectories': num_trajectories,
        'gt_rate': gt_rate,
        'problems_count': len(test_problem_paths),
        'total_transitions': total_transitions,
        'total_gt_transitions': total_gt_transitions,
        'learning_time_seconds': learning_time_seconds,
        **metrics,
        ALGORITHM_SPECIFIC_KEY: algorithm_specific or {},
    }
    return result
