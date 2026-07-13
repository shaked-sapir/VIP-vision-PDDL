"""Model evaluation and learning metrics persistence.

Extracted from experiment_runner.py to keep the main experiment loop lean.
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

from amlgym.metrics import syntactic_precision, syntactic_recall, problem_solving

from benchmark.evaluation.predictive_metrics import evaluate_predictive_power


def save_learning_metrics(
    output_dir: Path,
    report: dict,
    trajectory_mapping: Optional[Dict[str, str]] = None,
) -> dict:
    """Save learning metrics to JSON file.

    Args:
        output_dir: Directory to write learning_metrics.json into.
        report: Report dict from the learner.
        trajectory_mapping: Optional mapping of trajectory names to paths.

    Returns:
        The metrics dict that was persisted.
    """
    metrics = {
        "learning_time_seconds": report.get("total_time_seconds", None),
        "max_depth": report.get("max_depth", None),
        "nodes_expanded": report.get("nodes_expanded", None),
        "terminated_by": report.get("terminated_by", None),
        "conflict_free_model_count": report.get("conflict_free_model_count", None),
        "actual_timeout_seconds": report.get("actual_timeout_seconds", None),
        "fluent_patch_cost": report.get("fluent_patch_cost", None),
        "fluent_patch_weight": report.get("fluent_patch_weight", None),
        "model_patch_cost": report.get("model_patch_cost", None),
        "model_constraint_weight": report.get("model_constraint_weight", None),
        "max_search_nodes": report.get("max_search_nodes", None),
        "search_mode": report.get("search_mode", None),
        "node_choosing_strategy": report.get("node_choosing_strategy", None),
        "conflict_group_strategy": report.get("conflict_group_strategy", None),
        "fluent_branch_mode": report.get("fluent_branch_mode", None),
        "best_model_constraints": report.get("final_model_constraints", None),
        "best_fluent_patches": report.get("final_fluent_patches", None),
        "conflict_free_solutions_summary": report.get("conflict_free_solutions_summary", None),
    }

    if trajectory_mapping:
        metrics["trajectory_mapping"] = trajectory_mapping

    with open(output_dir / "learning_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def evaluate_model(
    model_path: str,
    domain_ref_path: Path,
    test_problems: List[str],
    planning_timeout: int = 60,
    profiler=None,
    test_states_path: str = None,
) -> dict:
    """Evaluate a learned model against a reference domain.

    Handles AMLGym SimpleDomainReader race conditions with retries.

    Args:
        model_path: Path to learned model PDDL file.
        domain_ref_path: Path to reference domain PDDL file.
        test_problems: List of test problem paths.
        planning_timeout: Timeout in seconds for planning during evaluation.
        profiler: Optional TimingProfiler instance for detailed timing.
        test_states_path: Path to test_states.json for predictive power metrics.

    Returns:
        Dictionary of evaluation metrics.
    """

    def _time_metric(metric_name, func):
        """Helper to time a metric computation."""
        if profiler:
            start = time.perf_counter()
            result = func()
            elapsed = time.perf_counter() - start
            profiler.add_detailed_timing(
                'eval_metrics',
                metric_name,
                elapsed,
                {'model_path': model_path, 'num_test_problems': len(test_problems)}
            )
            return result
        return func()

    max_retries = 5
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(random.uniform(0.1, 0.5))

            precision = _time_metric(
                'syntactic_precision',
                lambda: syntactic_precision(model_path, str(domain_ref_path)),
            )
            recall = _time_metric(
                'syntactic_recall',
                lambda: syntactic_recall(model_path, str(domain_ref_path)),
            )
            problem_solving_result = _time_metric(
                'problem_solving',
                lambda: problem_solving(
                    model_path, str(domain_ref_path), test_problems,
                    timeout=planning_timeout,
                ),
            )
            break

        except (FileNotFoundError, ValueError, IndexError) as e:
            if attempt < max_retries - 1:
                clean_file = f"{domain_ref_path}_clean"
                try:
                    if Path(clean_file).exists():
                        Path(clean_file).unlink()
                except OSError as cleanup_err:
                    print(f"Warning: failed to remove stale clean file {clean_file}: {cleanup_err}")
                continue
            else:
                print(f"Warning: Evaluation failed after {max_retries} attempts: {e}")
                precision = None
                recall = None
                problem_solving_result = None

    # Predictive power metrics
    if test_states_path:
        predictive = evaluate_predictive_power(
            model_path, str(domain_ref_path), test_states_path, test_problems,
        )
    else:
        predictive = {
            "pred_app_precision": None,
            "pred_app_recall": None,
            "pred_eff_precision": None,
            "pred_eff_recall": None,
        }

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
        'planning_timed_out_ratio': problem_solving_result.get('timed_out') if isinstance(problem_solving_result, dict) else None,
        **predictive,
    }
