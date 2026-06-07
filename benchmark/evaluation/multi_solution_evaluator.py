"""
Evaluate ALL conflict-free models produced by the search, not just the best one.

Scans the conflict_free_models/ directory for model_0, model_1, ..., model_N
and evaluates each with all metrics:
  - Syntactic precision / recall (from AMLGym)
  - Problem solving ratio (from AMLGym)
  - Predicted applicability precision / recall (from AMLGym)
  - Predicted effects precision / recall (from AMLGym)

Produces an ordered list of results (one per solution, in order of discovery)
that matches the table format from the meeting notes.

Usage:
    from benchmark.evaluation.multi_solution_evaluator import evaluate_all_solutions

    results = evaluate_all_solutions(
        conflict_free_models_dir=fold_dir / "conflict_free_models",
        ref_domain_path=Path("reference.pddl"),
        test_problem_paths=["p1.pddl", "p2.pddl"],
        test_states_path=fold_dir / "predictive_power_test_states" / "test_states.json",
        planning_timeout=60,
    )
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional

from amlgym.metrics import syntactic_precision, syntactic_recall, problem_solving

from benchmark.evaluation.predictive_metrics import evaluate_predictive_power

logger = logging.getLogger(__name__)


def evaluate_all_solutions(
    conflict_free_models_dir: Path,
    ref_domain_path: Path,
    test_problem_paths: List[str],
    test_states_path: Path,
    planning_timeout: int = 60,
    output_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    Evaluate every conflict-free model in discovery order.

    Args:
        conflict_free_models_dir: Directory containing conflict_free_model_0/,
            conflict_free_model_1/, ..., each with a model.pddl.
        ref_domain_path: Path to the reference domain PDDL file.
        test_problem_paths: List of test problem PDDL paths.
        test_states_path: Path to the pre-generated test_states.json.
        planning_timeout: Timeout for the planner in problem_solving evaluation.
        output_dir: If provided, save results to all_solutions_metrics.json here.

    Returns:
        List of dicts, one per solution in discovery order. Each dict contains:
            solution_index, pred_app_precision, pred_app_recall,
            pred_eff_precision, pred_eff_recall, solving_ratio,
            precision_overall, recall_overall, ...
    """
    if not conflict_free_models_dir.exists():
        logger.warning(f"No conflict_free_models_dir: {conflict_free_models_dir}")
        return []

    # Find all conflict-free model directories, sorted by index
    model_dirs = sorted(
        [
            d
            for d in conflict_free_models_dir.iterdir()
            if d.is_dir() and d.name.startswith("conflict_free_model_")
        ],
        key=lambda d: int(d.name.split("_")[-1]),
    )

    # If no conflict-free models were found, check for a final_model
    # (saved when search exhausts resources before finding any solution)
    has_final_model_fallback = False
    final_model_dir = conflict_free_models_dir / "final_model"
    if not model_dirs and final_model_dir.exists() and (final_model_dir / "model.pddl").exists():
        model_dirs = [final_model_dir]
        has_final_model_fallback = True
        print(f"  [MULTI-EVAL] No conflict-free models found; evaluating final_model (resource-limited).")

    if not model_dirs:
        logger.warning("No conflict-free model directories found (and no final_model fallback).")
        return []

    print(f"  [MULTI-EVAL] Found {len(model_dirs)} {'final (non-conflict-free)' if has_final_model_fallback else 'conflict-free'} model(s) to evaluate.")
    all_results = []

    for model_dir in model_dirs:
        model_path = model_dir / "model.pddl"
        if not model_path.exists():
            logger.warning(f"No model.pddl in {model_dir.name}, skipping.")
            continue

        if model_dir.name == "final_model":
            solution_index = -1  # Sentinel: search ended without a conflict-free solution
        else:
            solution_index = int(model_dir.name.split("_")[-1])
        print(f"  [MULTI-EVAL] Evaluating solution {solution_index}{'  (final_model fallback)' if solution_index == -1 else ''}...")

        # --- Syntactic metrics ---
        try:
            precision = syntactic_precision(str(model_path), str(ref_domain_path))
            recall = syntactic_recall(str(model_path), str(ref_domain_path))
        except Exception as e:
            logger.error(f"Syntactic eval failed for solution {solution_index}: {e}")
            precision = {}
            recall = {}

        # --- Problem solving ---
        try:
            solving = problem_solving(
                str(model_path), str(ref_domain_path), test_problem_paths,
                timeout=planning_timeout,
            )
        except Exception as e:
            logger.error(f"Problem solving eval failed for solution {solution_index}: {e}")
            solving = {}

        # --- Predictive power ---
        predictive = evaluate_predictive_power(
            learned_model_path=str(model_path),
            ref_domain_path=str(ref_domain_path),
            test_states_path=str(test_states_path),
            test_problem_paths=test_problem_paths,
        )

        result = {
            "solution_index": solution_index,
            # Syntactic
            "precision_precs_pos": precision.get("precs_pos"),
            "precision_precs_neg": precision.get("precs_neg"),
            "precision_eff_pos": precision.get("eff_pos"),
            "precision_eff_neg": precision.get("eff_neg"),
            "precision_overall": precision.get("mean") if isinstance(precision, dict) else precision,
            "recall_precs_pos": recall.get("precs_pos"),
            "recall_precs_neg": recall.get("precs_neg"),
            "recall_eff_pos": recall.get("eff_pos"),
            "recall_eff_neg": recall.get("eff_neg"),
            "recall_overall": recall.get("mean") if isinstance(recall, dict) else recall,
            # Problem solving
            "solving_ratio": solving.get("solving_ratio"),
            "false_plans_ratio": solving.get("false_plans_ratio"),
            "unsolvable_ratio": solving.get("unsolvable_ratio"),
            "planning_timed_out_ratio": solving.get("timed_out"),
            # Predictive power
            **predictive,
        }
        all_results.append(result)

        # Log key metrics
        print(
            f"  [MULTI-EVAL] Solution {solution_index}: "
            f"solving={solving.get('solving_ratio', '?')}, "
            f"pred_app_p={predictive.get('pred_app_precision', '?')}, "
            f"pred_eff_p={predictive.get('pred_eff_precision', '?')}"
        )

    # Save results
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "all_solutions_metrics.json"
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  [MULTI-EVAL] Results saved to {output_path}")

    return all_results
