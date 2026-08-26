"""
Shared predictive power evaluation helper.

Builds UPEnv simulators for each test problem that has S_test data,
then calls AMLGym's predictive_power() directly.

Used by both:
  - experiment_runner.evaluate_model()        (single-model evaluation)
  - multi_solution_evaluator.evaluate_all_solutions()  (batch evaluation)
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

from amlgym.metrics import predictive_power

from benchmark.evaluation.upenv_compat import CompatibleUPEnv

logger = logging.getLogger(__name__)

NULL_PREDICTIVE_RESULT: Dict[str, Optional[float]] = {
    "pred_app_precision": None,
    "pred_app_recall": None,
    "pred_eff_precision": None,
    "pred_eff_recall": None,
    "pred_undefined_reason": None,
}

#: ``pred_undefined_reason`` when the learned model is applicable in no test
#: state, so AMLGym averages over an empty operator set and returns NaN.
NO_APPLICABLE_OPERATOR = "no_applicable_operator"

#: ``pred_undefined_reason`` when no simulator could be built at all.
NO_SIMULATOR = "no_simulator"

#: ``pred_undefined_reason`` when ``predictive_power`` itself raised.
PREDICTIVE_POWER_FAILED = "predictive_power_failed"


def evaluate_predictive_power(
    learned_model_path: str,
    ref_domain_path: str,
    test_states_path: str,
    test_problem_paths: List[str],
) -> Dict[str, Optional[float]]:
    """
    Evaluate predictive power metrics for a single learned model.

    Builds UPEnv simulators for each test problem that has S_test data,
    then calls AMLGym's predictive_power() directly.

    Args:
        learned_model_path: Path to the learned domain PDDL file.
        ref_domain_path: Path to the reference (ground truth) domain PDDL file.
        test_states_path: Path to the test_states.json file.
        test_problem_paths: List of test problem PDDL paths.

    Returns:
        Dict with keys: pred_app_precision, pred_app_recall,
                        pred_eff_precision, pred_eff_recall.
        Values are None if evaluation could not be performed.
    """
    test_states_file = Path(test_states_path)
    if not test_states_file.exists():
        logger.warning(f"Test states file not found: {test_states_file}")
        return dict(NULL_PREDICTIVE_RESULT)

    with open(test_states_file, "r") as f:
        all_test_states = json.load(f)

    sim_learned_list = []
    sim_ref_list = []
    states_list = []

    for problem_path in test_problem_paths:
        problem_filename = Path(problem_path).name

        if problem_filename not in all_test_states:
            logger.warning(f"No test states for {problem_filename}, skipping.")
            continue

        problem_states = all_test_states[problem_filename]
        if not problem_states:
            continue

        try:
            sim_learned_list.append(CompatibleUPEnv(learned_model_path, problem_path))
            sim_ref_list.append(CompatibleUPEnv(ref_domain_path, problem_path))
            states_list.append(problem_states)
        except Exception as e:
            print(f"  [PRED] Failed to create CompatibleUPEnv for {problem_filename}: {e}")
            import traceback
            traceback.print_exc()

    if not sim_learned_list:
        logger.error(
            "No simulator could be built for '%s' against '%s' over %d test "
            "problem(s), so its predictive metrics are null. A learned domain "
            "that will not parse beside its own reference usually means the two "
            "spell a symbol differently.",
            learned_model_path, ref_domain_path, len(test_problem_paths),
        )
        print(f"  [PRED] ERROR: no simulators could be created — null predictive metrics.")
        return {**NULL_PREDICTIVE_RESULT, "pred_undefined_reason": NO_SIMULATOR}

    try:
        print(f"  [PRED] Running predictive_power() with {len(sim_learned_list)} simulators...")
        result = predictive_power(
            simulator_learned=sim_learned_list,
            simulator_ref=sim_ref_list,
            test_states=states_list,
            show_progress=False,
        )
        print(f"  [PRED] predictive_power() succeeded.")
        metrics = {
            "pred_app_precision": result["applicability"]["mean_precision"],
            "pred_app_recall": result["applicability"]["mean_recall"],
            "pred_eff_precision": result["predicted_effects"]["mean_precision"],
            "pred_eff_recall": result["predicted_effects"]["mean_recall"],
        }
        if any(v is None or math.isnan(v) for v in metrics.values()):
            logger.error(
                "'%s' is applicable in no test state, so AMLGym averaged over an "
                "empty operator set; its predictive metrics are undefined.",
                learned_model_path,
            )
            print("  [PRED] ERROR: model applicable nowhere — undefined predictive metrics.")
            return {
                **NULL_PREDICTIVE_RESULT,
                "pred_undefined_reason": NO_APPLICABLE_OPERATOR,
            }
        metrics["pred_undefined_reason"] = None
        return metrics
    except Exception as e:
        logger.error(f"predictive_power() failed: {e}")
        import traceback
        traceback.print_exc()
        return {**NULL_PREDICTIVE_RESULT, "pred_undefined_reason": PREDICTIVE_POWER_FAILED}
