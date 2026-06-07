"""
Generate S_test (predictive power test states) for evaluating learned domain models.

Follows the procedure described in the AMLGym paper (Stern et al.):
  1. For each test problem, solve it with a planner using the reference domain.
  2. Execute the plan but interleave random actions with probability p_rnd.
  3. After a random action, replan from the resulting state.
  4. Collect all visited states as S_test.

The generated states are saved as a JSON file compatible with AMLGym's
predictive_power / applicability / predicted_effects metric functions.

Usage:
    from benchmark.evaluation.test_states_generator import generate_predictive_power_test_states

    test_states_path = generate_predictive_power_test_states(
        domain_ref_path=Path("blocksworld.pddl"),
        test_problem_paths=["problem1.pddl", "problem2.pddl"],
        output_dir=fold_work_dir / "predictive_power_test_states",
        num_trajectories_per_problem=50,
    )
"""

import contextlib
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import unified_planning
from unified_planning.io import PDDLReader
from unified_planning.model import Problem, UPState
from unified_planning.plans import ActionInstance
from unified_planning.shortcuts import OneshotPlanner, SequentialSimulator

from tarski.io import PDDLReader as TarskiPDDLReader
from tarski.grounding import LPGroundingStrategy

# Disable printing of planning engine credits
unified_planning.shortcuts.get_environment().credits_stream = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Planner configuration (same as AMLGym's gen_states_predictability.py)
# ---------------------------------------------------------------------------
_DOWNWARD_SEARCH_CFG = (
    "let(hff,ff(),"
    "let(hcea,cea(),"
    "lazy_greedy([hff,hcea],preferred=[hff,hcea])))"
)

_DEFAULT_MAX_PLANNING_TIME = 120       # seconds
_DEFAULT_MAX_REPLANNING_TIME = 60      # seconds for feasibility check after random action
_DEFAULT_MAX_RANDOM_TRIALS = 3         # max re-samples when random action makes problem infeasible
_DEFAULT_TRAJ_LEN_MIN = 5
_DEFAULT_TRAJ_LEN_MAX = 30
_DEFAULT_P_RND = 0.2
_DEFAULT_NUM_TRAJECTORIES_PER_PROBLEM = 50


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_planner_cfg(timeout: int, use_optimal: bool = False) -> dict:
    """Build a planner configuration dict for unified-planning."""
    if use_optimal:
        return {"name": "fast-downward-opt"}
    return {
        "name": "fast-downward",
        "params": {
            "fast_downward_search_config": _DOWNWARD_SEARCH_CFG,
            "fast_downward_search_time_limit": f"{timeout}s",
        },
    }


def _ground_actions(domain_path: str, problem_path: str) -> dict:
    """Ground all actions for a problem using tarski's LP grounder."""
    reader = TarskiPDDLReader(raise_on_error=True)
    reader.parse_domain(domain_path)
    reader.parse_instance(problem_path)
    grounder = LPGroundingStrategy(reader.problem)
    return grounder.ground_actions()


def _replan(
    problem: Problem,
    current_state: UPState,
    action_instance: ActionInstance,
    planner_cfg: dict,
    max_replanning_time: int,
) -> Optional[object]:
    """Check that executing `action_instance` keeps the problem solvable; return new plan or None."""
    problem = problem.clone()
    for fluent in problem.initial_values:
        problem.set_initial_value(fluent, current_state.get_value(fluent))

    with SequentialSimulator(problem=problem) as sim:
        next_state = sim.apply(current_state, action_instance)

    problem = problem.clone()
    for fluent in problem.initial_values:
        problem.set_initial_value(fluent, next_state.get_value(fluent))

    with contextlib.redirect_stdout(open(os.devnull, "w")):
        with OneshotPlanner(problem_kind=problem.kind, **planner_cfg) as planner:
            result = planner.solve(problem, timeout=max_replanning_time)
    return result.plan


def _generate_single_trajectory(
    problem: Problem,
    domain_path: str,
    problem_path: str,
    ground_actions: dict,
    p_rnd: float = _DEFAULT_P_RND,
    traj_len_max: int = _DEFAULT_TRAJ_LEN_MAX,
    max_planning_time: int = _DEFAULT_MAX_PLANNING_TIME,
    max_replanning_time: int = _DEFAULT_MAX_REPLANNING_TIME,
    max_random_trials: int = _DEFAULT_MAX_RANDOM_TRIALS,
) -> List[UPState]:
    """
    Generate a single trajectory and return the list of visited UPStates.

    Follows the AMLGym paper procedure: execute planned actions, with probability
    p_rnd substitute a random applicable action (if it keeps the problem solvable).
    """
    planner_cfg = _build_planner_cfg(max_planning_time)
    replan_cfg = _build_planner_cfg(max_replanning_time)

    with SequentialSimulator(problem=problem) as simulator:
        current_state = simulator.get_initial_state()
        states = [current_state]
        plan = None

        while len(states) < traj_len_max:
            # Compute a plan from the current state
            if plan is None:
                prob_clone = problem.clone()
                for fluent in prob_clone.initial_values:
                    prob_clone.set_initial_value(fluent, current_state.get_value(fluent))

                with contextlib.redirect_stdout(open(os.devnull, "w")):
                    with OneshotPlanner(problem_kind=prob_clone.kind, **planner_cfg) as planner:
                        result = planner.solve(prob_clone, timeout=max_planning_time)
                        plan = result.plan

            if plan is None:
                logger.debug("Planning failed; stopping trajectory generation.")
                break

            for action_instance in plan.actions:
                # Possibly execute a random action instead
                if random.random() < p_rnd:
                    applicable = [
                        (problem.action(k.lower()), [problem.object(o.lower()) for o in objs])
                        for k, params in ground_actions.items()
                        for objs in params
                        if simulator._is_applicable(
                            current_state,
                            problem.action(k.lower()),
                            [problem.object(o.lower()) for o in objs],
                        )
                    ]
                    applicable = sorted(applicable, key=lambda x: f"{x[0]} - {x[1]}")

                    # Try random actions until one preserves solvability
                    trial = 0
                    new_plan = None
                    while trial < max_random_trials:
                        action, params = random.choice(applicable)
                        rand_action = ActionInstance(action, params)
                        new_plan = _replan(
                            problem, current_state, rand_action, replan_cfg, max_replanning_time
                        )
                        trial += 1
                        if new_plan is not None:
                            break

                    if new_plan is None:
                        # No feasible random action found; skip randomization this step
                        logger.debug("No feasible random action found; continuing with plan.")
                    else:
                        # Execute the random action
                        current_state = simulator.apply(current_state, rand_action)
                        states.append(current_state)
                        plan = new_plan  # use the replanned plan from here
                        break  # restart the plan-following loop

                # Execute the planned action
                current_state = simulator.apply(current_state, action_instance)
                if current_state is None:
                    logger.warning(f"Action {action_instance} failed; stopping trajectory.")
                    return states
                states.append(current_state)

            # Check if we finished the plan
            if plan is not None and (
                len(plan.actions) == 0 or action_instance == plan.actions[-1]
            ):
                logger.debug("Goal state reached.")
                break

    return states


def _upstate_to_literals(state: UPState) -> List[str]:
    """Convert a UPState to a list of string literals in AMLGym format.

    Positive fluents:  '(predicate_name obj1 obj2)'
    Negative fluents:  '(not (predicate_name obj1 obj2))'
    """
    literals = []
    for fluent_expr, value in state._values.items():
        name = fluent_expr.fluent().name
        objs = [str(o) for o in fluent_expr.args]
        if objs:
            formatted = f"({name} {' '.join(objs)})"
        else:
            formatted = f"({name})"

        if value.is_true():
            literals.append(formatted)
        else:
            literals.append(f"(not {formatted})")
    return literals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_predictive_power_test_states(
    domain_ref_path: Path,
    test_problem_paths: List[str],
    output_dir: Path,
    num_trajectories_per_problem: int = _DEFAULT_NUM_TRAJECTORIES_PER_PROBLEM,
    p_rnd: float = _DEFAULT_P_RND,
    traj_len_min: int = _DEFAULT_TRAJ_LEN_MIN,
    traj_len_max: int = _DEFAULT_TRAJ_LEN_MAX,
    max_planning_time: int = _DEFAULT_MAX_PLANNING_TIME,
    seed: int = 42,
) -> Path:
    """
    Generate predictive power test states (S_test) for a set of test problems.

    For each test problem, generates multiple trajectories using the reference
    domain and a planner with random action interleaving. All visited states
    are collected as S_test.

    Args:
        domain_ref_path: Path to the reference (ground truth) domain PDDL file.
        test_problem_paths: List of paths to PDDL test problem files.
        output_dir: Directory to save the generated test_states.json.
        num_trajectories_per_problem: Number of trajectories to generate per test problem.
        p_rnd: Probability of executing a random action instead of the planned one.
        traj_len_min: Minimum trajectory length (retries if shorter).
        traj_len_max: Maximum trajectory length.
        max_planning_time: Planner timeout in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Path to the generated test_states.json file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_states.json"

    # Skip if already generated
    if output_path.exists():
        logger.info(f"S_test already exists at {output_path}, skipping generation.")
        return output_path

    UPState.MAX_ANCESTORS = None
    reader = PDDLReader()
    domain_str = str(domain_ref_path)

    random.seed(seed)

    # Collect states per problem: {problem_filename: [state_literals_list, ...]}
    all_test_states: Dict[str, List[List[str]]] = {}

    for problem_path in test_problem_paths:
        problem_path_str = str(problem_path)
        problem_filename = Path(problem_path).name
        logger.info(f"Generating S_test for {problem_filename}...")

        # Parse and ground
        problem = reader.parse_problem(domain_str, problem_path_str)
        ground_acts = _ground_actions(domain_str, problem_path_str)

        collected_states: List[List[str]] = []
        seen_state_sets = set()  # deduplicate states

        trajectories_generated = 0
        max_attempts = num_trajectories_per_problem * 3  # allow retries for short trajectories
        attempts = 0

        while trajectories_generated < num_trajectories_per_problem and attempts < max_attempts:
            attempts += 1
            try:
                # Re-parse problem each time (planner modifies it)
                prob = reader.parse_problem(domain_str, problem_path_str)
                states = _generate_single_trajectory(
                    problem=prob,
                    domain_path=domain_str,
                    problem_path=problem_path_str,
                    ground_actions=ground_acts,
                    p_rnd=p_rnd,
                    traj_len_max=traj_len_max,
                    max_planning_time=max_planning_time,
                )
            except Exception as e:
                logger.debug(f"Trajectory generation failed: {e}")
                continue

            if len(states) < traj_len_min:
                logger.debug(
                    f"Trajectory too short ({len(states)} < {traj_len_min}), retrying."
                )
                continue

            trajectories_generated += 1

            # Convert states to string literals and deduplicate
            for state in states:
                literals = _upstate_to_literals(state)
                # Use frozenset of positive literals for dedup (neg literals are deterministic)
                pos_key = frozenset(l for l in literals if not l.startswith("(not "))
                if pos_key not in seen_state_sets:
                    seen_state_sets.add(pos_key)
                    collected_states.append(literals)

        all_test_states[problem_filename] = collected_states
        logger.info(
            f"  {problem_filename}: {len(collected_states)} unique states "
            f"from {trajectories_generated} trajectories"
        )

    # Save
    with open(output_path, "w") as f:
        json.dump(all_test_states, f, indent=2)

    total_states = sum(len(v) for v in all_test_states.values())
    logger.info(f"S_test saved to {output_path} ({total_states} total states)")
    print(
        f"  [S_TEST] Generated {total_states} unique test states "
        f"from {len(test_problem_paths)} problems → {output_path.name}"
    )

    return output_path


def load_test_states(test_states_path: Path) -> Dict[str, List[List[str]]]:
    """
    Load previously generated test states from JSON.

    Args:
        test_states_path: Path to test_states.json.

    Returns:
        Dict mapping problem filename → list of states,
        where each state is a list of string literals.
    """
    with open(test_states_path, "r") as f:
        return json.load(f)
