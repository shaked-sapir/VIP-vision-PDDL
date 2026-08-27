"""Generate S_test (predictive power test states) for evaluating learned models.

Follows the procedure described in the AMLGym paper (Stern et al.):

  1. For each test problem, solve it with a planner using the reference domain.
  2. Execute the plan but interleave random actions with probability p_rnd.
  3. After a random action, replan from the resulting state.
  4. Collect all visited states as S_test.

Steps 1-3 are :func:`src.trace_generation.sources.walk_problem`; this module owns
only the S_test-specific part: repeat until enough trajectories are collected,
drop the short ones, deduplicate states, and write them in AMLGym's literal
format, which is not the eval schema the trace-generation corpus uses.

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

import json
import math
import os
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Optional

from src.trace_generation.sources import (
    ground_actions_of,
    parse_problem,
    walk_problem,
)
from src.trace_generation.up_bridge import ground_fluents

logger = logging.getLogger(__name__)

_DEFAULT_TRAJ_LEN_MIN = 5
_DEFAULT_TRAJ_LEN_MAX = 30
_DEFAULT_P_RND = 0.2
_DEFAULT_NUM_TRAJECTORIES_PER_PROBLEM = 50
#: Walks per FOLD, split across its test problems -- upstream AMLGym's
#: ``TRAJ_PER_DOMAIN``. A per-problem rate multiplies with the test-split size;
#: at 500 test problems the old 50-per-problem rate meant 25,000 walks.
_DEFAULT_TRAJECTORIES_PER_FOLD = 100
#: Consecutive same-length rejected walks after which a problem is abandoned.
#: The planner is deterministic on a fixed problem and only ``p_rnd`` of steps
#: are random, so a problem whose walk repeats its length is plan-bound: more
#: attempts cannot lengthen it.
_ABANDON_AFTER_REPEATS = 2
_DEFAULT_MAX_PLANNING_TIME = 120       # seconds
_DEFAULT_MAX_REPLANNING_TIME = 60      # seconds, for the post-random-action check
_DEFAULT_MAX_RANDOM_TRIALS = 3         # re-samples when a random action blocks the goal


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _trajectory_states(
    problem: Any,
    ground_actions: dict,
    rng: random.Random,
    p_rnd: float,
    traj_len_max: int,
    max_planning_time: int,
    max_replanning_time: int,
    max_random_trials: int,
) -> List[Any]:
    """Walk the problem once and return the states it visited, in order.

    Returns an empty list when the walk yields no transition at all; such a
    trajectory is below any sensible ``traj_len_min`` and the caller drops it.
    """
    steps = list(walk_problem(
        problem,
        ground_actions,
        rng=rng,
        p_rnd=p_rnd,
        max_steps=max(traj_len_max - 1, 0),
        preserve_solvability=True,
        stop_at_goal=True,
        max_planning_time=max_planning_time,
        max_replanning_time=max_replanning_time,
        max_random_trials=max_random_trials,
    ))
    if not steps:
        return []
    return [steps[0].prev_state] + [step.next_state for step in steps]


def _upstate_to_literals(state: Any, fluents: Sequence[Any]) -> List[str]:
    """Convert a UPState to its true fluents, as '(predicate_name obj1 obj2)'.

    False fluents are omitted; ``CompatibleUPEnv._state_from_literals`` rebuilds
    a state from the positive literals over an all-false base, so absence is how
    falsity is expressed.

    Every fluent is evaluated explicitly, because a UPState may store only its
    difference from an ancestor rather than the whole assignment.
    """
    literals = []
    for fluent_expr in fluents:
        if not state.get_value(fluent_expr).is_true():
            continue
        name = fluent_expr.fluent().name
        objs = [str(o) for o in fluent_expr.args]
        literals.append(f"({name} {' '.join(objs)})" if objs else f"({name})")
    return literals


def _collect_problem_states(
    domain_str: str,
    problem_path_str: str,
    rng: random.Random,
    num_trajectories: int,
    p_rnd: float,
    traj_len_min: int,
    traj_len_max: int,
    max_planning_time: int,
) -> List[List[str]]:
    """Walk one problem repeatedly, returning its deduplicated visited states."""
    ground_acts = ground_actions_of(Path(domain_str), Path(problem_path_str))

    collected_states: List[List[str]] = []
    seen_state_sets = set()

    trajectories_generated = 0
    attempts = 0
    max_attempts = num_trajectories * 3  # allow retries for short trajectories
    last_rejected_length = None
    repeats = 0

    while trajectories_generated < num_trajectories and attempts < max_attempts:
        attempts += 1
        # Re-parse the problem each time; planning mutates it.
        problem = parse_problem(Path(domain_str), Path(problem_path_str))
        fluents = ground_fluents(problem)
        try:
            states = _trajectory_states(
                problem, ground_acts, rng, p_rnd, traj_len_max,
                max_planning_time, _DEFAULT_MAX_REPLANNING_TIME,
                _DEFAULT_MAX_RANDOM_TRIALS,
            )
        except Exception as e:
            logger.debug(f"Trajectory generation failed: {e}")
            continue

        if len(states) < traj_len_min:
            repeats = repeats + 1 if len(states) == last_rejected_length else 0
            last_rejected_length = len(states)
            if repeats >= _ABANDON_AFTER_REPEATS:
                logger.debug(
                    f"Trajectory length {len(states)} repeated "
                    f"{repeats + 1}x and stays under {traj_len_min}; the walk is "
                    f"plan-bound, abandoning this problem."
                )
                break
            logger.debug(
                f"Trajectory too short ({len(states)} < {traj_len_min}), retrying."
            )
            continue

        last_rejected_length = None
        repeats = 0
        trajectories_generated += 1

        for state in states:
            literals = _upstate_to_literals(state, fluents)
            key = frozenset(literals)
            if key not in seen_state_sets:
                seen_state_sets.add(key)
                collected_states.append(literals)

    return collected_states


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_predictive_power_test_states(
    domain_ref_path: Path,
    test_problem_paths: List[str],
    output_dir: Path,
    num_trajectories_per_problem: Optional[int] = None,
    trajectories_per_fold: int = _DEFAULT_TRAJECTORIES_PER_FOLD,
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
        num_trajectories_per_problem: Walks per test problem. ``None`` (the
            default) derives it from ``trajectories_per_fold``; pass an int to
            pin the per-problem rate instead.
        trajectories_per_fold: Walk budget for the whole fold, split evenly
            across its test problems. Ignored when
            ``num_trajectories_per_problem`` is given.
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

    if num_trajectories_per_problem is None:
        num_trajectories_per_problem = max(
            1, math.ceil(trajectories_per_fold / max(1, len(test_problem_paths)))
        )
        logger.info(
            f"S_test budget: {trajectories_per_fold} walks over "
            f"{len(test_problem_paths)} problems -> "
            f"{num_trajectories_per_problem} per problem"
        )

    rng = random.Random(seed)
    domain_str = str(domain_ref_path)

    # Collect states per problem: {problem_filename: [state_literals_list, ...]}
    all_test_states: Dict[str, List[List[str]]] = {}

    for problem_path in test_problem_paths:
        problem_filename = Path(problem_path).name
        logger.info(f"Generating S_test for {problem_filename}...")

        collected_states = _collect_problem_states(
            domain_str, str(problem_path), rng,
            num_trajectories_per_problem, p_rnd,
            traj_len_min, traj_len_max, max_planning_time,
        )

        all_test_states[problem_filename] = collected_states
        logger.info(
            f"  {problem_filename}: {len(collected_states)} unique states "
            f"from up to {num_trajectories_per_problem} trajectories"
        )

    # Written via a temp file and renamed: the cache guard above is a
    # check-then-write, and this dir is now shared by every training size of the
    # fold, so a reader must never observe a half-written file.
    tmp_path = output_path.with_suffix(f".tmp{os.getpid()}")
    with open(tmp_path, "w") as f:
        json.dump(all_test_states, f, indent=2)
    os.replace(tmp_path, output_path)

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
