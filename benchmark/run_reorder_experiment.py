#!/usr/bin/env python3
"""
Reorder experiment: run fold 0 with trajectories [problem7, problem6, problem3]
instead of the original inner_0 order [problem3, problem7, problem6].

This tests whether trajectory ORDER within a solving subgroup affects the
conflict search's ability to find solving models.

Usage (from the VIP-vision-PDDL directory):
    python -m benchmark.run_reorder_experiment

Output goes to:
    benchmark/data/new_experiments/blocksworld/TO=300__largest__cv5/testing/
        fold0_numtrajs3_gtrate0/inner_99/

Compare with inner_0 (same trajectories, different order).
"""

import sys
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────
benchmark_path = Path(__file__).resolve().parent
sys.path.insert(0, str(benchmark_path.parent))  # so "from benchmark.…" works

from benchmark.experiment_running_helpers.run_fold import run_single_fold
from benchmark.amlgym_testing import evaluate_model, save_learning_metrics

# ── parameters (matching TO=300__largest__cv5 experiment exactly) ──────
DOMAIN_NAME = "blocksworld"
DOMAIN_REF_PATH = benchmark_path / "domains" / "blocksworld" / "blocksworld.pddl"
DATA_DIR = benchmark_path / "data" / "blocksworld" / "multi_problem_02-01-2026T14:16:59__model=gpt-5.1__steps=300__planner"
TRAJECTORIES_DIR = DATA_DIR / "training" / "trajectories"
TESTING_DIR = benchmark_path / "data" / "new_experiments" / "blocksworld" / "TO=300__largest__cv5" / "testing"

FOLD = 0
NUM_TRAJECTORIES = 3
GT_RATE = 0
MODE = "masked"

# Denoising / search params (from learning_metrics.json of inner_0)
CONFLICT_SEARCH_TIMEOUT = 300
PLANNING_TIMEOUT = 60
FLUENT_PATCH_COST = 1.0
FLUENT_PATCH_WEIGHT = 1.0
MODEL_PATCH_COST = 1.0
MODEL_CONSTRAINT_WEIGHT = 0.0
MAX_SEARCH_NODES = None
SEARCH_MODE = "dfs"
NODE_CHOOSING_STRATEGY = "model_patch_first"
CONFLICT_GROUP_STRATEGY = "largest"

# Reorder parameters
# seed 619 → random.sample(train_problem_dirs, 3) = [problem7, problem6, problem3]
# (verified: seed 42 gave inner_0's [problem3, problem7, problem6])
TRAJECTORY_SEED = 619
INNER_FOLD_IDX = 99  # distinct from existing inner_0..inner_4


def main():
    # Resolve problem directories (same as amlgym_testing.py line 827)
    problem_dirs = sorted([d for d in TRAJECTORIES_DIR.iterdir() if d.is_dir()])
    n_problems = len(problem_dirs)

    print(f"{'='*70}")
    print(f"REORDER EXPERIMENT: trajectories [problem7, problem6, problem3]")
    print(f"  (inner_0 used [problem3, problem7, problem6])")
    print(f"{'='*70}")
    print(f"  Domain:       {DOMAIN_NAME}")
    print(f"  Fold:         {FOLD}")
    print(f"  Num trajs:    {NUM_TRAJECTORIES}")
    print(f"  GT rate:      {GT_RATE}%")
    print(f"  Traj seed:    {TRAJECTORY_SEED}")
    print(f"  Inner idx:    {INNER_FOLD_IDX}")
    print(f"  Timeout:      {CONFLICT_SEARCH_TIMEOUT}s")
    print(f"  Strategy:     {CONFLICT_GROUP_STRATEGY} / {NODE_CHOOSING_STRATEGY}")
    print(f"  Output:       fold0_numtrajs3_gtrate0/inner_{INNER_FOLD_IDX}/")
    print(f"{'='*70}\n")

    # Verify seed produces the expected order
    import random
    indices = list(range(n_problems))
    random.seed(42 + FOLD)
    random.shuffle(indices)
    n_train = max(1, min(int(0.8 * n_problems), n_problems - 1))
    train_problem_dirs = [problem_dirs[i] for i in indices[:n_train]]
    random.seed(TRAJECTORY_SEED)
    selected = random.sample(train_problem_dirs, NUM_TRAJECTORIES)
    order = [d.name for d in selected]
    print(f"  Seed {TRAJECTORY_SEED} produces order: {order}")
    assert order == ["problem7", "problem6", "problem3"], \
        f"Unexpected order {order} — expected [problem7, problem6, problem3]"
    print(f"  ✓ Order verified\n")

    # Run the fold
    results = run_single_fold(
        fold=FOLD,
        problem_dirs=problem_dirs,
        n_problems=n_problems,
        num_trajectories=NUM_TRAJECTORIES,
        gt_rate=GT_RATE,
        domain_ref_path=DOMAIN_REF_PATH,
        testing_dir=TESTING_DIR,
        bench_name=DOMAIN_NAME,
        mode=MODE,
        evaluate_model_func=evaluate_model,
        save_learning_metrics_func=save_learning_metrics,
        conflict_search_timeout=CONFLICT_SEARCH_TIMEOUT,
        planning_timeout=PLANNING_TIMEOUT,
        fluent_patch_cost=FLUENT_PATCH_COST,
        fluent_patch_weight=FLUENT_PATCH_WEIGHT,
        model_patch_cost=MODEL_PATCH_COST,
        model_constraint_weight=MODEL_CONSTRAINT_WEIGHT,
        max_search_nodes=MAX_SEARCH_NODES,
        search_mode=SEARCH_MODE,
        node_choosing_strategy=NODE_CHOOSING_STRATEGY,
        conflict_group_strategy=CONFLICT_GROUP_STRATEGY,
        _inner_fold_idx=INNER_FOLD_IDX,
        _trajectory_seed=TRAJECTORY_SEED,
    )

    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    for r in results:
        phase = r.get("phase", "?")
        solving = r.get("solving_ratio", None)
        false_plans = r.get("false_plans_ratio", None)
        print(f"  {phase}: solving={solving}, false_plans={false_plans}")

    # Quick comparison pointer
    output_dir = TESTING_DIR / f"fold0_numtrajs3_gtrate0" / f"inner_{INNER_FOLD_IDX}"
    print(f"\nOutput saved to: {output_dir}")
    print(f"Compare with:    {TESTING_DIR / 'fold0_numtrajs3_gtrate0' / 'inner_0'}")


if __name__ == "__main__":
    main()
