import csv
from pathlib import Path
from datetime import datetime
import random

import pandas as pd
import matplotlib.pyplot as plt

from amlgym.benchmarks import *
from amlgym.algorithms import *
from amlgym.metrics import *

from benchmark.amlgym_models.NOISY_PISAM import NOISY_PISAM
from benchmark.amlgym_models.PISAM import PISAM
from benchmark.amlgym_models.PO_ROSAME import PO_ROSAME
from src.utils.pddl import propagate_frame_axioms_in_trajectory

# =============================================================================
# CONFIG & PATHS
# =============================================================================

benchmark_path = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark")

print_metrics()

# Each domain -> list of experiment directories holding trajectories
experiment_data_dirs = {
    "blocksworld": [
        # "experiment_30-11-2025T12:47:58__steps=10",
        # "experiment_30-11-2025T13:03:16__steps=25",
        "multi_problem_04-12-2025T12:00:44__model=gpt-5.1__steps=50__planner"
    ],
    "n_puzzle_typed": [
        # "experiment_30-11-2025T13:17:05__steps=10",
        # "experiment_30-11-2025T13:28:43__steps=25",
        "experiment_30-11-2025T13:28:47__steps=50"
    ],
    "hanoi": [
        # "experiment_30-11-2025T19:11:13__steps=10",
        # "experiment_01-12-2025T10:37:42__steps=25",
        "experiment_01-12-2025T10:38:09__steps=50"
    ],
    "hiking": [
        # "experiment_01-12-2025T01:41:34__steps=10",
        # "experiment_01-12-2025T02:03:07__steps=25",
        "experiment_01-12-2025T02:03:49__steps=50",
    ],
    "maze": [
        "experiment_03-12-2025T13:23:27__steps=10"
    ]
}

# Mapping from your internal domain labels to benchmark names
domain_name_mappings = {
    # 'hiking': 'hiking',
    # 'maze': 'maze',
    # 'hanoi': 'hanoi',
    'blocksworld': 'blocksworld',
    # 'n_puzzle_typed': 'npuzzle',
}

# Domains that are not directly in amlgym's registry (paths & problems)
not_in_amlgym_domains = {
    'maze': {
        "trajectory_training_problem": "problem0",
        "domain_path": benchmark_path / 'domains' / 'maze' / 'maze.pddl',
        "problems_paths": sorted(
            str(p) for p in Path(
                "/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/domains/maze/problems"
            ).glob("problem*.pddl") if "problem0" not in str(p)
        )
    },
    "hanoi": {
        "trajectory_training_problem": "problem0",  # for documenting, not to put in the problems to be solved
        "domain_path": benchmark_path / 'domains' / 'hanoi' / 'hanoi.pddl',
        "problems_paths":
            sorted(
                str(p) for p in Path(
                    "/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/domains/hanoi/problems"
                ).glob("problem*.pddl") if "problem0" not in str(p)
            )
            +
            sorted(
                str(p) for p in Path(
                    "/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/domains/hanoi/problems_test"
                ).glob("test*.pddl") if "test_problem0" not in str(p)
            )
    },
    "hiking": {
        "trajectory_training_problem": "problem2",
        "domain_path": benchmark_path / 'domains' / 'hiking' / 'hiking.pddl',
        "problems_paths": sorted(
            str(p) for p in Path(
                "/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/domains/hiking/problems"
            ).glob("problem*.pddl") if "problem2" not in str(p)
        )
    },
    'blocksworld': {
        "trajectory_training_problem": "problem7",
        "domain_path": benchmark_path / 'domains' / 'blocksworld' / 'blocksworld.pddl',
        "problems_paths": sorted(
            str(p) for p in Path(
                benchmark_path / 'domains' / 'blocksworld' / 'test_problems'
            ).glob("problem*.pddl")
        )
    }
}

# =============================================================================
# EXPERIMENT LOOP WITH CV & VARYING #TRAJECTORIES
# =============================================================================

all_results = []

# number of CV folds
N_FOLDS = 5

# all metric keys we collect per run
metric_cols = [
    "precision_precs_pos",
    "precision_precs_neg",
    "precision_eff_pos",
    "precision_eff_neg",
    "precision_overall",
    "recall_precs_pos",
    "recall_precs_neg",
    "recall_eff_pos",
    "recall_eff_neg",
    "recall_overall",
    "problems_count",
    "solving_ratio",
    "false_plans_ratio",
    "unsolvable_ratio",
    "timed_out",
]

for domain_name, bench_name in domain_name_mappings.items():
    domain_ref_path = not_in_amlgym_domains[domain_name]["domain_path"]

    for dir_name in experiment_data_dirs[domain_name]:
        trajectories_dir = Path(
            benchmark_path / 'data' / domain_name / dir_name / 'training' / 'trajectories'
        )
        problem_dirs = sorted([d for d in trajectories_dir.iterdir() if d.is_dir()])
        n_problems = len(problem_dirs)

        # Get all problem directories (each contains trajectory + problem files)

        if n_problems < 2:
            raise ValueError(f"Domain {domain_name} has too few problems ({n_problems}) for 80/20 CV.")

        if n_problems == 0:
            print(f"WARNING: no problem directories found for {domain_name} / {dir_name}, skipping.")
            continue

        # how many trajectories per learning run (same as number of problems since 1 traj per problem)
        traj_settings = sorted({1})
        # traj_settings = list(reversed(sorted({1, min(3, n_problems), min(5,int(0.8*n_problems)), int(0.8*n_problems)})))
        # traj_settings = sorted({1, min(3, n_problems), min(5,int(0.8*n_problems)), int(0.8*n_problems)})

        print(f"\n{'=' * 80}")
        print(f"Domain: {bench_name} | data dir: {dir_name}")
        print(f"Total problems available: {n_problems}")
        print(f"Problem directories: {[d.name for d in problem_dirs]}")
        print(f"Trajectory settings to evaluate: {traj_settings}")
        print(f"Cross-validation folds: {N_FOLDS}")
        print(f"{'=' * 80}")

        base_indices = list(range(n_problems))

        for fold in range(N_FOLDS):
            if fold != 0:
                continue # debug only
            indices = base_indices[:]
            random.seed(42 + fold)  # deterministic per fold
            random.shuffle(indices)

            n_train = max(1, int(round(0.8 * n_problems)))
            if n_train >= n_problems:
                n_train = n_problems - 1

            train_idx = indices[:n_train]
            test_idx = indices[n_train:]

            # Get trajectory and problem paths from the selected problem directories
            train_problem_dirs = [problem_dirs[i] for i in train_idx]
            test_problem_dirs = [problem_dirs[i] for i in test_idx]

            # Extract trajectory paths for training
            train_trajectories = []
            for prob_dir in train_problem_dirs:
                traj_files = list(prob_dir.glob("*.trajectory"))
                if traj_files:
                    train_trajectories.append(str(traj_files[0]))

            # Extract problem paths for testing
            test_problems = []
            for prob_dir in test_problem_dirs:
                pddl_files = list(prob_dir.glob("*.pddl"))
                if pddl_files:
                    test_problems.append(str(pddl_files[0]))

            print(f"\n--- Domain {bench_name} | dir={dir_name} | fold {fold + 1}/{N_FOLDS} ---")
            print(f"Train problem dirs ({len(train_problem_dirs)}): {[d.name for d in train_problem_dirs]}")
            print(f"Train trajectories ({len(train_trajectories)}): {[Path(p).parent.name for p in train_trajectories]}")
            print(f"Test problem dirs  ({len(test_problem_dirs)}): {[d.name for d in test_problem_dirs]}")
            print(f"Test problems  ({len(test_problems)}): {[Path(p).name for p in test_problems]}")

            for num_trajs_used in traj_settings:
                # Select trajectories from the training set
                selected_trajs = train_trajectories[:num_trajs_used]

                print(f"\n>>> Fold {fold + 1}, num_trajs_used={num_trajs_used}")
                print(f"    Using {len(selected_trajs)} trajectories for learning")
                print(f"    Selected: {[Path(p).parent.name for p in selected_trajs]}")

                # # =========================
                # # PO-ROSAME
                # # =========================
                # print(f"\n{'=' * 80}")
                # print(f"PO-ROSAME - Domain: {bench_name}, Trajectories: {num_trajs_used}, Fold: {fold + 1}")
                # print(f"{'=' * 80}")
                #
                # po_rosame = PO_ROSAME()
                # rosame_model = po_rosame.learn(domain_ref_path, selected_trajs)
                #
                # porosame_domain_eval_path = f'POROSAME_{domain_name}_fold{fold}_trajs{num_trajs_used}.pddl'
                # with open(porosame_domain_eval_path, 'w') as f:
                #     f.write(rosame_model)
                #
                # rosame_precision = syntactic_precision(porosame_domain_eval_path, domain_ref_path)
                # rosame_recall = syntactic_recall(porosame_domain_eval_path, domain_ref_path)
                # rosame_problem_solving = problem_solving(
                #     porosame_domain_eval_path, domain_ref_path, test_problems, timeout=60
                # )
                #
                # rosame_result = {
                #     'domain': bench_name,
                #     'algorithm': 'ROSAME',
                #     'fold': fold,
                #     'num_trajs_used': num_trajs_used,
                #     'problems_count': len(test_problems),
                #     'precision_precs_pos': rosame_precision.get('precs_pos', None) if isinstance(rosame_precision, dict) else None,
                #     'precision_precs_neg': rosame_precision.get('precs_neg', None) if isinstance(rosame_precision, dict) else None,
                #     'precision_eff_pos': rosame_precision.get('eff_pos', None) if isinstance(rosame_precision, dict) else None,
                #     'precision_eff_neg': rosame_precision.get('eff_neg', None) if isinstance(rosame_precision, dict) else None,
                #     'precision_overall': rosame_precision.get('mean', None) if isinstance(rosame_precision, dict) else rosame_precision,
                #     'recall_precs_pos': rosame_recall.get('precs_pos', None) if isinstance(rosame_recall, dict) else None,
                #     'recall_precs_neg': rosame_recall.get('precs_neg', None) if isinstance(rosame_recall, dict) else None,
                #     'recall_eff_pos': rosame_recall.get('eff_pos', None) if isinstance(rosame_recall, dict) else None,
                #     'recall_eff_neg': rosame_recall.get('eff_neg', None) if isinstance(rosame_recall, dict) else None,
                #     'recall_overall': rosame_recall.get('mean', None) if isinstance(rosame_recall, dict) else rosame_recall,
                #     'solving_ratio': rosame_problem_solving.get('solving_ratio', None) if isinstance(rosame_problem_solving, dict) else None,
                #     'false_plans_ratio': rosame_problem_solving.get('false_plans_ratio', None) if isinstance(rosame_problem_solving, dict) else None,
                #     'unsolvable_ratio': rosame_problem_solving.get('unsolvable_ratio', None) if isinstance(rosame_problem_solving, dict) else None,
                #     'timed_out': rosame_problem_solving.get('timed_out', None) if isinstance(rosame_problem_solving, dict) else None,
                # }
                # all_results.append(rosame_result)
                #
                # # =========================
                # # PISAM
                # # =========================
                # print(f"\n{'=' * 80}")
                # print(f"PISAM - Domain: {bench_name}, Trajectories: {num_trajs_used}, Fold: {fold + 1}")
                # print(f"{'=' * 80}")
                #
                # pisam = PISAM()
                # pisam_model = pisam.learn(domain_ref_path, selected_trajs)
                #
                # pisam_domain_eval_path = f'PISAM_{domain_name}_fold{fold}_trajs{num_trajs_used}.pddl'
                # with open(pisam_domain_eval_path, 'w') as f:
                #     f.write(pisam_model)
                #
                # pisam_precision = syntactic_precision(pisam_domain_eval_path, domain_ref_path)
                # pisam_recall = syntactic_recall(pisam_domain_eval_path, domain_ref_path)
                # pisam_problem_solving = problem_solving(
                #     pisam_domain_eval_path, domain_ref_path, test_problems, timeout=60
                # )
                #
                # pisam_result = {
                #     'domain': bench_name,
                #     'algorithm': 'PISAM',
                #     'fold': fold,
                #     'num_trajs_used': num_trajs_used,
                #     'problems_count': len(test_problems),
                #     'precision_precs_pos': pisam_precision.get('precs_pos', None) if isinstance(pisam_precision, dict) else None,
                #     'precision_precs_neg': pisam_precision.get('precs_neg', None) if isinstance(pisam_precision, dict) else None,
                #     'precision_eff_pos': pisam_precision.get('eff_pos', None) if isinstance(pisam_precision, dict) else None,
                #     'precision_eff_neg': pisam_precision.get('eff_neg', None) if isinstance(pisam_precision, dict) else None,
                #     'precision_overall': pisam_precision.get('mean', None) if isinstance(pisam_precision, dict) else pisam_precision,
                #     'recall_precs_pos': pisam_recall.get('precs_pos', None) if isinstance(pisam_recall, dict) else None,
                #     'recall_precs_neg': pisam_recall.get('precs_neg', None) if isinstance(pisam_recall, dict) else None,
                #     'recall_eff_pos': pisam_recall.get('eff_pos', None) if isinstance(pisam_recall, dict) else None,
                #     'recall_eff_neg': pisam_recall.get('eff_neg', None) if isinstance(pisam_recall, dict) else None,
                #     'recall_overall': pisam_recall.get('mean', None) if isinstance(pisam_recall, dict) else pisam_recall,
                #     'solving_ratio': pisam_problem_solving.get('solving_ratio', None) if isinstance(pisam_problem_solving, dict) else None,
                #     'false_plans_ratio': pisam_problem_solving.get('false_plans_ratio', None) if isinstance(pisam_problem_solving, dict) else None,
                #     'unsolvable_ratio': pisam_problem_solving.get('unsolvable_ratio', None) if isinstance(pisam_problem_solving, dict) else None,
                #     'timed_out': pisam_problem_solving.get('timed_out', None) if isinstance(pisam_problem_solving, dict) else None,
                # }
                # all_results.append(pisam_result)

                # =========================
                # NOISY_PISAM
                # =========================
                print(f"\n{'=' * 80}")
                print(f"NOISY_PISAM - Domain: {bench_name}, Trajectories: {num_trajs_used}, Fold: {fold + 1}")
                print(f"{'=' * 80}")

                noisy_pisam = NOISY_PISAM()
                noisy_pisam_model = noisy_pisam.learn(domain_ref_path, selected_trajs)

                noisy_pisam_domain_eval_path = f'NOISY_PISAM_{domain_name}_fold{fold}_trajs{num_trajs_used}.pddl'
                with open(noisy_pisam_domain_eval_path, 'w') as f:
                    f.write(noisy_pisam_model)

                noisy_pisam_precision = syntactic_precision(noisy_pisam_domain_eval_path, domain_ref_path)
                noisy_pisam_recall = syntactic_recall(noisy_pisam_domain_eval_path, domain_ref_path)
                noisy_pisam_problem_solving = problem_solving(
                    noisy_pisam_domain_eval_path, domain_ref_path, test_problems, timeout=60
                )

                noisy_pisam_result = {
                    'domain': bench_name,
                    'algorithm': 'NOISY_PISAM',
                    'fold': fold,
                    'num_trajs_used': num_trajs_used,
                    'problems_count': len(test_problems),
                    'precision_precs_pos': noisy_pisam_precision.get('precs_pos', None) if isinstance(noisy_pisam_precision, dict) else None,
                    'precision_precs_neg': noisy_pisam_precision.get('precs_neg', None) if isinstance(noisy_pisam_precision, dict) else None,
                    'precision_eff_pos': noisy_pisam_precision.get('eff_pos', None) if isinstance(noisy_pisam_precision, dict) else None,
                    'precision_eff_neg': noisy_pisam_precision.get('eff_neg', None) if isinstance(noisy_pisam_precision, dict) else None,
                    'precision_overall': noisy_pisam_precision.get('mean', None) if isinstance(noisy_pisam_precision, dict) else noisy_pisam_precision,
                    'recall_precs_pos': noisy_pisam_recall.get('precs_pos', None) if isinstance(noisy_pisam_recall, dict) else None,
                    'recall_precs_neg': noisy_pisam_recall.get('reccs_neg', None) if isinstance(noisy_pisam_recall, dict) else None
                    if isinstance(noisy_pisam_recall, dict) and 'reccs_neg' in noisy_pisam_recall else
                    noisy_pisam_recall.get('recall_precs_neg', None) if isinstance(noisy_pisam_recall, dict) and 'recall_precs_neg' in noisy_pisam_recall else None,
                    'recall_eff_pos': noisy_pisam_recall.get('eff_pos', None) if isinstance(noisy_pisam_recall, dict) else None,
                    'recall_eff_neg': noisy_pisam_recall.get('eff_neg', None) if isinstance(noisy_pisam_recall, dict) else None,
                    'recall_overall': noisy_pisam_recall.get('mean', None) if isinstance(noisy_pisam_recall, dict) else noisy_pisam_recall,
                    'solving_ratio': noisy_pisam_problem_solving.get('solving_ratio', None) if isinstance(noisy_pisam_problem_solving, dict) else None,
                    'false_plans_ratio': noisy_pisam_problem_solving.get('false_plans_ratio', None) if isinstance(noisy_pisam_problem_solving, dict) else None,
                    'unsolvable_ratio': noisy_pisam_problem_solving.get('unsolvable_ratio', None) if isinstance(noisy_pisam_problem_solving, dict) else None,
                    'timed_out': noisy_pisam_problem_solving.get('timed_out', None) if isinstance(noisy_pisam_problem_solving, dict) else None,
                }
                all_results.append(noisy_pisam_result)

# =============================================================================
# AGGREGATE OVER FOLDS: MEAN & STD
# =============================================================================

print("\n" + "=" * 80)
print("AGGREGATING RESULTS OVER FOLDS (MEAN & STD)")
print("=" * 80)

df_all = pd.DataFrame(all_results)

group_cols = ["domain", "algorithm", "num_trajs_used"]
grouped = df_all.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()

# Flatten multiindex columns
flat_cols = []
for col in grouped.columns:
    if isinstance(col, tuple):
        base, stat = col
        if stat == "":
            flat_cols.append(base)
        else:
            flat_cols.append(f"{base}_{stat}")
    else:
        flat_cols.append(col)
grouped.columns = flat_cols

# Build df_avg with:
#   - <metric>  = mean
#   - <metric>_std = std
df_avg = grouped[group_cols].copy()
for m in metric_cols:
    mean_col = f"{m}_mean"
    std_col = f"{m}_std"
    df_avg[m] = grouped[mean_col]
    df_avg[f"{m}_std"] = grouped[std_col]

avg_results = df_avg.to_dict(orient="records")

# =============================================================================
# BUILD EXCEL REPORT (MEANS) - ONE SHEET PER NUM_TRAJS_USED
# =============================================================================
def clean_excel_value(v):
    if v is None:
        return ""
    if isinstance(v, float) and (pd.isna(v) or pd.isnull(v)):
        return ""
    if isinstance(v, float) and (v == float("inf") or v == float("-inf")):
        return ""
    return v

print("\n" + "=" * 80)
print("GENERATING EXCEL REPORT")
print("=" * 80)

timestamp = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
xlsx_filename = f"benchmark_results_{timestamp}.xlsx"
xlsx_path = benchmark_path / xlsx_filename

precision_metrics = [
    "precision_precs_pos",
    "precision_precs_neg",
    "precision_eff_pos",
    "precision_eff_neg",
    "precision_overall",
]
recall_metrics = [
    "recall_precs_pos",
    "recall_precs_neg",
    "recall_eff_pos",
    "recall_eff_neg",
    "recall_overall",
]
problem_metrics = [
    "problems_count",
    "solving_ratio",
    "false_plans_ratio",
    "unsolvable_ratio",
    "timed_out",
]

from collections import defaultdict
by_trajs: dict[str, list[dict]] = defaultdict(list)
for r in avg_results:
    by_trajs[str(int(r["num_trajs_used"]))].append(r)

with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
    workbook = writer.book
    thin_border = workbook.add_format({"border": 1})
    thick_left = workbook.add_format({"border": 1, "left": 2})
    thick_right = workbook.add_format({"border": 1, "right": 2})

    for num_trajs in sorted(by_trajs.keys(), key=lambda x: int(x)):
        results = by_trajs[num_trajs]
        sheet_name = f"trajs={num_trajs}"
        sheet = workbook.add_worksheet(sheet_name)
        writer.sheets[sheet_name] = sheet

        domains = sorted({r["domain"] for r in results})
        algorithms = sorted({r["algorithm"] for r in results})

        # map (domain, algorithm) -> result dict
        res_map: dict[tuple[str, str], dict] = {}
        for r in results:
            res_map[(r["domain"], r["algorithm"])] = r

        # -----------------------------
        # Helper: write syntactic P/R table (means)
        # -----------------------------
        def write_syn_table(start_row: int) -> tuple[int, int, int]:
            """
            writes syntactic P/R table starting at start_row
            returns (first_row, last_row, last_col)
            """
            row0 = start_row      # type row (Precision / Recall)
            row1 = start_row + 1  # metric row
            row2 = start_row + 2  # algorithm row

            # first column is for Domain
            sheet.write(row0, 0, "", thin_border)
            sheet.write(row1, 0, "", thin_border)
            sheet.write(row2, 0, "Domain", thin_border)

            col = 1
            type_spans: dict[str, tuple[int, int]] = {}
            metric_spans: dict[tuple[str, str], tuple[int, int]] = {}

            # reserve columns and record spans
            for t, metrics in [("Precision", precision_metrics),
                               ("Recall", recall_metrics)]:
                type_start = col
                for m in metrics:
                    metric_start = col
                    for _alg in algorithms:
                        col += 1
                    metric_end = col - 1
                    metric_spans[(t, m)] = (metric_start, metric_end)
                type_end = col - 1
                type_spans[t] = (type_start, type_end)

            # write merged type headers
            for t, (c_start, c_end) in type_spans.items():
                sheet.merge_range(row0, c_start, row0, c_end, t, thin_border)

            # write merged metric headers
            for (t, m), (c_start, c_end) in metric_spans.items():
                sheet.merge_range(row1, c_start, row1, c_end, m, thin_border)

            # write algorithm names
            col_ptr = 1
            for t, metrics in [("Precision", precision_metrics),
                               ("Recall", recall_metrics)]:
                for m in metrics:
                    for alg in algorithms:
                        sheet.write(row2, col_ptr, alg, thin_border)
                        col_ptr += 1

            # data rows
            for i, dom in enumerate(domains):
                r_idx = row2 + 1 + i
                sheet.write(r_idx, 0, dom, thin_border)
                c = 1
                for t, metrics in [("Precision", precision_metrics),
                                   ("Recall", recall_metrics)]:
                    for m in metrics:
                        for alg in algorithms:
                            val = res_map.get((dom, alg), {}).get(m, "")
                            sheet.write(r_idx, c, clean_excel_value(val), thin_border)
                            c += 1

            first_row = row0
            last_row = row2 + len(domains)
            last_col = col - 1

            # thicker vertical borders between metric groups
            for (_t, _m), (start_c, end_c) in metric_spans.items():
                sheet.conditional_format(
                    first_row, start_c, last_row, start_c,
                    {"type": "formula", "criteria": "TRUE", "format": thick_left},
                )
                sheet.conditional_format(
                    first_row, end_c, last_row, end_c,
                    {"type": "formula", "criteria": "TRUE", "format": thick_right},
                )

            return first_row, last_row, last_col

        # -----------------------------
        # Helper: write problem-solving table (means)
        # -----------------------------
        def write_prob_table(start_row: int) -> tuple[int, int, int]:
            """
            writes problem-solving table starting at start_row
            returns (first_row, last_row, last_col)
            """
            row0 = start_row      # group row
            row1 = start_row + 1  # metric row
            row2 = start_row + 2  # algorithm row

            sheet.write(row0, 0, "", thin_border)
            sheet.write(row1, 0, "", thin_border)
            sheet.write(row2, 0, "Domain", thin_border)

            col = 1
            metric_spans: dict[str, tuple[int, int]] = {}

            group_name = "ProblemSolving"
            group_start = col
            for m in problem_metrics:
                metric_start = col
                for _alg in algorithms:
                    col += 1
                metric_end = col - 1
                metric_spans[m] = (metric_start, metric_end)
            group_end = col - 1

            sheet.merge_range(row0, group_start, row0, group_end, group_name, thin_border)

            # merged metric headers
            for m, (c_start, c_end) in metric_spans.items():
                sheet.merge_range(row1, c_start, row1, c_end, m, thin_border)

            # algorithm names
            col_ptr = 1
            for m in problem_metrics:
                for alg in algorithms:
                    sheet.write(row2, col_ptr, alg, thin_border)
                    col_ptr += 1

            # data rows
            for i, dom in enumerate(domains):
                r_idx = row2 + 1 + i
                sheet.write(r_idx, 0, dom, thin_border)
                c = 1
                for m in problem_metrics:
                    for alg in algorithms:
                        val = res_map.get((dom, alg), {}).get(m, "")
                        sheet.write(r_idx, c, clean_excel_value(val), thin_border)
                        c += 1

            first_row = row0
            last_row = row2 + len(domains)
            last_col = col - 1

            # thick borders between metrics
            for _m, (start_c, end_c) in metric_spans.items():
                sheet.conditional_format(
                    first_row, start_c, last_row, start_c,
                    {"type": "formula", "criteria": "TRUE", "format": thick_left},
                )
                sheet.conditional_format(
                    first_row, end_c, last_row, end_c,
                    {"type": "formula", "criteria": "TRUE", "format": thick_right},
                )

            return first_row, last_row, last_col

        # 1) syntactic P/R table at top
        syn_first, syn_last, syn_last_col = write_syn_table(start_row=0)

        # 2) problem-solving table some rows below
        gap = 5
        prob_start = syn_last + 1 + gap
        prob_first, prob_last, prob_last_col = write_prob_table(start_row=prob_start)

print(f"\n✓ Excel report saved to: {xlsx_path}")
print(f"  Total raw runs (all folds): {len(all_results)}")
print("  Sheets (num_trajs_used):", ", ".join(sorted(by_trajs.keys(), key=lambda x: int(x))))
print("\n" + "=" * 80)
print("ALL EXPERIMENTS COMPLETED")
print("\n" + "=" * 80)

# =============================================================================
# PLOTTING METRIC TRENDS VS #TRAJECTORIES (WITH STD ERROR BARS)
# =============================================================================

plots_output_dir = benchmark_path / "plots"
plots_output_dir.mkdir(exist_ok=True)


def plot_metric_trends(results, metric_key, metric_title, save_dir):
    """
    Creates a line plot of a given metric as a function of num_trajs_used,
    with one line per algorithm, and error bars (std over folds).
    """
    df = pd.DataFrame(results)
    df["num_trajs_used"] = df["num_trajs_used"].astype(int)

    algorithms = sorted(df["algorithm"].unique())

    plt.figure(figsize=(8, 5))

    for algo in algorithms:
        sub = df[df["algorithm"] == algo].sort_values("num_trajs_used")
        y = sub[metric_key]
        x = sub["num_trajs_used"]
        yerr_col = f"{metric_key}_std"
        if yerr_col in sub.columns:
            yerr = sub[yerr_col]
        else:
            yerr = None

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=4,
            label=algo,
        )

    plt.title(f"{metric_title} vs #training trajectories")
    plt.xlabel("#training trajectories")
    plt.ylabel(metric_title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    save_path = Path(save_dir) / f"{metric_key}_vs_num_trajs.png"
    plt.savefig(save_path)
    print(f"✓ Saved plot: {save_path}")
    plt.close()


plot_metric_trends(df_avg, "solving_ratio", "Solving Ratio", plots_output_dir)
plot_metric_trends(df_avg, "false_plans_ratio", "False Plan Ratio", plots_output_dir)
plot_metric_trends(df_avg, "unsolvable_ratio", "Unsolvable Ratio", plots_output_dir)

print("✓ All metric plots generated (means with std error bars).")