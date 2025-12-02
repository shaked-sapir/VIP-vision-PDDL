import csv
from pathlib import Path
from datetime import datetime

import pandas as pd
from amlgym.benchmarks import *
from amlgym.algorithms import *
from amlgym.metrics import *

from benchmark.amlgym_models.NOISY_PISAM import NOISY_PISAM
from benchmark.amlgym_models.PISAM import PISAM
from benchmark.amlgym_models.PO_ROSAME import PO_ROSAME

benchmark_path = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark")

print_metrics()

experiment_data_dirs = {
    "blocksworld": [
        # "experiment_30-11-2025T12:47:58__steps=10",
        # "experiment_30-11-2025T13:03:16__steps=25",
        "experiment_30-11-2025T13:03:31__steps=50"
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
    ]
}

# Collect all experiment results for CSV report
all_results = []

domain_name_mappings = {
    # 'hiking': 'hiking',
    'hanoi': 'hanoi',
    'blocksworld': 'blocksworld',
    'n_puzzle_typed': 'npuzzle',
}

not_in_amlgym_domains = {
    "hanoi": {
        "trajectory_training_problem": "problem0", # for documenting, not to put in the problems to be solved
        "domain_path": benchmark_path / 'domains' / 'hanoi' / 'hanoi.pddl',
        "problems_paths": sorted(str(p) for p in Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/domains/hanoi/problems").glob("problem*.pddl") if "problem0" not in str(p))
    },
    "hiking": {
        "trajectory_training_problem": "problem2",
        "domain_path": benchmark_path / 'domains' / 'hiking' / 'hiking.pddl',
        "problems_paths": sorted(str(p) for p in Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/domains/hiking/problems").glob("problem*.pddl") if "problem2" not in str(p))
    }
}

for domain_name, bench_name in domain_name_mappings.items():
    # Get domain reference and problem paths for this domain
    domain_ref_path = get_domain_path(bench_name) if domain_name not in not_in_amlgym_domains else \
        not_in_amlgym_domains[domain_name]["domain_path"]
    probs_paths = get_problems_path(bench_name) if domain_name not in not_in_amlgym_domains else \
        not_in_amlgym_domains[domain_name]["problems_paths"]

    for dir in experiment_data_dirs[domain_name]:

        num_steps = dir.split('steps=')[1]
        """=========================
                PO-ROSAME TESTING
        =========================="""
        print(f"\n{'=' * 80}")
        print(f"PO-ROSAME - Domain: {bench_name}, Steps: {num_steps}")
        print(f"{'=' * 80}")

        po_rosame = PO_ROSAME()

        rosame_traj_paths = Path(benchmark_path / 'data' / domain_name / dir / 'training' / 'rosame_trace')
        rosame_trajectory_paths = sorted(str(p) for p in rosame_traj_paths.glob("*/*.trajectory"))

        rosame_model = po_rosame.learn(domain_ref_path, rosame_trajectory_paths)
        print(rosame_model)

        porosame_domain_eval_path = f'POROSAME_{domain_name}_domain_learned.pddl'
        with open(porosame_domain_eval_path, 'w') as f:
            f.write(rosame_model)

        rosame_precision = syntactic_precision(porosame_domain_eval_path, domain_ref_path)
        print("Precision:", rosame_precision)

        rosame_recall = syntactic_recall(porosame_domain_eval_path, domain_ref_path)
        print("Recall:", rosame_recall)

        print(f"problem_paths: {probs_paths}")
        rosame_problem_solving = problem_solving(porosame_domain_eval_path, domain_ref_path, probs_paths, timeout=60)
        print("Problem Solving:", rosame_problem_solving)

        # Collect PO-ROSAME results
        rosame_result = {
            'domain': bench_name,
            'algorithm': 'ROSAME',
            'num_steps': num_steps,
            'precision_precs_pos': rosame_precision.get('precs_pos', None) if isinstance(rosame_precision,
                                                                                         dict) else None,
            'precision_precs_neg': rosame_precision.get('precs_neg', None) if isinstance(rosame_precision,
                                                                                         dict) else None,
            'precision_eff_pos': rosame_precision.get('eff_pos', None) if isinstance(rosame_precision, dict) else None,
            'precision_eff_neg': rosame_precision.get('eff_neg', None) if isinstance(rosame_precision, dict) else None,
            'precision_overall': rosame_precision.get('mean', None) if isinstance(rosame_precision,
                                                                                  dict) else rosame_precision,
            'recall_precs_pos': rosame_recall.get('precs_pos', None) if isinstance(rosame_recall, dict) else None,
            'recall_precs_neg': rosame_recall.get('precs_neg', None) if isinstance(rosame_recall, dict) else None,
            'recall_eff_pos': rosame_recall.get('eff_pos', None) if isinstance(rosame_recall, dict) else None,
            'recall_eff_neg': rosame_recall.get('eff_neg', None) if isinstance(rosame_recall, dict) else None,
            'recall_overall': rosame_recall.get('mean', None) if isinstance(rosame_recall, dict) else rosame_recall,
            'problems_count': len(probs_paths),
            'solving_ratio': rosame_problem_solving.get('solving_ratio', None) if isinstance(rosame_problem_solving,
                                                                                             dict) else None,
            'false_plans_ratio': rosame_problem_solving.get('false_plans_ratio', None) if isinstance(
                rosame_problem_solving, dict) else None,
            'unsolvable_ratio': rosame_problem_solving.get('unsolvable_ratio', None) if isinstance(
                rosame_problem_solving, dict) else None,
            'timed_out': rosame_problem_solving.get('timed_out', None) if isinstance(rosame_problem_solving,
                                                                                     dict) else None,
        }
        all_results.append(rosame_result)

        """=========================
               PISAM TESTING
               =========================="""
        print(f"\n{'=' * 80}")
        print(f"PISAM - Domain: {bench_name}, Steps: {num_steps}")
        print(f"{'=' * 80}")

        pisam = PISAM()

        pisam_traces_dir_path = Path(benchmark_path / 'data' / domain_name / dir / 'training' / 'pi_sam_traces')
        pisam_trajectory_paths = sorted(str(p) for p in pisam_traces_dir_path.glob("trace_*/*.trajectory"))

        pisam_model = pisam.learn(domain_ref_path, pisam_trajectory_paths)
        print(pisam_model)

        pisam_domain_eval_path = f'PISAM_{domain_name}_domain_learned.pddl'
        with open(pisam_domain_eval_path, 'w') as f:
            f.write(pisam_model)

        pisam_precision = syntactic_precision(pisam_domain_eval_path, domain_ref_path)
        print("Precision:", pisam_precision)

        pisam_recall = syntactic_recall(pisam_domain_eval_path, domain_ref_path)
        print("Recall:", pisam_recall)

        print(f"problem_paths: {probs_paths}")
        pisam_problem_solving = problem_solving(pisam_domain_eval_path, domain_ref_path, probs_paths, timeout=60)
        print("Problem Solving:", pisam_problem_solving)

        # Collect PISAM results
        pisam_result = {
            'domain': bench_name,
            'algorithm': 'PISAM',
            'num_steps': num_steps,
            'precision_precs_pos': pisam_precision.get('precs_pos', None) if isinstance(pisam_precision,
                                                                                        dict) else None,
            'precision_precs_neg': pisam_precision.get('precs_neg', None) if isinstance(pisam_precision,
                                                                                        dict) else None,
            'precision_eff_pos': pisam_precision.get('eff_pos', None) if isinstance(pisam_precision, dict) else None,
            'precision_eff_neg': pisam_precision.get('eff_neg', None) if isinstance(pisam_precision, dict) else None,
            'precision_overall': pisam_precision.get('mean', None) if isinstance(pisam_precision,
                                                                                 dict) else pisam_precision,
            'recall_precs_pos': pisam_recall.get('precs_pos', None) if isinstance(pisam_recall, dict) else None,
            'recall_precs_neg': pisam_recall.get('precs_neg', None) if isinstance(pisam_recall, dict) else None,
            'recall_eff_pos': pisam_recall.get('eff_pos', None) if isinstance(pisam_recall, dict) else None,
            'recall_eff_neg': pisam_recall.get('eff_neg', None) if isinstance(pisam_recall, dict) else None,
            'recall_overall': pisam_recall.get('mean', None) if isinstance(pisam_recall, dict) else pisam_recall,
            'problems_count': len(probs_paths),
            'solving_ratio': pisam_problem_solving.get('solving_ratio', None) if isinstance(pisam_problem_solving,
                                                                                            dict) else None,
            'false_plans_ratio': pisam_problem_solving.get('false_plans_ratio', None) if isinstance(
                pisam_problem_solving, dict) else None,
            'unsolvable_ratio': pisam_problem_solving.get('unsolvable_ratio', None) if isinstance(pisam_problem_solving,
                                                                                                  dict) else None,
            'timed_out': pisam_problem_solving.get('timed_out', None) if isinstance(pisam_problem_solving,
                                                                                    dict) else None,
        }
        all_results.append(pisam_result)

        """=========================
        NOISY_PISAM TESTING
        =========================="""
        print(f"\n{'=' * 80}")
        print(f"NOISY_PISAM - Domain: {bench_name}, Steps: {num_steps}")
        print(f"{'=' * 80}")

        noisy_pisam = NOISY_PISAM()

        noisy_pisam_traces_dir_path = Path(benchmark_path / 'data' / domain_name / dir / 'training' / 'pi_sam_traces')
        noisy_pisam_trajectory_paths = sorted(str(p) for p in noisy_pisam_traces_dir_path.glob("trace_*/*.trajectory"))

        noisy_pisam_model = noisy_pisam.learn(domain_ref_path, noisy_pisam_trajectory_paths)
        print(noisy_pisam_model)

        noisy_pisam_domain_eval_path = f'NOISY_PISAM_{domain_name}_domain_learned.pddl'
        with open(noisy_pisam_domain_eval_path, 'w') as f:
            f.write(noisy_pisam_model)

        noisy_pisam_precision = syntactic_precision(noisy_pisam_domain_eval_path, domain_ref_path)
        print("Precision:", noisy_pisam_precision)

        noisy_pisam_recall = syntactic_recall(noisy_pisam_domain_eval_path, domain_ref_path)
        print("Recall:", noisy_pisam_recall)

        print(f"problem_paths: {probs_paths}")
        noisy_pisam_problem_solving = problem_solving(noisy_pisam_domain_eval_path, domain_ref_path, probs_paths, timeout=60)
        print("Problem Solving:", noisy_pisam_problem_solving)

        # Collect NOISY_PISAM results
        noisy_pisam_result = {
            'domain': bench_name,
            'algorithm': 'NOISY_PISAM',
            'num_steps': num_steps,
            'precision_precs_pos': noisy_pisam_precision.get('precs_pos', None) if isinstance(noisy_pisam_precision,
                                                                                        dict) else None,
            'precision_precs_neg': noisy_pisam_precision.get('precs_neg', None) if isinstance(noisy_pisam_precision,
                                                                                        dict) else None,
            'precision_eff_pos': noisy_pisam_precision.get('eff_pos', None) if isinstance(noisy_pisam_precision, dict) else None,
            'precision_eff_neg': noisy_pisam_precision.get('eff_neg', None) if isinstance(noisy_pisam_precision, dict) else None,
            'precision_overall': noisy_pisam_precision.get('mean', None) if isinstance(noisy_pisam_precision,
                                                                                    dict) else noisy_pisam_precision,
            'recall_precs_pos': noisy_pisam_recall.get('precs_pos', None) if isinstance(noisy_pisam_recall, dict) else None,
            'recall_precs_neg': noisy_pisam_recall.get('precs_neg', None) if isinstance(noisy_pisam_recall, dict) else None,
            'recall_eff_pos': noisy_pisam_recall.get('eff_pos', None) if isinstance(noisy_pisam_recall, dict) else None,
            'recall_eff_neg': noisy_pisam_recall.get('eff_neg', None) if isinstance(noisy_pisam_recall, dict) else None,
            'recall_overall': noisy_pisam_recall.get('mean', None) if isinstance(noisy_pisam_recall, dict) else noisy_pisam_recall,
            'problems_count': len(probs_paths),
            'solving_ratio': noisy_pisam_problem_solving.get('solving_ratio', None) if isinstance(noisy_pisam_problem_solving,
                                                                                            dict) else None,
            'false_plans_ratio': noisy_pisam_problem_solving.get('false_plans_ratio', None) if isinstance(
                noisy_pisam_problem_solving, dict) else None,
            'unsolvable_ratio': noisy_pisam_problem_solving.get('unsolvable_ratio', None) if isinstance(noisy_pisam_problem_solving,
                                                                                                  dict) else None,
            'timed_out': noisy_pisam_problem_solving.get('timed_out', None) if isinstance(noisy_pisam_problem_solving,
                                                                                    dict) else None,
        }
        all_results.append(noisy_pisam_result)


# ------------------------------------------------------------------------------------
# Build Excel report: one sheet per num_steps, with 2 tables per sheet
# ------------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("GENERATING EXCEL REPORT")
print("=" * 80)

timestamp = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
xlsx_filename = f"benchmark_results_{timestamp}.xlsx"
xlsx_path = benchmark_path / xlsx_filename

# Metrics to organize
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

# Group results by num_steps
from collections import defaultdict
by_steps: dict[str, list[dict]] = defaultdict(list)
for r in all_results:
    by_steps[str(r["num_steps"])].append(r)


with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
    workbook = writer.book
    thin_border = workbook.add_format({"border": 1})
    thick_left = workbook.add_format({"border": 1, "left": 2})
    thick_right = workbook.add_format({"border": 1, "right": 2})

    for num_steps in sorted(by_steps.keys(), key=lambda x: int(x)):
        results = by_steps[num_steps]
        sheet_name = f"steps={num_steps}"
        sheet = workbook.add_worksheet(sheet_name)
        writer.sheets[sheet_name] = sheet

        domains = sorted({r["domain"] for r in results})
        algorithms = sorted({r["algorithm"] for r in results})

        # map (domain, algorithm) -> result dict
        res_map: dict[tuple[str, str], dict] = {}
        for r in results:
            res_map[(r["domain"], r["algorithm"])] = r

        # -----------------------------
        # Helper: write syntactic P/R table
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

            # write algorithm names (unmerged)
            col_ptr = 1
            for t, metrics in [("Precision", precision_metrics),
                               ("Recall", recall_metrics)]:
                for m in metrics:
                    for alg in algorithms:
                        sheet.write(row2, col_ptr, alg, thin_border)
                        col_ptr += 1

            # write data rows (no blank row)
            for i, dom in enumerate(domains):
                r = row2 + 1 + i
                sheet.write(r, 0, dom, thin_border)
                c = 1
                for t, metrics in [("Precision", precision_metrics),
                                   ("Recall", recall_metrics)]:
                    for m in metrics:
                        for alg in algorithms:
                            val = res_map.get((dom, alg), {}).get(m, "")
                            sheet.write(r, c, val, thin_border)
                            c += 1

            first_row = row0
            last_row = row2 + len(domains)
            last_col = col - 1  # last used column

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
        # Helper: write problem-solving table
        # -----------------------------
        def write_prob_table(start_row: int) -> tuple[int, int, int]:
            """
            writes problem-solving table starting at start_row
            returns (first_row, last_row, last_col)
            """
            row0 = start_row      # group row ("ProblemSolving")
            row1 = start_row + 1  # metric row
            row2 = start_row + 2  # algorithm row

            sheet.write(row0, 0, "", thin_border)
            sheet.write(row1, 0, "", thin_border)
            sheet.write(row2, 0, "Domain", thin_border)

            col = 1
            metric_spans: dict[str, tuple[int, int]] = {}

            # reserve columns and record spans
            group_name = "ProblemSolving"
            group_start = col
            for m in problem_metrics:
                metric_start = col
                for _alg in algorithms:
                    col += 1
                metric_end = col - 1
                metric_spans[m] = (metric_start, metric_end)
            group_end = col - 1

            # merged group header across all metrics
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
                r = row2 + 1 + i
                sheet.write(r, 0, dom, thin_border)
                c = 1
                for m in problem_metrics:
                    for alg in algorithms:
                        val = res_map.get((dom, alg), {}).get(m, "")
                        sheet.write(r, c, val, thin_border)
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

        # 2) problem-solving table a few rows below
        gap = 5
        prob_start = syn_last + 1 + gap
        prob_first, prob_last, prob_last_col = write_prob_table(start_row=prob_start)

print(f"\n✓ Excel report saved to: {xlsx_path}")
print(f"  Total experiments: {len(all_results)}")
print("  Sheets (num_steps):", ", ".join(sorted(by_steps.keys(), key=lambda x: int(x))))
print("\n" + "=" * 80)
print("ALL EXPERIMENTS COMPLETED")
print("\n" + "=" * 80)
