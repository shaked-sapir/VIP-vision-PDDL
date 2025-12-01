import csv
from pathlib import Path
from datetime import datetime

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
        "experiment_30-11-2025T12:47:58__steps=10",
        "experiment_30-11-2025T13:03:16__steps=25",
        "experiment_30-11-2025T13:03:31__steps=50"
    ],
    "n_puzzle_typed": [
        "experiment_30-11-2025T13:17:05__steps=10",
        "experiment_30-11-2025T13:28:43__steps=25",
        "experiment_30-11-2025T13:28:47__steps=50"
    ],
    "hanoi": [
        "experiment_30-11-2025T19:11:13__steps=10",
    ]
}

# Collect all experiment results for CSV report
all_results = []

amlgym_domain_name_mappings = {
    'hanoi': 'hanoi',
    'blocksworld': 'blocksworld',
    'n_puzzle_typed': 'npuzzle',
}

not_in_amlgym_domains = {
    "hanoi": {
        "domain_path": benchmark_path / 'domains' / 'hanoi' / 'hanoi.pddl',
        "problems_paths": sorted(str(p) for p in Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/domains/hanoi/problems").glob("problem*.pddl") if "problem0" not in str(p))
    }
}

for amlgym_domain_name, amlgym_bench_name in amlgym_domain_name_mappings.items():
    # Get domain reference and problem paths for this domain
    domain_ref_path = get_domain_path(amlgym_bench_name) if amlgym_domain_name not in not_in_amlgym_domains else \
        not_in_amlgym_domains[amlgym_domain_name]["domain_path"]
    probs_paths = get_problems_path(amlgym_bench_name) if amlgym_domain_name not in not_in_amlgym_domains else \
        not_in_amlgym_domains[amlgym_domain_name]["problems_paths"]

    for dir in experiment_data_dirs[amlgym_domain_name]:

        num_steps = dir.split('steps=')[1]

        """=========================
        NOISY_PISAM TESTING
        =========================="""
        print(f"\n{'=' * 80}")
        print(f"PISAM - Domain: {amlgym_bench_name}, Steps: {num_steps}")
        print(f"{'=' * 80}")

        noisy_pisam = NOISY_PISAM()

        noisy_pisam_traces_dir_path = Path(benchmark_path / 'data' / amlgym_domain_name / dir / 'training' / 'pi_sam_traces')
        noisy_pisam_trajectory_paths = sorted(str(p) for p in noisy_pisam_traces_dir_path.glob("trace_*/*.trajectory"))

        noisy_pisam_model = noisy_pisam.learn(domain_ref_path, noisy_pisam_trajectory_paths)
        print(noisy_pisam_model)

        noisy_pisam_domain_eval_path = f'NOISY_PISAM_{amlgym_domain_name}_domain_learned.pddl'
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
            'domain': amlgym_bench_name,
            'algorithm': 'NOISY_PISAM',
            'num_steps': num_steps,
            'precision_precs_pos': noisy_pisam_precision.get('precs_pos', None) if isinstance(noisy_pisam_precision,
                                                                                        dict) else None,
            'precision_precs_neg': noisy_pisam_precision.get('precs_neg', None) if isinstance(noisy_pisam_precision,
                                                                                        dict) else None,
            'precision_eff_pos': noisy_pisam_precision.get('eff_pos', None) if isinstance(noisy_pisam_precision, dict) else None,
            'precision_eff_neg': noisy_pisam_precision.get('eff_neg', None) if isinstance(noisy_pisam_precision, dict) else None,
            'precision_overall': noisy_pisam_precision.get('overall', None) if isinstance(noisy_pisam_precision,
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

        """=========================
        PISAM TESTING
        =========================="""
        # print(f"\n{'='*80}")
        # print(f"PISAM - Domain: {amlgym_bench_name}, Steps: {num_steps}")
        # print(f"{'='*80}")
        #
        # pisam = PISAM()
        #
        # pisam_traces_dir_path = Path(benchmark_path / 'data' / amlgym_domain_name / dir / 'training' / 'pi_sam_traces')
        # pisam_trajectory_paths = sorted(str(p) for p in pisam_traces_dir_path.glob("trace_*/*.trajectory"))
        #
        # pisam_model = pisam.learn(domain_ref_path, pisam_trajectory_paths)
        # print(pisam_model)
        #
        # pisam_domain_eval_path = f'PISAM_{amlgym_domain_name}_domain_learned.pddl'
        # with open(pisam_domain_eval_path, 'w') as f:
        #     f.write(pisam_model)
        #
        # pisam_precision = syntactic_precision(pisam_domain_eval_path, domain_ref_path)
        # print("Precision:", pisam_precision)
        #
        # pisam_recall = syntactic_recall(pisam_domain_eval_path, domain_ref_path)
        # print("Recall:", pisam_recall)
        #
        # print(f"problem_paths: {probs_paths}")
        # pisam_problem_solving = problem_solving(pisam_domain_eval_path, domain_ref_path, probs_paths, timeout=60)
        # print("Problem Solving:", pisam_problem_solving)
        #
        # # Collect PISAM results
        # pisam_result = {
        #     'domain': amlgym_bench_name,
        #     'algorithm': 'PISAM',
        #     'num_steps': num_steps,
        #     'precision_precs_pos': pisam_precision.get('precs_pos', None) if isinstance(pisam_precision, dict) else None,
        #     'precision_precs_neg': pisam_precision.get('precs_neg', None) if isinstance(pisam_precision, dict) else None,
        #     'precision_eff_pos': pisam_precision.get('eff_pos', None) if isinstance(pisam_precision, dict) else None,
        #     'precision_eff_neg': pisam_precision.get('eff_neg', None) if isinstance(pisam_precision, dict) else None,
        #     'precision_overall': pisam_precision.get('overall', None) if isinstance(pisam_precision, dict) else pisam_precision,
        #     'recall_precs_pos': pisam_recall.get('precs_pos', None) if isinstance(pisam_recall, dict) else None,
        #     'recall_precs_neg': pisam_recall.get('precs_neg', None) if isinstance(pisam_recall, dict) else None,
        #     'recall_eff_pos': pisam_recall.get('eff_pos', None) if isinstance(pisam_recall, dict) else None,
        #     'recall_eff_neg': pisam_recall.get('eff_neg', None) if isinstance(pisam_recall, dict) else None,
        #     'recall_overall': pisam_recall.get('mean', None) if isinstance(pisam_recall, dict) else pisam_recall,
        #     'problems_count': len(probs_paths),
        #     'solving_ratio': pisam_problem_solving.get('solving_ratio', None) if isinstance(pisam_problem_solving, dict) else None,
        #     'false_plans_ratio': pisam_problem_solving.get('false_plans_ratio', None) if isinstance(pisam_problem_solving, dict) else None,
        #     'unsolvable_ratio': pisam_problem_solving.get('unsolvable_ratio', None) if isinstance(pisam_problem_solving, dict) else None,
        #     'timed_out': pisam_problem_solving.get('timed_out', None) if isinstance(pisam_problem_solving, dict) else None,
        # }
        # all_results.append(pisam_result)
        #
        #
        # """=========================
        # PO-ROSAME TESTING
        # =========================="""
        # print(f"\n{'='*80}")
        # print(f"PO-ROSAME - Domain: {amlgym_bench_name}, Steps: {num_steps}")
        # print(f"{'='*80}")
        #
        # po_rosame = PO_ROSAME()
        #
        # rosame_traj_paths = Path(benchmark_path / 'data' / amlgym_domain_name / dir / 'training' / 'rosame_trace')
        # rosame_trajectory_paths = sorted(str(p) for p in pisam_traces_dir_path.glob("trace_*/*.trajectory"))
        #
        # rosame_model = po_rosame.learn(domain_ref_path, rosame_trajectory_paths)
        # print(rosame_model)
        #
        # porosame_domain_eval_path = f'POROSAME_{amlgym_domain_name}_domain_learned.pddl'
        # with open(porosame_domain_eval_path, 'w') as f:
        #     f.write(rosame_model)
        #
        # rosame_precision = syntactic_precision(porosame_domain_eval_path, domain_ref_path)
        # print("Precision:", rosame_precision)
        #
        # rosame_recall = syntactic_recall(porosame_domain_eval_path, domain_ref_path)
        # print("Recall:", rosame_recall)
        #
        # print(f"problem_paths: {probs_paths}")
        # rosame_problem_solving = problem_solving(porosame_domain_eval_path, domain_ref_path, probs_paths, timeout=60)
        # print("Problem Solving:", rosame_problem_solving)
        #
        # # Collect PO-ROSAME results
        # rosame_result = {
        #     'domain': amlgym_bench_name,
        #     'algorithm': 'ROSAME',
        #     'num_steps': num_steps,
        #     'precision_precs_pos': rosame_precision.get('precs_pos', None) if isinstance(rosame_precision, dict) else None,
        #     'precision_precs_neg': rosame_precision.get('precs_neg', None) if isinstance(rosame_precision, dict) else None,
        #     'precision_eff_pos': rosame_precision.get('eff_pos', None) if isinstance(rosame_precision, dict) else None,
        #     'precision_eff_neg': rosame_precision.get('eff_neg', None) if isinstance(rosame_precision, dict) else None,
        #     'precision_overall': rosame_precision.get('overall', None) if isinstance(rosame_precision, dict) else rosame_precision,
        #     'recall_precs_pos': rosame_recall.get('precs_pos', None) if isinstance(rosame_recall, dict) else None,
        #     'recall_precs_neg': rosame_recall.get('precs_neg', None) if isinstance(rosame_recall, dict) else None,
        #     'recall_eff_pos': rosame_recall.get('eff_pos', None) if isinstance(rosame_recall, dict) else None,
        #     'recall_eff_neg': rosame_recall.get('eff_neg', None) if isinstance(rosame_recall, dict) else None,
        #     'recall_overall': rosame_recall.get('mean', None) if isinstance(rosame_recall, dict) else rosame_recall,
        #     'problems_count': len(probs_paths),
        #     'solving_ratio': rosame_problem_solving.get('solving_ratio', None) if isinstance(rosame_problem_solving, dict) else None,
        #     'false_plans_ratio': rosame_problem_solving.get('false_plans_ratio', None) if isinstance(rosame_problem_solving, dict) else None,
        #     'unsolvable_ratio': rosame_problem_solving.get('unsolvable_ratio', None) if isinstance(rosame_problem_solving, dict) else None,
        #     'timed_out': rosame_problem_solving.get('timed_out', None) if isinstance(rosame_problem_solving, dict) else None,
        # }
        # all_results.append(rosame_result)

# Write all results to CSV report
print("\n" + "="*80)
print("GENERATING CSV REPORT")
print("="*80)

timestamp = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
csv_filename = f"benchmark_results_{timestamp}.csv"
csv_path = benchmark_path / csv_filename

# Define CSV columns
csv_columns = [
    'domain',
    'algorithm',
    'num_steps',
    'precision_precs_pos',
    'precision_precs_neg',
    'precision_eff_pos',
    'precision_eff_neg',
    'precision_overall',
    'recall_precs_pos',
    'recall_precs_neg',
    'recall_eff_pos',
    'recall_eff_neg',
    'recall_overall',
    'problems_count',
    'solving_ratio',
    'false_plans_ratio',
    'unsolvable_ratio',
    'timed_out'
]

# Write to CSV
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)

    # Write header
    writer.writeheader()

    # Write all experiment results
    for result in all_results:
        writer.writerow(result)

print(f"\n✓ CSV report saved to: {csv_path}")
print(f"  Total experiments: {len(all_results)}")
print(f"  Columns: {len(csv_columns)}")
print("\nReport Summary:")
print(f"  Domains tested: {len(set(r['domain'] for r in all_results))}")
print(f"  Algorithms tested: {len(set(r['algorithm'] for r in all_results))}")
print(f"  Configurations: {len(all_results)}")
print("\n" + "="*80)
print("ALL EXPERIMENTS COMPLETE")
print("="*80)