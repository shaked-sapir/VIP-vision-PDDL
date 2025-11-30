from pathlib import Path

from amlgym.benchmarks import *
from amlgym.algorithms import *
from amlgym.metrics import *

from benchmark.amlgym_models.PISAM import PISAM
from benchmark.amlgym_models.PO_ROSAME import PO_ROSAME

from pathlib import Path

benchmark_path = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark")
amlgym_domain_name = 'npuzzle'
domain_ref_path = get_domain_path(amlgym_domain_name)
probs_paths = get_problems_path(amlgym_domain_name)

print_metrics()

# rosame = get_algorithm('ROSAME')
# domain_ref_path = get_domain_path('blocksworld')
# traj_paths = get_trajectories_path('blocksworld')
# model = rosame.learn(domain_ref_path, traj_paths)
# print(model)

"""=========================
PISAM TESTING
=========================="""
pisam = PISAM()

pisam_traces_dir_path = Path(benchmark_path / 'data' / amlgym_domain_name / 'training' / 'pi_sam_traces')
pisam_trajectory_paths = sorted(str(p) for p in pisam_traces_dir_path.glob("trace_*/*.trajectory"))

pisam_model = pisam.learn(domain_ref_path, pisam_trajectory_paths)
print(pisam_model)

pisam_domain_eval_path = f'PISAM_{amlgym_domain_name}_domain_learned.pddl'
with open(pisam_domain_eval_path, 'w') as f:
    f.write(pisam_model)


precision = syntactic_precision(pisam_domain_eval_path, domain_ref_path)
print(precision)

recall = syntactic_recall(pisam_domain_eval_path, domain_ref_path)
print(recall)

print(f"problem_paths: {probs_paths}")
metrics = problem_solving(pisam_domain_eval_path, domain_ref_path, probs_paths, timeout=60)
print(metrics)



"""=========================
PO-ROSAME TESTING
=========================="""
po_rosame = PO_ROSAME()

rosame_traj_paths = {
    "blocksworld": [f'{benchmark_path}/data/blocksworld/training/rosame_trace/problem7_images/problem7.trajectory'],
    "npuzzle": [f'{benchmark_path}/data/npuzzle/training/rosame_trace/eight01x_images/eight01x.trajectory']
}

rosame_model = po_rosame.learn(domain_ref_path, rosame_traj_paths[amlgym_domain_name])
print(rosame_model)


porosame_domain_eval_path = f'POROSAME_{amlgym_domain_name}_domain_learned.pddl'
with open(porosame_domain_eval_path, 'w') as f:
    f.write(rosame_model)


precision = syntactic_precision(porosame_domain_eval_path, domain_ref_path)
print(precision)

recall = syntactic_recall(porosame_domain_eval_path, domain_ref_path)
print(recall)

print(f"problem_paths: {probs_paths}")
metrics = problem_solving(porosame_domain_eval_path, domain_ref_path, probs_paths, timeout=60)
print(metrics)