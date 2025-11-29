from amlgym.benchmarks import *
from amlgym.algorithms import *
from amlgym.metrics import *

from benchmark.PO_ROSAME.PO_ROSAME import PO_ROSAME

print_metrics()

# rosame = get_algorithm('ROSAME')
# domain_ref_path = get_domain_path('blocksworld')
# traj_paths = get_trajectories_path('blocksworld')
# model = rosame.learn(domain_ref_path, traj_paths)
# print(model)

po_rosame = PO_ROSAME()
domain_ref_path = get_domain_path('blocksworld')
traj_paths = ['/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/data/blocksworld/training/rosame_trace/problem1.trajectory']
model = po_rosame.learn(domain_ref_path, traj_paths)
print(model)


domain_eval_path = 'domain_learned.pddl'
with open(domain_eval_path, 'w') as f:
    f.write(model)


precision = syntactic_precision(domain_eval_path, domain_ref_path)
print(precision)

recall = syntactic_recall(domain_eval_path, domain_ref_path)
print(recall)

probs_paths = get_problems_path('blocksworld')
print(probs_paths)

metrics = problem_solving(domain_eval_path, domain_ref_path, probs_paths, timeout=60)
print(metrics)