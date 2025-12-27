#!/usr/bin/env python3
"""Test if problem8 frame axiom propagation hangs"""

from pathlib import Path
from src.utils.pddl import propagate_frame_axioms_selective
import signal
import sys

def timeout_handler(signum, frame):
    print("TIMEOUT: Frame axiom propagation took too long!")
    sys.exit(1)

# Set timeout of 30 seconds
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)

try:
    problem8_dir = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/data/blocksworld/multi_problem_04-12-2025T12:00:44__model=gpt-5.1__steps=50__planner/training/trajectories/problem8")
    domain_path = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/domains/blocksworld/blocksworld.pddl")

    traj_path = problem8_dir / "problem8.trajectory"
    masking_path = problem8_dir / "problem8.masking_info"

    print(f"Testing frame axiom propagation on problem8...")
    print(f"Trajectory: {traj_path}")
    print(f"Masking: {masking_path}")
    print(f"Domain: {domain_path}")

    final_traj, final_masking = propagate_frame_axioms_selective(
        traj_path, masking_path, domain_path, gt_state_indices={0}, mode="after_gt_only"
    )

    print(f"SUCCESS! Frame axioms applied")
    print(f"Output trajectory: {final_traj}")
    print(f"Output masking: {final_masking}")

    signal.alarm(0)  # Cancel timeout

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
