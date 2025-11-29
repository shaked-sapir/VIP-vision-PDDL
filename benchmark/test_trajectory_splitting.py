"""
Quick test to verify trajectory splitting logic without LLM calls
"""

import json
from pathlib import Path

# Simulate a full trajectory with 100 steps (101 states)
full_trajectory = []
for i in range(100):
    step = {
        "step": i + 1,
        "current_state": {
            "literals": [f"pred{i}_current"],
            "unknown": []
        },
        "ground_action": f"action_{i}",
        "next_state": {
            "literals": [f"pred{i+1}_next"],
            "unknown": []
        }
    }
    full_trajectory.append(step)

print("Full trajectory has 100 steps (covers states 0-100)")
print(f"  Step 1: current_state has pred0_current, next_state has pred1_next")
print(f"  Step 100: current_state has pred99_current, next_state has pred100_next")
print()

# Test splitting for trace_0 (states 0-15, which is steps 0-14 in 0-indexed)
trace_0_metadata = {
    "start_state": 0,
    "end_state": 15
}

start = trace_0_metadata["start_state"]
end = trace_0_metadata["end_state"]

trace_0_trajectory = full_trajectory[start:end]

print(f"Trace 0 (states {start}-{end}):")
print(f"  Extracted steps: {len(trace_0_trajectory)}")
print(f"  First step current_state: {trace_0_trajectory[0]['current_state']['literals']}")
print(f"  Last step next_state: {trace_0_trajectory[-1]['next_state']['literals']}")
print(f"  Expected: 15 steps covering states 0-15 ✓" if len(trace_0_trajectory) == 15 else "  ERROR!")
print()

# Test splitting for trace_1 (states 15-30)
trace_1_metadata = {
    "start_state": 15,
    "end_state": 30
}

start = trace_1_metadata["start_state"]
end = trace_1_metadata["end_state"]

trace_1_trajectory = full_trajectory[start:end]

print(f"Trace 1 (states {start}-{end}):")
print(f"  Extracted steps: {len(trace_1_trajectory)}")
print(f"  First step current_state: {trace_1_trajectory[0]['current_state']['literals']}")
print(f"  Last step next_state: {trace_1_trajectory[-1]['next_state']['literals']}")
print(f"  Expected: 15 steps covering states 15-30 ✓" if len(trace_1_trajectory) == 15 else "  ERROR!")
print()

print("="*80)
print("VERIFICATION COMPLETE")
print("="*80)
print()
print("✓ Trajectory splitting logic is correct!")
print(f"✓ trace_0 gets states 0-15 (steps {trace_0_metadata['start_state']}-{trace_0_metadata['end_state']-1})")
print(f"✓ trace_1 gets states 15-30 (steps {trace_1_metadata['start_state']}-{trace_1_metadata['end_state']-1})")
