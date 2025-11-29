"""
Demo version of noise generator - processes only trace_0 for testing
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmark.noise_generator import add_llm_noise_to_blocks_data

if __name__ == "__main__":
    # Process only trace_0 as a demonstration
    data_dir = Path(__file__).parent / "data" / "blocks" / "training"

    # Temporarily rename other traces so only trace_0 is processed
    our_traces_dir = data_dir / "our_algorithms_traces"
    trace_0_dir = our_traces_dir / "trace_0"

    if not trace_0_dir.exists():
        print("Error: trace_0 not found")
        sys.exit(1)

    print("="*80)
    print("DEMO: Processing only trace_0 (16 images)")
    print("This demonstrates the noise generation process")
    print("="*80)
    print()

    # Process just trace_0 by temporarily moving other traces
    import shutil
    temp_dir = our_traces_dir.parent / "temp_traces"
    temp_dir.mkdir(exist_ok=True)

    # Move all traces except trace_0
    for trace_dir in our_traces_dir.iterdir():
        if trace_dir.name != "trace_0" and trace_dir.is_dir():
            shutil.move(str(trace_dir), str(temp_dir / trace_dir.name))

    # NOTE: We cannot skip ROSAME trace anymore since our traces depend on it
    # The full ROSAME trace must be processed first to generate the trajectory
    # that we split for our traces

    try:
        add_llm_noise_to_blocks_data(
            data_dir=data_dir,
            use_uncertain=True
        )
    finally:
        # Restore all traces
        print("\nRestoring other traces...")
        for trace_dir in temp_dir.iterdir():
            shutil.move(str(trace_dir), str(our_traces_dir / trace_dir.name))
        temp_dir.rmdir()
        print("✓ Restored")
