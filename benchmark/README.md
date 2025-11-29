# Benchmark System

Comprehensive benchmark for comparing action model learning algorithms:
- **PI-SAM** (Standard SAM learning)
- **Noisy PI-SAM** (Conflict-driven patch search)
- **ROSAME** (Robust Statistical Action Model Estimation)

## Quick Start

### 1. Generate Training Data

```bash
python benchmark/data_generator.py --domain blocks --num-steps 100 --trace-length 15
```

This creates:
- 100-step trajectory for ROSAME (101 images)
- 6 non-overlapping 15-step traces for our algorithms

### 2. Add LLM Noise

```bash
python benchmark/noise_generator.py --domain blocks
```

This processes all images with LLM vision and creates:
- `.trajectory` and `.masking_info` files for all traces
- Probability observations for ROSAME

**Note**: This step takes 10-20+ minutes for 101 LLM API calls.

### 3. Run Experiments

```bash
python benchmark/experiment_runner.py --domain blocks
```

This runs all three algorithms on 14 test problems and collects metrics.

### 4. Generate Results

```bash
python benchmark/results_analyzer.py --domain blocks
```

This creates CSV files and comparison plots.

## Directory Structure

```
benchmark/
├── README.md                    # This file
├── data_generator.py            # Generate training trajectories
├── noise_generator.py           # Add LLM vision noise
├── experiment_runner.py         # Run all experiments
├── results_analyzer.py          # Analyze and visualize results
│
├── domains/
│   └── blocks/
│       ├── blocks_no_handfull.pddl          # Equalized domain
│       ├── prompts.py                        # LLM prompts (no handfull)
│       └── equalized_fluent_classifier.py    # Fluent classifier
│
├── data/
│   └── blocks/
│       └── training/
│           ├── rosame_trace/                 # 100-step trace
│           │   ├── images/                   # 101 images
│           │   ├── problem1.trajectory       # Full LLM trajectory
│           │   ├── problem1.masking_info     # Full masking
│           │   └── rosame_probability_observations.json
│           └── our_algorithms_traces/
│               ├── trace_0/                  # States 0-15
│               │   ├── images/               # 16 images
│               │   ├── trace_metadata.json
│               │   ├── problem1.trajectory   # Split from ROSAME
│               │   └── problem1.masking_info # Split from ROSAME
│               ├── trace_1/                  # States 15-30
│               └── ...
│
└── results/
    └── blocks/
        ├── benchmark_results.json
        ├── pisam/
        │   ├── learned_model.pddl
        │   └── results.json
        ├── noisy_pisam/
        │   ├── learned_model.pddl
        │   └── results.json
        └── rosame/
            ├── learned_model.pddl
            └── results.json
```

## Test Problems

The benchmark uses 14 test problems:

**From `pddl/blocks` (4 problems)**:
- problem3.pddl, problem5.pddl, problem7.pddl, problem9.pddl
- (Excludes problem1.pddl which was used for training)

**From `pddl/blocks_test` (5 problems)**:
- problem2.pddl, problem4.pddl, problem6.pddl, problem8.pddl, problem10.pddl

**From `pddl/blocks_medium` (5 problems)**:
- problem0.pddl, problem2.pddl, problem3.pddl, problem4.pddl, problem5.pddl

## Key Features

### Domain Equalization
All experiments use the equalized blocks domain **without the `handfull` predicate**, matching ROSAME's definition.

### Consistent LLM Classifications
All training traces use the **same LLM classifications** from the ROSAME processing, ensuring fair comparison.

### Trajectory Splitting
Our algorithm traces are **split from the full ROSAME trajectory**:
- trace_0: steps 0-14 (states 0-15)
- trace_1: steps 15-29 (states 15-30)
- etc.

This ensures all algorithms see the same noisy observations.

## Evaluation Metrics

The benchmark collects:
- **Learning time**: Time to learn action model
- **Model accuracy**: Comparison with ground truth
- **Planning performance**: Success rate on test problems
- **Plan quality**: Plan length and optimality

## Results

Results are saved in:
- `results/blocks/benchmark_results.json` - Overall results
- `results/blocks/comparison.csv` - Metrics comparison table
- `results/blocks/plots/` - Visualization plots

## Implementation Status

- ✅ Stage 1: Domain equalization
- ✅ Stage 2: Training data generation
- ✅ Stage 3: LLM noise addition
- 🔄 Stage 4: Experiment execution (in progress)
- ⏳ Stage 5: Results analysis
- ⏳ Stage 6: Visualization

## Notes

- The noise generation step is the slowest (10-20+ min for 101 images)
- All traces use the same LLM noise for fairness
- Training data is excluded from test set
- Evaluation uses standard PDDLGym problem sets
