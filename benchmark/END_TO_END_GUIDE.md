# Benchmark System - End-to-End Guide

## What Was Built

A comprehensive benchmark system to compare three action model learning algorithms:
- **PI-SAM**: Standard SAM learning
- **Noisy PI-SAM**: Conflict-driven patch search
- **ROSAME**: Robust Statistical Action Model Estimation

## What I Did - Complete Breakdown

### 🔧 Stage 1: Domain Equalization

**Problem**: ROSAME uses blocks domain WITHOUT the `handfull` predicate, but our domain has it.

**Solution**: Created equalized domain files to ensure fair comparison.

**Files Created:**
```
benchmark/domains/blocks/
├── blocks_no_handfull.pddl          # PDDL domain without handfull
├── prompts.py                        # LLM prompts updated (no handfull)
├── equalized_fluent_classifier.py    # Classifier using new prompts
└── __init__.py                       # Domain config
```

**Key Changes:**
- Removed `(handfull ?robot)` predicate from domain
- Updated all 4 actions (pick-up, put-down, stack, unstack):
  - Removed handfull from preconditions
  - Removed handfull from effects
- Updated LLM prompts to NOT ask about handfull
- Created fluent classifier that doesn't generate handfull predicates

---

### 📊 Stage 2: Training Data Generation

**Problem**: Need training data for all algorithms from the same problem.

**Solution**: Generate one long trajectory and split it.

**Script**: `benchmark/data_generator.py`

**What it does:**
1. Uses PDDLGym to create 100-step trajectory from `problem1.pddl`
2. Renders 101 images (states 0-100)
3. Saves full trajectory for ROSAME
4. Splits into 6 traces of 15 steps each for our algorithms

**Output Structure:**
```
benchmark/data/blocks/training/
├── rosame_trace/
│   ├── images/
│   │   ├── state_0000.png    # 101 images total
│   │   ├── state_0001.png
│   │   └── ...
│   ├── actions.json           # List of 100 actions
│   └── problem1.pddl_trajectory.json  # Ground truth
│
└── our_algorithms_traces/
    ├── trace_0/
    │   ├── images/
    │   │   ├── state_0000.png  # 16 images (renumbered from ROSAME states 0-15)
    │   │   └── ...
    │   ├── actions.json        # 15 actions
    │   ├── trace_metadata.json # start_state: 0, end_state: 15
    │   └── problem1_trace_0_trajectory.json  # Ground truth
    ├── trace_1/  # States 15-30
    ├── trace_2/  # States 30-45
    ├── trace_3/  # States 45-60
    ├── trace_4/  # States 60-75
    └── trace_5/  # States 75-90
```

**Key Feature**: All traces come from SAME trajectory, ensuring consistency.

---

### 🎨 Stage 3: LLM Noise Addition

**Problem**: Need to add vision noise to make data realistic (LLM may misclassify predicates).

**Solution**: Process all images with LLM vision pipeline.

**Script**: `benchmark/noise_generator.py`

**What it does:**

**Step 1: Process ROSAME trace (101 images)**
1. Detect objects from first image using LLMBlocksObjectDetector
2. For each of 101 images:
   - Classify all predicates using EqualizedBlocksFluentClassifier
   - Get confidence scores: 0 (FALSE), 1 (UNCERTAIN), 2 (TRUE)
3. Build full imaged trajectory (100 steps)
4. Save to ROSAME directory:
   - `problem1.trajectory` - Full trajectory in PDDL format
   - `problem1.masking_info` - Unknown predicates per state
   - `rosame_probability_observations.json` - Probabilities for ROSAME

**Step 2: Split for our algorithm traces**
1. For each trace_0 through trace_5:
   - Load trace_metadata.json to get start_state and end_state
   - Extract relevant portion: `rosame_trajectory[start_state:end_state]`
   - Save to trace directory:
     - `problem1.trajectory` - Split portion
     - `problem1.masking_info` - Split masking info

**Critical Insight**:
- All 6 traces get SAME LLM classifications (from ROSAME processing)
- No re-classification - just splitting
- Ensures all algorithms see IDENTICAL noisy observations
- Fair comparison!

**Example Flow:**
```
ROSAME processing:
  state_0000.png → LLM → predicates with scores → store
  state_0001.png → LLM → predicates with scores → store
  ...
  state_0100.png → LLM → predicates with scores → store

Build full trajectory → problem1.trajectory (100 steps)

Split for trace_0 (states 0-15):
  Extract steps 0-14 → trace_0/problem1.trajectory
  Extract masking states 0-15 → trace_0/problem1.masking_info

Split for trace_1 (states 15-30):
  Extract steps 15-29 → trace_1/problem1.trajectory
  Extract masking states 15-30 → trace_1/problem1.masking_info
```

**Current Status**: Running in background (~25 min so far)

---

### 🧪 Stage 4: Experiment Framework

**Problem**: Need to run all algorithms and collect results.

**Solution**: Created experiment runner framework.

**Script**: `benchmark/experiment_runner.py`

**What it does:**

**1. Select Test Problems (14 total)**
- 4 from pddl/blocks (excluding problem1.pddl used for training)
- 5 from pddl/blocks_test
- 5 from pddl/blocks_medium

**2. Run PI-SAM Experiment**
- Load domain: `blocks_no_handfull.pddl`
- Load all 6 training traces:
  - Read `trace_N/problem1.trajectory`
  - Read `trace_N/problem1.masking_info`
  - Parse using pddl_plus_parser
  - Create masked observations
- Run `PiSAMLearner.learn_action_model(observations)`
- Save learned model to `results/blocks/pisam/learned_model.pddl`
- Evaluate on 14 test problems (TODO)
- Record metrics: learning time, accuracy, success rates

**3. Run Noisy PI-SAM Experiment**
- Same as PI-SAM but uses:
  - `ConflictDrivenPatchSearch`
  - `SimpleNoisyPisamLearner`
- Attempts to repair conflicts in learned model
- (Currently placeholder - needs implementation)

**4. Run ROSAME Experiment**
- Use ROSAME's probability-based observations
- Call ROSAME external system (if available)
- (Currently placeholder - needs implementation)

**5. Save Results**
- All results to `results/blocks/benchmark_results.json`
- Individual algorithm results in subdirectories

---

### 📈 Stage 5: Results Analysis (Skeleton Created)

**Script**: `benchmark/results_analyzer.py`

**What it will do:**
1. Load results from all algorithms
2. Generate `comparison.csv` with metrics:
   - Algorithm name
   - Learning time
   - Per-test-problem: success, plan length, optimality
3. Create visualization plots (Stage 6)

---

## End-to-End Execution Steps

### Prerequisites
```bash
# Activate virtual environment
source venv11/bin/activate

# Ensure config.yaml has OpenAI API key
```

### Step 1: Generate Training Data (~2 minutes)
```bash
python benchmark/data_generator.py --domain blocks --num-steps 100 --trace-length 15
```

**Output:**
- Creates 101 images for ROSAME trace
- Creates 6 traces × 16 images for our algorithms
- Total: 197 images

**Verification:**
```bash
# Check ROSAME trace
ls benchmark/data/blocks/training/rosame_trace/images/ | wc -l
# Should show: 101

# Check our traces
ls benchmark/data/blocks/training/our_algorithms_traces/
# Should show: trace_0 trace_1 trace_2 trace_3 trace_4 trace_5
```

---

### Step 2: Add LLM Noise (~10-30 minutes)
```bash
python benchmark/noise_generator.py --domain blocks
```

**What happens:**
- Processes 101 images with LLM vision
- Makes ~202 API calls (1 object detection + 101 fluent classifications + overhead)
- Creates .trajectory and .masking_info files

**Why it's slow:**
- OpenAI API rate limiting
- Network latency
- Sequential processing

**Verification:**
```bash
# Check ROSAME files
ls -lh benchmark/data/blocks/training/rosame_trace/problem1.*
# Should show: problem1.trajectory, problem1.masking_info

# Check trace files
ls benchmark/data/blocks/training/our_algorithms_traces/trace_0/problem1.*
# Should show: problem1.trajectory, problem1.masking_info

# Count all trajectory files (should be 7: 1 ROSAME + 6 traces)
find benchmark/data/blocks/training -name "problem1.trajectory" | wc -l
```

---

### Step 3: Run Experiments (~10-30 minutes)
```bash
python benchmark/experiment_runner.py --domain blocks
```

**What happens:**
1. Loads domain and training data
2. Runs PI-SAM learning on 6 traces
3. Runs Noisy PI-SAM learning (when implemented)
4. Runs ROSAME (when implemented)
5. Evaluates each learned model on 14 test problems
6. Saves results

**Output:**
```
benchmark/results/blocks/
├── benchmark_results.json        # Overall results
├── pisam/
│   ├── learned_model.pddl        # Learned domain model
│   └── results.json              # PI-SAM specific results
├── noisy_pisam/
│   ├── learned_model.pddl
│   └── results.json
└── rosame/
    ├── learned_model.pddl
    └── results.json
```

---

### Step 4: Analyze Results (~1 minute)
```bash
python benchmark/results_analyzer.py --domain blocks
```

**What happens:**
1. Loads all experiment results
2. Generates comparison CSV
3. Creates visualization plots

**Output:**
```
benchmark/results/blocks/
├── comparison.csv               # Metrics comparison table
└── plots/
    ├── learning_time.png        # Bar chart
    ├── success_rates.png        # Bar chart
    ├── plan_lengths.png         # Box plot
    └── per_problem.png          # Grouped bar chart
```

---

## Quick Summary: What Each File Does

### Core Scripts
1. **data_generator.py**: Creates trajectory images from PDDLGym
2. **noise_generator.py**: Adds LLM vision noise to images
3. **experiment_runner.py**: Runs all algorithms and collects results
4. **results_analyzer.py**: Generates CSVs and plots

### Domain Files
5. **blocks_no_handfull.pddl**: Equalized PDDL domain
6. **prompts.py**: LLM vision prompts
7. **equalized_fluent_classifier.py**: Vision classifier

### Documentation
8. **README.md**: User guide
9. **END_TO_END_GUIDE.md**: This file
10. **PROGRESS_SUMMARY.md**: Development progress

---

## Current Status

✅ **Stage 1**: Domain equalization - DONE
✅ **Stage 2**: Training data generation - DONE
🔄 **Stage 3**: LLM noise addition - RUNNING (~25 min, 10-35 min remaining)
✅ **Stage 4**: Experiment framework - DONE (implementation pending)
⏳ **Stage 5**: Results analysis - Skeleton ready
⏳ **Stage 6**: Visualization - Skeleton ready

---

## What Needs Implementation

After noise generation completes:

1. **PI-SAM Integration** (~1-2 hours)
   - Already have `PiSAMLearner`
   - Need to integrate with test problem evaluation
   - Measure accuracy, success rates, plan quality

2. **Noisy PI-SAM Integration** (~2-3 hours)
   - Have `ConflictDrivenPatchSearch` and `SimpleNoisyPisamLearner`
   - Need to integrate with experiment framework
   - Add conflict resolution metrics

3. **ROSAME Integration** (~1-2 hours)
   - Prepare data in ROSAME's format
   - Run ROSAME external system (if available)
   - Parse ROSAME output

4. **Test Evaluation** (~2-3 hours)
   - For each learned model:
     - Load test problem
     - Generate plan using Fast-Downward
     - Check if plan valid
     - Measure plan length
     - Compare to optimal plan

5. **Results Analysis** (~1-2 hours)
   - CSV generation with all metrics
   - Statistical tests
   - Visualization plots

**Total remaining work: ~8-13 hours**

---

## Example: Complete Run

```bash
# 1. Generate data (2 min)
python benchmark/data_generator.py --domain blocks

# 2. Add noise (10-30 min)
python benchmark/noise_generator.py --domain blocks

# Wait for completion...

# 3. Run experiments (10-30 min)
python benchmark/experiment_runner.py --domain blocks

# 4. Analyze results (1 min)
python benchmark/results_analyzer.py --domain blocks

# 5. View results
cat benchmark/results/blocks/comparison.csv
open benchmark/results/blocks/plots/
```

**Total time: ~25-65 minutes** (mostly waiting for LLM API calls)

---

## Key Insights

### Why This Approach Works

1. **Fair Comparison**: All algorithms use SAME noisy observations
2. **Realistic**: LLM vision adds real-world noise (misclassifications)
3. **Reproducible**: Same training data, same test problems
4. **Comprehensive**: 14 test problems, multiple metrics
5. **Automated**: One command to run everything

### Design Decisions

1. **Why split from ROSAME trace?**
   - Ensures identical noisy observations across all algorithms
   - Prevents bias from different LLM classifications
   - Fair comparison

2. **Why remove handfull predicate?**
   - ROSAME uses domain without it
   - handfull is redundant (always opposite of handempty)
   - Levels the playing field

3. **Why 6 traces of 15 steps?**
   - 100 steps / 15 = 6.67 → 6 complete traces
   - 15 steps is enough for learning patterns
   - Multiple traces help with generalization

4. **Why these test problems?**
   - 14 problems cover different difficulty levels
   - Exclude training problem to test generalization
   - Standard PDDLGym benchmarks

---

**Last Updated**: 2025-11-29 (Noise generator running ~25 min)
