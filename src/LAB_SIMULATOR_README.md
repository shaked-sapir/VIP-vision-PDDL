# Lab Simulator - Multi-Learner Comparison Framework

## Overview

The `lab_simulator.py` provides a ready-to-run comparison framework for evaluating **PI-SAM** vs **Noisy Conflict Search** on the same cross-validation folds. It runs both learning algorithms on identical data and combines their results into a single CSV for easy comparison.

## Key Features

- **Off-Shelf Solution**: No need to implement abstract classes or configurations
- **Direct Comparison**: Both learners run on the same masked observations
- **Comprehensive Results**: Combines learning reports, patch statistics, and timing info
- **Organized Output**: Separate directories per learner + combined comparison CSV

## Architecture

### LabSimulatorRunner Class

The main class that orchestrates the multi-learner cross-validation:

```python
class LabSimulatorRunner:
    """Compare PI-SAM vs Noisy Conflict Search on the same cross-validation folds."""
```

**Key Methods:**

1. **`run_pisam_learner()`** - Runs standard PI-SAM learning
2. **`run_noisy_conflict_search()`** - Runs conflict-driven patch search
3. **`learn_model_offline()`** - Runs both learners on a single fold
4. **`run_cross_validation()`** - Main entry point, runs all folds

### Workflow

```
Working Directory (with trajectories + masking info)
    ↓
Create K-Fold Splits
    ↓
Initialize Statistics Infrastructure for Both Learners:
    ├─ Learning Statistics Managers
    ├─ Semantic Performance Calculators
    └─ Domain Validators
    ↓
For each fold:
    ├─ Load & Mask Observations (once, shared)
    ├─ Run PI-SAM Learner:
    │   ├─ Learn model
    │   ├─ Export domain
    │   ├─ Validate domain
    │   ├─ Calculate semantic performance
    │   └─ Export fold statistics
    └─ Run Noisy Conflict Search:
        ├─ Learn model with conflict resolution
        ├─ Export domain
        ├─ Validate domain
        ├─ Calculate semantic performance
        └─ Export fold statistics (with patch info)
    ↓
Finalize Results for Both Learners:
    ├─ Export combined semantic performance
    ├─ Export complete validation statistics
    └─ Export complete learning statistics
```

## Directory Structure

After running, the working directory will contain:

```
working_dir/
├── blocks.pddl                                    # Domain file
├── problem1.pddl, problem1.trajectory             # Problem files & trajectories
├── problem1.masking_info                          # Masking info files
├── fold_0/, fold_1/, ...                          # Cross-validation folds
└── ultimate_results_directory/                    # ★ MAIN OUTPUT DIRECTORY
    ├── pisam/                                     # PI-SAM complete results
    │   ├── sam_learning_blocks_combined_semantic_performance.csv
    │   ├── sam_learning_blocks_semantic_performance.csv
    │   ├── sam_learning_statistics.csv
    │   ├── validation_statistics.csv
    │   └── domains_backup/
    │       ├── sam_learning_fold_0_blocks.pddl
    │       ├── sam_learning_fold_1_blocks.pddl
    │       └── ...
    └── noisy_conflict_search/                     # Noisy Conflict Search complete results
        ├── sam_learning_blocks_combined_semantic_performance.csv
        ├── sam_learning_blocks_semantic_performance.csv
        ├── sam_learning_statistics.csv
        ├── validation_statistics.csv
        └── domains_backup/
            ├── sam_learning_fold_0_blocks.pddl
            ├── sam_learning_fold_1_blocks.pddl
            └── ...
```

Each learner directory contains the **full experiment output structure**:
- **Combined semantic performance** - aggregated results across all folds
- **Per-fold semantic performance** - detailed per-fold metrics
- **Learning statistics** - action model learning statistics
- **Validation statistics** - domain validation metrics
- **Domains backup** - learned domain files for each fold

## Usage

### Configuration Modes

The lab simulator now supports **config-driven** operation with two modes:

#### Mode 1: Use Existing Trajectories (Default)

```python
# In main():
domain_name = 'blocks'  # or 'hanoi'
generate_trajectories = False
```

Uses existing trajectory directory for quick testing and iteration.

#### Mode 2: Generate Trajectories from Scratch

```python
# In main():
domain_name = 'blocks'  # or 'hanoi'
generate_trajectories = True
```

Automatically generates trajectories using LLM vision pipeline based on config.

### Running the Lab Simulator

Simply run the script:

```bash
cd /Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL
source venv11/bin/activate
python src/lab_simulator.py
```

### Switching Domains

To switch from blocks to hanoi, just change one variable:

```python
# In main():
domain_name = 'hanoi'  # That's it!
```

All domain-specific parameters (gym environment, object detection models, fluent classification models, paths) are automatically loaded from `config.yaml`.

### Customizing Parameters

Edit the `main()` function in `lab_simulator.py`:

```python
lab = LabSimulatorRunner(
    working_directory_path=Path("your/working/dir"),
    domain_file_name="blocks.pddl",
    problem_prefix="problem",
    n_split=5,                    # Number of folds
    fluent_patch_cost=1,          # Cost of fluent patches
    model_patch_cost=1,           # Cost of model patches
    max_search_nodes=None,        # None = unlimited search
    seed=42                       # Random seed
)
```

### Using as a Library

```python
from pathlib import Path
from src.lab_simulator import LabSimulatorRunner, create_trajectories_for_lab
from src.utils.config import load_config

# Load config
config = load_config()
domain_name = 'blocks'
domain_config = config['domains'][domain_name]

# Generate trajectories
working_dir = create_trajectories_for_lab(
    domain_name=domain_name,
    gym_domain_name=domain_config['gym_domain_name'],
    problems=[f"problem{i}" for i in range(10)],
    num_steps=20,
    openai_apikey=config['openai']['api_key'],
    object_detection_model_name=domain_config['object_detection']['model_name'],
    object_detection_temperature=domain_config['object_detection']['temperature'],
    fluent_classification_model_name=domain_config['fluent_classification']['model_name'],
    fluent_classification_temperature=domain_config['fluent_classification']['temperature'],
    pddl_domain_file=Path(config['paths']['root']) / domain_config['domain_file'],
    problem_dir=Path(config['paths']['root']) / domain_config['problem_dir']
)

# Create simulator
lab = LabSimulatorRunner(
    working_directory_path=working_dir,
    domain_file_name=f"{domain_name}.pddl",
    n_split=5,
    fluent_patch_cost=1,
    model_patch_cost=1,
    max_search_nodes=100,  # Limit search for efficiency
    seed=42
)

# Run comparison
pisam_dir, noisy_dir = lab.run_cross_validation()
print(f"PI-SAM results: {pisam_dir}")
print(f"Noisy results: {noisy_dir}")
```

## Configuration File

The lab simulator loads domain-specific settings from `config.yaml`:

```yaml
domains:
  blocks:
    gym_domain_name: "PDDLEnvBlocks-v0"
    domain_file: "src/domains/blocks/blocks.pddl"
    problem_dir: "src/domains/blocks/problems"
    object_detection:
      model_name: "gpt-4o"
      temperature: 1.0
    fluent_classification:
      model_name: "gpt-4o"
      temperature: 0.0

  hanoi:
    gym_domain_name: "PDDLEnvHanoi-v0"
    domain_file: "src/domains/hanoi/hanoi.pddl"
    problem_dir: "src/domains/hanoi/problems"
    object_detection:
      model_name: "gpt-4o"
      temperature: 1.0
    fluent_classification:
      model_name: "gpt-4o"
      temperature: 0.0

openai:
  api_key: "sk-..."

paths:
  root: "/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL"
```

This allows switching domains by just changing `domain_name` in the code - all paths, models, and parameters are loaded automatically.

## Trajectory Generation

### `create_trajectories_for_lab()` Function

Generates trajectories using the appropriate trajectory handler for the domain.

**Process**:
1. Selects trajectory handler based on domain:
   - `blocks` → `LLMBlocksImageTrajectoryHandler`
   - `hanoi` → `LLMHanoiImageTrajectoryHandler`
2. Generates trajectories sequentially for each problem
3. Extracts masking info from LLM results (unknown predicates)
4. Saves to working directory:
   - `*.trajectory` files
   - `*.masking_info` files
   - `*.pddl` problem files
   - Domain file

**Parameters**:
- `domain_name`: 'blocks' or 'hanoi'
- `gym_domain_name`: Gym environment name
- `problems`: List of problem names (e.g., ['problem0', 'problem1'])
- `num_steps`: Steps per trajectory
- `openai_apikey`: OpenAI API key
- `object_detection_model_name`: Model for object detection
- `object_detection_temperature`: Temperature for object detection
- `fluent_classification_model_name`: Model for fluent classification
- `fluent_classification_temperature`: Temperature for fluent classification
- `pddl_domain_file`: Path to PDDL domain file
- `problem_dir`: Directory containing problem files
- `experiment_dir_path`: Base directory for experiments (default: "lab_experiments")

**Returns**: Path to working directory with generated trajectories

**Example**:
```python
working_dir = create_trajectories_for_lab(
    domain_name='hanoi',
    gym_domain_name='PDDLEnvHanoi-v0',
    problems=['problem0', 'problem1', 'problem2'],
    num_steps=10,
    openai_apikey='sk-...',
    object_detection_model_name='gpt-4o',
    object_detection_temperature=1.0,
    fluent_classification_model_name='gpt-4o',
    fluent_classification_temperature=0.0,
    pddl_domain_file=Path('src/domains/hanoi/hanoi.pddl'),
    problem_dir=Path('src/domains/hanoi/problems')
)
# Returns: lab_experiments/hanoi_lab_cv__steps=10__<timestamp>/
```

## Output Format

Each learner produces a **complete set of result files** in its own directory:

### 1. Combined Semantic Performance CSV

**File**: `sam_learning_blocks_combined_semantic_performance.csv`

Contains aggregated semantic performance metrics across all folds:
- Fold number
- Number of trajectories
- Precision, Recall, F1 scores
- Action-level statistics
- **Noisy-only**: Solution status, conflicts, patch counts

### 2. Per-Fold Semantic Performance CSV

**File**: `sam_learning_blocks_semantic_performance.csv`

Detailed per-fold semantic performance with action-level metrics.

### 3. Learning Statistics CSV

**File**: `sam_learning_statistics.csv`

Action model learning statistics:
- Learning time
- Number of actions learned
- Precondition/effect counts
- **Noisy-only**: Patch diff details

### 4. Validation Statistics CSV

**File**: `validation_statistics.csv`

Domain validation results:
- Test set problem solving
- Plan quality metrics
- Solver performance

### Example: Comparing Results

```python
import pandas as pd

# Load PI-SAM results
pisam_results = pd.read_csv("ultimate_results_directory/pisam/sam_learning_blocks_combined_semantic_performance.csv")

# Load Noisy Conflict Search results
noisy_results = pd.read_csv("ultimate_results_directory/noisy_conflict_search/sam_learning_blocks_combined_semantic_performance.csv")

# Compare
print("PI-SAM F1:", pisam_results['f1'].mean())
print("Noisy F1:", noisy_results['f1'].mean())
```

## Comparison with Other Simulators

| Feature | Regular Simulator | Noisy Conflict Search Simulator | Lab Simulator |
|---------|-------------------|--------------------------------|---------------|
| **Purpose** | Generate trajectories + CV with PI-SAM | Generate trajectories + CV with conflict search | Compare learners side-by-side |
| **Trajectory Generation** | Yes | Yes | **Yes (optional)** |
| **Learners** | PI-SAM only | Conflict search only | Both PI-SAM + Conflict search |
| **Input** | Problems list | Problems list | Working directory OR config |
| **Output** | Single learner results | Single learner results | Side-by-side comparison |
| **Config-Driven** | Partial | Partial | **Full config support** |
| **Domain Switching** | Manual | Manual | **Change one variable** |
| **Use Case** | Initial experiments | Noisy data experiments | Learner comparison + evaluation |

## When to Use Lab Simulator

✅ **Use Lab Simulator when:**
- You want to compare PI-SAM vs Conflict Search on the same data
- You want to analyze the differences in learning strategies
- You want to measure the overhead of conflict resolution
- **NEW**: You want to generate trajectories and immediately compare learners
- **NEW**: You want config-driven experiments with easy domain switching
- You already have generated trajectories with masking info

❌ **Don't use Lab Simulator when:**
- You only want to test one learner (use specific simulator for efficiency)
- You need fine-grained control over trajectory generation per learner

## Performance Considerations

1. **Learning Time**: Conflict search is typically slower than PI-SAM
   - Limit with `max_search_nodes` for faster experiments
   - Use `None` for thorough comparison

2. **Memory**: Both learners work on the same observations in memory
   - Consider fewer folds for large datasets
   - Process runs sequentially (one fold at a time)

3. **Disk Space**: Stores domains and reports for both learners
   - ~2x storage compared to single learner
   - Cleaned up automatically between folds

## Troubleshooting

### Missing Masking Info Files

**Error**: `Masking info file not found for problemX`

**Solution**: Ensure `.masking_info` files exist in the working directory:
```bash
ls -la working_dir/*.masking_info
```

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'src.pi_sam'`

**Solution**: Activate the virtual environment:
```bash
source venv11/bin/activate
```

### Empty Results

**Error**: No results files generated

**Solution**: Check that result directories were created:
```bash
ls -la working_dir/ultimate_results_directory/pisam/
ls -la working_dir/ultimate_results_directory/noisy_conflict_search/
```

Look for the combined semantic performance files:
```bash
ls -la working_dir/ultimate_results_directory/pisam/sam_learning_blocks_combined_semantic_performance.csv
ls -la working_dir/ultimate_results_directory/noisy_conflict_search/sam_learning_blocks_combined_semantic_performance.csv
```

## Advanced Usage

### Analyzing Results

```python
import pandas as pd

# Load PI-SAM combined semantic performance
pisam_df = pd.read_csv("ultimate_results_directory/pisam/sam_learning_blocks_combined_semantic_performance.csv")

# Load Noisy Conflict Search combined semantic performance
noisy_df = pd.read_csv("ultimate_results_directory/noisy_conflict_search/sam_learning_blocks_combined_semantic_performance.csv")

# Compare F1 scores
print("PI-SAM Average F1:", pisam_df['f1'].mean())
print("Noisy Average F1:", noisy_df['f1'].mean())

# Compare learning times
pisam_stats = pd.read_csv("ultimate_results_directory/pisam/sam_learning_statistics.csv")
noisy_stats = pd.read_csv("ultimate_results_directory/noisy_conflict_search/sam_learning_statistics.csv")

print("\nLearning Times:")
print("PI-SAM:", pisam_stats['learning_time'].astype(float).mean())
print("Noisy:", noisy_stats['learning_time'].astype(float).mean())

# Check conflict resolution (noisy only)
if 'solution_found' in noisy_df.columns:
    print(f"\nNoisy Success Rate: {(noisy_df['solution_found'] == 'True').mean()}")
    print(f"Average Conflicts: {noisy_df['final_conflicts'].astype(int).mean()}")
```

### Custom Learner Integration

To add a new learner, modify `learn_model_offline()`:

```python
def learn_model_offline(self, fold_num, train_set_dir_path, test_set_dir_path):
    # ... existing code ...

    # Add your custom learner
    custom_domain, custom_report = self.run_custom_learner(
        partial_domain, masked_observations
    )
    custom_report["fold"] = str(fold_num)
    custom_report["learner"] = "custom"

    # Save custom results
    # ... save logic ...

    return pisam_report, noisy_report, custom_report
```

## Future Enhancements

Potential improvements:
- [ ] Parallel learner execution (if independent)
- [ ] Statistical significance testing between learners
- [ ] Visualization of comparison results
- [ ] Support for more than 2 learners
- [ ] Aggregate statistics across all folds
- [ ] Export to different formats (JSON, Excel)

## See Also

- `NOISY_CONFLICT_SEARCH_SIMULATOR_README.md` - Conflict search simulator
- `simulator.py` - Regular PI-SAM simulator
- `conflict_search.py` - Conflict-driven patch search implementation
