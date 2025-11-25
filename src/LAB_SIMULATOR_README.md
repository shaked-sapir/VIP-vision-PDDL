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

### Running the Lab Simulator

Simply run the script:

```bash
cd /Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL
source venv11/bin/activate
python src/lab_simulator.py
```

The script is pre-configured to use:
- **Working directory**: `src/noisy_conflict_experiments/noisy_conflict_cv__steps=20__model=gpt-5-mini__temp=1.0__22-11-2025T21:08:22`
- **Domain file**: `blocks.pddl`
- **Cross-validation**: 5 folds
- **Conflict search**: Unlimited nodes, equal patch costs (1:1)

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
from src.lab_simulator import LabSimulatorRunner

# Create simulator
lab = LabSimulatorRunner(
    working_directory_path=Path("path/to/experiment"),
    domain_file_name="blocks.pddl",
    n_split=5,
    fluent_patch_cost=1,
    model_patch_cost=1,
    max_search_nodes=100,  # Limit search for efficiency
    seed=42
)

# Run comparison
results_file = lab.run_cross_validation()
print(f"Results: {results_file}")
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
| **Purpose** | Generate trajectories + CV with PI-SAM | Generate trajectories + CV with conflict search | Compare learners on existing data |
| **Trajectory Generation** | Yes | Yes | No (uses existing) |
| **Learners** | PI-SAM only | Conflict search only | Both PI-SAM + Conflict search |
| **Input** | Problems list | Problems list | Working directory with data |
| **Output** | Single learner results | Single learner results | Side-by-side comparison |
| **Use Case** | Initial experiments | Noisy data experiments | Learner comparison |

## When to Use Lab Simulator

✅ **Use Lab Simulator when:**
- You already have generated trajectories with masking info
- You want to compare PI-SAM vs Conflict Search on the same data
- You want to analyze the differences in learning strategies
- You want to measure the overhead of conflict resolution

❌ **Don't use Lab Simulator when:**
- You need to generate new trajectories (use regular/noisy simulators)
- You only want to test one learner (use specific simulator)
- You don't have masking info files (generate them first)

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
