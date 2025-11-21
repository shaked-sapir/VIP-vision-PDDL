# Noisy Conflict Search Simulator

## Overview

The `noisy_conflict_search_simulator.py` extends the base `simulator.py` to use **conflict-driven patch search** during cross-validation instead of regular PI-SAM learning. This allows the system to detect and resolve conflicts in noisy observations by finding minimal patches (data flips and model constraints).

## Key Differences from Regular Simulator

| Feature | Regular Simulator | Noisy Conflict Search Simulator |
|---------|-------------------|--------------------------------|
| Learning Algorithm | PI-SAM (direct learning) | Conflict-Driven Patch Search |
| Handles Conflicts | No | Yes - detects and resolves |
| Output | Learned domain | Learned domain + patch statistics |
| Noise Handling | Assumes clean data | Finds minimal corrections |
| Search Strategy | N/A | Best-first with cost heuristic |
| Trajectory Generation | Parallel | Sequential (one by one) |

## Architecture

### NoisyConflictSearchExperimentRunner

Extends `OfflinePiSamExperimentRunner` to use conflict-driven search:

```python
class NoisyConflictSearchExperimentRunner(OfflinePiSamExperimentRunner):
    """
    Replaces PI-SAM learning with conflict-driven search.

    Key Methods:
    - _apply_learning_algorithm(): Uses ConflictDrivenPatchSearch instead of PISAMLearner
    - Enriches learning report with patch diff statistics
    """
```

**What it does:**
1. Inherits trajectory loading and masking from base runner
2. Overrides learning to use `ConflictDrivenPatchSearch`
3. Adds patch statistics to the learning report:
   - `solution_found`: Whether conflict-free model was found
   - `final_conflicts`: Number of unresolved conflicts
   - `solution_cost`: Total cost of patches applied
   - `model_patches_added/removed/changed`: Counts and details
   - `fluent_patches_added/removed`: Counts and details

### NoisyConflictSearchSimulator

Extends `Simulator` to use the new experiment runner:

```python
class NoisyConflictSearchSimulator(Simulator):
    """
    Simulator with conflict-driven search for cross-validation.

    Key Methods:
    - run_cross_validation_with_conflict_search(): Main entry point
    """
```

## Usage

### Basic Usage

```python
from pathlib import Path
from src.noisy_conflict_search_simulator import NoisyConflictSearchSimulator

# Create simulator
simulator = NoisyConflictSearchSimulator(
    domain_name="PDDLEnvBlocks-v0",
    openai_apikey="your-api-key",
    pddl_domain_file=Path("src/domains/blocks/blocks.pddl"),
    pddl_problem_dir=Path("src/domains/blocks"),
    visual_components_model_name="gpt-4o",
    experiment_dir_path=Path("noisy_conflict_experiments"),
    fluent_patch_cost=1,
    model_patch_cost=1,
    max_search_nodes=100,
    seed=42
)

# Run cross-validation
results_dir = simulator.run_cross_validation_with_conflict_search(
    problems=["problem1", "problem2", "problem3"],
    num_steps=25,
    experiment_name="noisy_conflict_cv"
)

print(f"Results saved to: {results_dir}")
```

### Configuration Parameters

#### Simulator Parameters

- **`fluent_patch_cost`** (int, default=1): Cost of adding a fluent patch in search
  - Higher values penalize data corrections
  - Lower values prefer fixing data over constraining model

- **`model_patch_cost`** (int, default=1): Cost of adding a model constraint in search
  - Higher values penalize model constraints
  - Lower values prefer constraining model over fixing data

- **`max_search_nodes`** (int, default=None): Maximum nodes to explore
  - `None`: Unlimited (searches until solution or exhaustion)
  - Positive integer: Stops after N nodes explored
  - Use to limit computation time

- **`seed`** (int, default=42): Random seed for reproducibility

#### Cross-Validation Parameters

- **`problems`** (List[str]): List of problem names to process
- **`num_steps`** (int): Number of steps per trajectory
- **`experiment_name`** (str): Name prefix for experiment directory

**Note**: Trajectory generation runs sequentially (one problem at a time) to ensure stability and easier debugging.

## Output Files

The simulator produces the same directory structure as the regular simulator, but with enriched reports:

```
noisy_conflict_experiments/
└── noisy_conflict_cv__steps=25__[timestamp]/
    ├── blocks.pddl                    # Domain file
    ├── problem1.pddl                  # Problem files
    ├── problem1.trajectory            # Trajectory files
    ├── problem1.masking_info          # Masking info
    ├── fold_0/                        # Cross-validation folds
    │   ├── learned_domain.pddl
    │   └── train_set/
    └── results_directory/
        ├── sam_learning_blocks_combined_semantic_performance.csv  # ENRICHED
        └── ... (other result files)
```

## Enriched Learning Report

The learning report now includes additional fields:

### Solution Statistics
- `solution_found`: "True" or "False"
- `final_conflicts`: Number of unresolved conflicts
- `solution_cost`: Total cost (model + fluent patches)

### Patch Counts
- `total_model_constraints`: Final number of model constraints
- `total_fluent_patches`: Final number of fluent patches
- `model_patches_added`: Number of model patches added
- `model_patches_removed`: Number of model patches removed
- `model_patches_changed`: Number of model patches changed
- `fluent_patches_added`: Number of fluent patches added
- `fluent_patches_removed`: Number of fluent patches removed

### Patch Details (semicolon-separated lists)
- `model_patches_added_detail`: "FORBID holding(?x) in eff of pick-up; ..."
- `model_patches_removed_detail`: "REQUIRE ontable(?x) in eff of put-down; ..."
- `fluent_patches_added_detail`: "(ontable e) at obs[0][2].next; ..."
- `fluent_patches_removed_detail`: "(holding a) at obs[0][1].prev; ..."

## Example Output

```csv
fold,problem,solution_found,final_conflicts,solution_cost,model_patches_added,fluent_patches_added,...
1,problem1,True,0,3,2,1,...
1,problem2,True,0,0,0,0,...
2,problem1,True,0,5,3,2,...
```

## Running the Example

Run the built-in example:

```bash
cd /Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL
source venv11/bin/activate
python src/noisy_conflict_search_simulator.py
```

This will:
1. Generate trajectories for problems 1-5 (in parallel)
2. Run 5-fold cross-validation with conflict search
3. Save enriched results to `noisy_conflict_experiments/`

## Comparison with Regular Simulator

### When to use Regular Simulator
- Data is clean and well-formed
- No conflicts expected
- Fast learning without search needed

### When to use Noisy Conflict Search Simulator
- Data may contain noise or errors
- Want to detect and quantify conflicts
- Need to understand what corrections are required
- Interested in minimal patch sets for consistency

## Performance Considerations

1. **Search Nodes**: Limit with `max_search_nodes` to control runtime
   - Start with 100-500 nodes for quick experiments
   - Use unlimited for thorough search

2. **Patch Costs**: Balance data vs. model corrections
   - Equal costs (1:1) - no preference
   - Higher fluent cost - prefer model constraints
   - Higher model cost - prefer data corrections

3. **Sequential Processing**: Trajectory generation runs one problem at a time
   - More stable than parallel processing
   - Easier to debug when issues occur
   - Progress is clearly visible
   - Total time scales linearly with number of problems

## Integration with Existing Code

The noisy conflict search simulator is fully compatible with existing experiment infrastructure:

- Uses same `OfflineBasicExperimentRunner` base class
- Produces same CSV format (with additional columns)
- Works with existing validation and statistics tools
- Can be used as drop-in replacement for `Simulator`

## Troubleshooting

### High Number of Conflicts
- Increase `max_search_nodes` to allow more exploration
- Adjust patch costs to prefer different resolution strategies
- Check masking info for overly aggressive masking

### Slow Performance
- Reduce `max_search_nodes` for faster (approximate) solutions
- Use fewer problems or shorter trajectories for testing
- Consider reducing `num_steps` for initial experiments

### Missing Patch Details
- Ensure ConflictDrivenPatchSearch returns enriched reports
- Check that `patch_diff` is present in report
- Verify CSV export includes new columns

## Future Enhancements

Potential improvements:
- [ ] Add search tree logging option for cross-validation
- [ ] Support for initial patches from prior knowledge
- [ ] Adaptive cost adjustment during search
- [ ] Conflict type statistics in reports
- [ ] Visualization of patch patterns across folds
