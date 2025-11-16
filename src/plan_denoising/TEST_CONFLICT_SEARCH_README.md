# Conflict-Driven Patch Search Test Suite

## Overview

The `test_conflict_search.py` file provides a comprehensive test suite for the `ConflictDrivenPatchSearch` algorithm, demonstrating the complete workflow from basic learning to conflict-driven search with visualization.

## Test Structure

### Test 1: Regular PISAM Learning
**Purpose**: Baseline learning with standard PI-SAM algorithm

**What it does**:
- Creates a regular `PISAMLearner` instance
- Learns action model from problem7 trajectory with masking
- Displays comprehensive learning report
- Shows learned actions with preconditions and effects count

**Output**:
```
PISAM Learning Report:
  learning_time: <time>
  observed_actions: <count>
  ...

Learned Actions:
  Action: pick-up
    Preconditions: X literals
    Effects: Y effects
  ...
```

### Test 2: SimpleNoisyPisamLearner (No Patches)
**Purpose**: Verify that SimpleNoisyPisamLearner behaves identically to PISAM when no patches are applied

**What it does**:
- Creates a `SimpleNoisyPisamLearner` instance
- Learns with empty patch sets (no fluent patches, no model patches)
- Displays learning report and conflict count
- Shows learned actions

**Expected**:
- Zero conflicts (should behave like regular PISAM)
- Same results as Test 1

### Test 3: Compare PISAM and SimpleNoisyPisamLearner
**Purpose**: Verify equivalence of the two learners

**What it does**:
- Compares action sets from Test 1 and Test 2
- Performs action-by-action comparison
- Checks precondition and effect counts for each action

**Output**:
```
Comparison Results:
  Actions in PISAM: X
  Actions in SimpleNoisy: X
  ✓ Action sets are EQUAL

Action-by-Action Comparison:
  ✓ pick-up:
      Preconditions: PISAM=3, SimpleNoisy=3
      Effects: PISAM=2, SimpleNoisy=2
  ...
```

### Test 4: ConflictDrivenPatchSearch with Tree Visualization
**Purpose**: Run the full conflict-driven search and generate visualization

**What it does**:
1. Creates `ConflictDrivenPatchSearch` instance
2. Runs search with **no max_nodes limit** (searches until solution or exhaustion)
3. Logs every node expansion to `ConflictTreeLogger`
4. Displays comprehensive search results:
   - Number of nodes expanded
   - Whether solution was found
   - Model constraints applied
   - Fluent patches applied
   - Remaining conflicts (if any)
5. Generates detailed conflict-tree file

**Output Format**:
```
Search Results:
  Nodes Expanded: N
  Solution Found: True/False
  Final Conflicts: X
  Model Constraints: Y
  Fluent Patches: Z

Model Constraints Applied:
  FORBID holding(?x) in eff of pick-up
  REQUIRE ontable(?x) in eff of put-down
  ...

Fluent Patches Applied:
  Flip '(on a b)' at obs[0][2].next
  Flip '(clear c)' at obs[0][5].prev
  ...

Conflict Tree saved to: .../conflict_search_tree.txt
```

## Conflict Tree Visualization File

The generated `conflict_search_tree.txt` file contains a complete traversal of the search tree with the following information for each node:

### Node Information
```
────────────────────────────────────────────────────────────────────────────────────────────────────
Node #1 (Depth=0)
────────────────────────────────────────────────────────────────────────────────────────────────────
Cost: 0
Patches: 0 model + 0 fluent
Conflicts: 5
Status: ✗ Has Conflicts

Model Constraints:
  (none yet)

Fluent Patches:
  (none yet)

Conflicts:
  1. forbid_effect_vs_must
     Action: pick-up
     PBL: holding(?x)
     Grounded: holding(a)
     Location: obs[0][3]
  2. require_effect_vs_cannot
     Action: put-down
     PBL: ontable(?x)
     Grounded: ontable(b)
     Location: obs[0][7]
  ...
```

### When Solution is Found
```
────────────────────────────────────────────────────────────────────────────────────────────────────
Node #42 (Depth=8)
────────────────────────────────────────────────────────────────────────────────────────────────────
Cost: 8
Patches: 3 model + 5 fluent
Conflicts: 0
Status: ✓ SOLUTION

Model Constraints:
  - FORBID holding(?x) in eff of pick-up
  - REQUIRE clear(?x) in pre of stack
  - FORBID ontable(?y) in eff of unstack

Fluent Patches:
  - Flip '(on a b)' at obs[0][2].next
  - Flip '(clear c)' at obs[0][5].prev
  - Flip '(not (holding d))' at obs[0][8].next
  ...

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
SOLUTION FOUND!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

## Running the Tests

### Run all tests in sequence:
```bash
cd /Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL
python -m pytest src/plan_denoising/test_conflict_search.py -v
```

Or using unittest:
```bash
python src/plan_denoising/test_conflict_search.py
```

### Run a specific test:
```bash
python src/plan_denoising/test_conflict_search.py TestConflictDrivenPatchSearch.test_4_conflict_search_with_tree_visualization
```

Or using the helper function:
```python
from src.plan_denoising.test_conflict_search import run_single_test
run_single_test('test_4_conflict_search_with_tree_visualization')
```

## Output Files

After running the tests, the following file will be generated:

**Location**: `src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/conflict_search_tree.txt`

This file contains:
- Complete search tree traversal
- Every node expanded during search
- All conflicts encountered
- All patches applied at each node
- Solution path (if found)

## Understanding the Results

### Successful Search
If the search finds a conflict-free model:
- `Solution Found: True`
- `Final Conflicts: 0`
- The tree file will show the solution node marked with stars
- Model constraints and fluent patches represent the minimal fixes needed

### Unsuccessful Search
If no conflict-free model is found:
- `Solution Found: False`
- `Final Conflicts: X` (X > 0)
- The tree file shows all explored nodes
- The last node represents the best attempt

## Key Metrics

1. **Nodes Expanded**: Number of search states explored
   - Lower is better (more efficient search)
   - Indicates search complexity

2. **Cost**: Total number of patches (model + fluent)
   - Lower is better (fewer changes to data/model)
   - Search prioritizes low-cost solutions

3. **Depth**: How many branching decisions were made
   - Indicates solution path length
   - Deeper solutions required more conflict resolutions

## Example Interpretation

If the search expands 50 nodes and finds a solution with:
- 3 model constraints
- 5 fluent patches
- Total cost: 8

This means:
- The algorithm explored 50 different patch combinations
- The optimal solution requires 8 total changes
- 3 changes constrain the learned model
- 5 changes flip observations in the data
- This is the minimal set of changes that produces a conflict-free learned model

## Architecture

### ConflictTreeLogger
A simple logger class that tracks and logs all node expansions during search:
- Node ID, depth, cost
- Patches applied
- Conflicts detected
- Solution status
- Implements `log_node()` method called by `ConflictDrivenPatchSearch`

### ConflictDrivenPatchSearch Integration
The `ConflictDrivenPatchSearch` class accepts an optional `logger` parameter:
- When provided, the search calls `logger.log_node()` after each expansion
- Maintains depth tracking internally
- No wrapper or reimplementation needed - uses the actual search algorithm
- Logger is completely optional (pass `None` for no logging)

**Usage Example**:
```python
# Create logger
tree_logger = ConflictTreeLogger()

# Create search with logger
search = ConflictDrivenPatchSearch(
    partial_domain_template=domain,
    negative_preconditions_policy=NegativePreconditionPolicy.hard,
    seed=42,
    logger=tree_logger,  # Optional parameter
)

# Run search - logger is automatically called for each node
learned_domain, conflicts, model_constraints, fluent_patches, report = search.run(
    observations=[masked_observation],
    max_nodes=None
)

# Save tree visualization
tree_logger.save_to_file(Path("conflict_search_tree.txt"))
```

## Test Data

All tests use:
- **Domain**: blocks world (src/domains/blocks/blocks.pddl)
- **Problem**: problem7
- **Trajectory**: 25 steps from LLM experiment
- **Masking**: Partial observability masks
- **Location**: src/experiments/llm_cv_test__PDDLEnvBlocks-v0__gpt-5-mini__steps=25__03-11-2025T22:25:06/
