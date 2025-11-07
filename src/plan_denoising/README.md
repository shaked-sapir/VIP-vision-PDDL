# Plan Denoising Module

This module provides tools for detecting and repairing inconsistencies in vision-based PDDL trajectories.

## Overview

When learning action models from vision-based observations, errors in fluent classification can lead to inconsistencies in the trajectory. An **inconsistency** occurs when:

1. Two transitions have the same action
2. Both transitions start from the same state (same fluents)
3. The transitions lead to different next states

This violates determinism: executing the same action in the same state should always produce the same result.

## Components

### 1. InconsistencyDetector

Detects inconsistencies in PDDL trajectory files.

```python
from src.plan_denoising import InconsistencyDetector

detector = InconsistencyDetector()
inconsistencies = detector.detect_inconsistencies_in_trajectory("trajectory.trajectory")

# Print results
detector.print_inconsistencies(inconsistencies)
```

**How it works:**
- Parses a `.trajectory` file into transitions
- Groups transitions by (action, prev_state)
- Identifies groups where next_states differ

### 2. DataRepairer

Uses LLM (GPT-4 Vision) to determine which state classification was incorrect.

```python
from src.plan_denoising import DataRepairer

repairer = DataRepairer(openai_apikey="your-key")

# Repair an inconsistency
repaired_obs, repair_op, repair_choice = repairer.repair_inconsistency(
    observation,
    inconsistency,
    image1_path="state_0006.png",
    image2_path="state_0013.png",
    domain_name="blocks"
)
```

**How it works:**
- Takes images of both conflicting next_states
- Asks GPT-4 Vision which image contains the conflicting fluent
- Fixes the state with the incorrect classification

### 3. ConflictTree

Tracks repair decisions in a tree structure for backtracking.

```python
from src.plan_denoising import ConflictTree, RepairChoice

tree = ConflictTree()

# Add a repair decision
node = tree.add_repair(inconsistency, repair_operation, RepairChoice.FIRST)

# Backtrack if needed
tree.backtrack()

# Try alternative repair
if tree.has_unexplored_alternative():
    alt_choice = tree.get_alternative_repair_choice()
```

**Features:**
- Tracks sequence of repair decisions
- Enables backtracking when a repair path fails
- Supports trying both repair choices for each inconsistency

### 4. PlanDenoiser

Main orchestrator that runs the complete denoising loop.

```python
from src.plan_denoising import PlanDenoiser

denoiser = PlanDenoiser(
    domain=domain,
    openai_apikey="your-key",
    image_directory=Path("./images"),
    domain_name="blocks",
    max_iterations=50,
    max_backtracks=5
)

# Denoise a trajectory
denoised_obs, learned_domain, conflict_tree = denoiser.denoise_from_trajectory_file(
    Path("trajectory.trajectory"),
    use_llm_verification=True
)
```

## Algorithm

The denoising algorithm follows this loop:

```
WHILE inconsistencies exist AND iterations < max_iterations:
    1. Detect inconsistencies in current observation
    2. If no inconsistencies: DONE

    3. Select first inconsistency
    4. Use LLM to determine which transition to repair
    5. Repair the observation
    6. Add repair to conflict tree

    7. Run PI-SAM on repaired observation
    8. Validate learned model against trajectory

    9. IF model is consistent:
          Continue to step 1 (next inconsistency)
       ELSE (model inconsistent):
          IF backtracks < max_backtracks AND alternative exists:
              Backtrack in tree
              Try alternative repair (fix the other transition)
          ELSE:
              FAIL (no more alternatives)

RETURN denoised observation, learned domain, conflict tree
```

## Data Structures

### Inconsistency

```python
@dataclass
class Inconsistency:
    transition1_index: int          # Index of first transition
    transition2_index: int          # Index of second transition
    action_name: str                # Action in both transitions
    conflicting_fluent: str         # Fluent that differs
    fluent_in_trans1_next: bool     # Is fluent in trans1's next_state?
    fluent_in_trans2_next: bool     # Is fluent in trans2's next_state?
```

### RepairOperation

```python
@dataclass
class RepairOperation:
    transition_index: int    # Which transition was repaired
    state_type: str         # 'prev_state' or 'next_state'
    fluent_changed: str     # Which fluent was modified
    old_value: bool         # Was fluent present before?
    new_value: bool         # Is fluent present after?
```

### ConflictNode

Represents a decision point in the repair process.

```python
class ConflictNode:
    inconsistency: Inconsistency       # The inconsistency being resolved
    repair_operation: RepairOperation  # The repair performed
    repair_choice: RepairChoice        # Which transition was repaired (FIRST/SECOND)
    parent: Optional[ConflictNode]     # Parent node
    children: List[ConflictNode]       # Child nodes (subsequent decisions)
    pi_sam_result: Optional[Any]       # Result of PI-SAM after this repair
```

## Usage Examples

### Example 1: Detect Inconsistencies Only

```python
from pathlib import Path
from src.plan_denoising import InconsistencyDetector

detector = InconsistencyDetector()
inconsistencies = detector.detect_inconsistencies_in_trajectory(
    Path("./trajectories/problem1.trajectory")
)

print(f"Found {len(inconsistencies)} inconsistencies")
for incons in inconsistencies:
    print(f"  - {incons}")
```

### Example 2: Full Denoising Pipeline

```python
from pathlib import Path
from pddl_plus_parser.lisp_parsers import DomainParser
from src.plan_denoising import PlanDenoiser

# Load domain
domain = DomainParser("domain.pddl", partial_parsing=True).parse_domain()

# Create denoiser
denoiser = PlanDenoiser(
    domain=domain,
    openai_apikey="sk-your-openai-key",
    image_directory=Path("./images"),
    domain_name="blocks"
)

# Denoise
denoised_obs, learned_domain, tree = denoiser.denoise_from_trajectory_file(
    Path("./trajectories/problem1.trajectory")
)

# Results
print(f"Repaired {len(tree.get_current_repairs())} inconsistencies")
print(f"Final learned domain: {learned_domain}")
```

### Example 3: Manual Repair with Custom Logic

```python
from src.plan_denoising import DataRepairer, InconsistencyDetector
from src.plan_denoising.conflict_tree import RepairChoice

# Detect inconsistencies
detector = InconsistencyDetector()
observation = detector.load_trajectory("trajectory.trajectory")
inconsistencies = detector.detect_inconsistencies_from_observation(observation)

# Manually repair using custom logic
repairer = DataRepairer(openai_apikey="your-key")

for incons in inconsistencies:
    # Custom logic to choose which transition to repair
    if incons.transition1_index < incons.transition2_index:
        repair_choice = RepairChoice.FIRST
    else:
        repair_choice = RepairChoice.SECOND

    # Determine correct fluent value (without LLM)
    fluent_should_be_present = incons.fluent_in_trans2_next

    # Repair
    observation, repair_op = repairer.repair_observation(
        observation, incons, repair_choice, fluent_should_be_present
    )
    print(f"Repaired: {repair_op}")
```

## Integration with PI-SAM

The denoiser is designed to work with the PI-SAM learning framework:

```python
from src.pi_sam import PISAMLearner
from src.plan_denoising import PlanDenoiser

# After denoising
denoised_obs, learned_domain, tree = denoiser.denoise(observation)

# The learned_domain is already a PI-SAM result
# You can use it directly for planning or further analysis

# Or manually run PI-SAM again
learner = PISAMLearner(domain)
final_domain, report = learner.learn_action_model([denoised_obs])
```

## Configuration

Key parameters for `PlanDenoiser`:

- **max_iterations**: Maximum denoising iterations (default: 100)
- **max_backtracks**: Maximum backtracking attempts (default: 10)
- **use_llm_verification**: Whether to use LLM for repair choice (default: True)
- **domain_name**: Domain name for LLM prompts (e.g., "blocks", "hanoi")

## Troubleshooting

**Issue**: Detector finds no inconsistencies
- **Solution**: Verify trajectory file format and parsing

**Issue**: LLM returns "NEITHER" for all verifications
- **Solution**: Check image quality, improve prompts in `DataRepairer`

**Issue**: Max iterations reached without convergence
- **Solution**: Increase `max_iterations` or `max_backtracks`

**Issue**: PI-SAM fails to learn valid model
- **Solution**: Check if repaired trajectory is truly consistent

## Future Enhancements

Potential improvements for the plan denoising system:

1. **Batch repair**: Repair multiple related inconsistencies simultaneously
2. **Confidence scoring**: Use LLM confidence to prioritize repairs
3. **Incremental PI-SAM**: Update model incrementally instead of re-learning
4. **Multi-modal verification**: Combine multiple images or sensors
5. **Caching**: Cache LLM responses for repeated verifications
6. **Parallel repair exploration**: Try multiple repair paths in parallel

## References

- **PI-SAM**: Partial Information SAM learning algorithm
- **GPT-4 Vision**: OpenAI's multimodal language model for image understanding
- **PDDL**: Planning Domain Definition Language
