# DataRepairer Updates

## Overview

The `DataRepairer` class has been updated to align with the new modular detector architecture and to provide clearer, more maintainable code with comprehensive documentation.

## Key Changes

### 1. Terminology Updates

**Before:**
- Used `Inconsistency` and `DeterminismViolation` interchangeably
- Method names like `repair_inconsistency()`

**After:**
- Consistently uses `EffectsViolation` (from the new modular `EffectsDetector`)
- Method renamed to `repair_violation()` for clarity
- All parameter names updated from `inconsistency` to `violation`

### 2. Code Organization

The class is now organized into clear sections:

```python
# ==================== Image Processing ====================
# Base64 encoding utilities

# ==================== LLM Prompting ====================
# Prompt generation and LLM verification

# ==================== Repair Decision Logic ====================
# Logic for determining which transition to repair

# ==================== State Repair ====================
# Low-level state modification

# ==================== High-Level Interface ====================
# Main public API
```

### 3. Documentation Improvements

**Module-Level Documentation:**
- Added comprehensive module docstring explaining the overall purpose
- Describes the repair approach and assumptions

**Class-Level Documentation:**
- Detailed explanation of how the class works
- Clear description of the three-step process

**Method Documentation:**
- Every method now has a detailed docstring with:
  - Clear purpose statement
  - Args section with parameter descriptions
  - Returns section with return value descriptions
  - Notes section where appropriate

**Inline Comments:**
- Step-by-step comments throughout complex methods
- Explanatory comments for key decision points

### 4. Code Flow Improvements

**Before:**
```python
def determine_repair_choice(self, inconsistency, ...):
    llm_result = self.verify_fluent_with_llm(...)
    if llm_result == "IMAGE1":
        # Complex logic without explanations
        ...
```

**After:**
```python
def determine_repair_choice(self, violation, ...):
    # Step 1: Query the LLM about fluent presence
    llm_result = self.verify_fluent_with_llm(...)

    # Step 2: Interpret LLM result and determine repair
    # The logic: if LLM says fluent is in IMAGE1, then transition1's
    # classification is correct, so we repair transition2 to match it.

    if llm_result == "IMAGE1":
        # Ground truth: fluent is present in IMAGE1
        # Therefore, transition2 should also have the fluent
        ...
```

### 5. Method Improvements

#### `_repair_state()` Method

Enhanced with detailed explanation of State structure:

```python
"""
Repair a state by modifying a fluent's truth value.

This modifies the State object in-place by:
1. Finding the GroundedPredicate matching fluent_str
2. Removing the old predicate
3. Adding a corrected predicate with the right is_positive value

Note:
    The state's internal structure organizes predicates by name in
    state.state_predicates[predicate_name], so we must:
    - Extract the predicate name from fluent_str
    - Find the matching GroundedPredicate in that set
    - Replace it with a corrected version
"""
```

#### `determine_repair_choice()` Method

Added clear explanation of the repair logic:

```python
# The logic: if LLM says fluent is in IMAGE1, then transition1's
# classification is correct, so we repair transition2 to match it.

if llm_result == "IMAGE1":
    # Ground truth: fluent is present in IMAGE1
    # Therefore, transition2 should also have the fluent
    if violation.fluent_in_trans1_next and not violation.fluent_in_trans2_next:
        # Trans1 correctly has it, trans2 is missing it
        return RepairChoice.SECOND, True
    ...
```

### 6. Integration with Modular Architecture

**Imports:**
```python
from src.plan_denoising.detectors.effects_detector import EffectsViolation
from src.plan_denoising.conflict_tree import RepairOperation, RepairChoice
```

**Type Hints:**
- All methods properly typed with `EffectsViolation`
- Clear separation between violation detection and repair

**Workflow:**
1. `EffectsDetector` detects violations in trajectory
2. `DataRepairer` uses LLM to determine correct state
3. `ConflictTree` tracks repair decisions for backtracking

## Usage Pattern

### Basic Repair

```python
# Detect violations
detector = InconsistencyDetector(domain)
violations = detector.detect_effects_violations_in_trajectory(trajectory_path)

# Initialize repairer
repairer = DataRepairer(openai_apikey="your-key")

# Repair a violation
observation, repair_op, repair_choice = repairer.repair_violation(
    observation=observation,
    violation=violations[0],
    image1_path=Path("state_1.png"),
    image2_path=Path("state_2.png"),
    domain_name="blocks"
)
```

### Step-by-Step Control

```python
# Step 1: LLM verification only
llm_result = repairer.verify_fluent_with_llm(
    image1_path, image2_path, fluent, domain_name
)

# Step 2: Determine repair strategy
repair_choice, should_be_present = repairer.determine_repair_choice(
    violation, image1_path, image2_path, domain_name
)

# Step 3: Apply repair
observation, repair_op = repairer.repair_observation(
    observation, violation, repair_choice, should_be_present
)
```

## Benefits

1. **Clarity**: Every step is clearly documented and explained
2. **Maintainability**: Code organization makes it easy to find and modify specific functionality
3. **Extensibility**: Clear structure makes it easy to add new repair strategies
4. **Correctness**: Better alignment with the modular detector architecture
5. **Debuggability**: Detailed logging and clear variable names aid debugging

## Related Files

- `src/plan_denoising/data_repairer.py` - Main implementation
- `src/plan_denoising/example_data_repairer_usage.py` - Usage examples
- `src/plan_denoising/detectors/effects_detector.py` - Violation detection
- `src/plan_denoising/conflict_tree.py` - Repair tracking
