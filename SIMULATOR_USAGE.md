# VIP-vision-PDDL Simulator Usage Guide

## Overview

The VIP-vision-PDDL simulator is a comprehensive tool for learning PDDL action models from visual observations using the PI-SAM algorithm. It supports multiple masking strategies, cross-validation experiments, and flexible configuration.

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

1. Copy the example configuration file:
```bash
cp config.example.yaml config.yaml
```

2. Edit `config.yaml` and add your OpenAI API key:
```yaml
openai:
  api_key: "sk-your-actual-api-key-here"
```

**Note:** The `config.yaml` file is gitignored to protect your API keys.

---

## Usage

The simulator supports two modes:

### Simple Mode

Run the pipeline for a single problem (for testing and quick experiments):

```bash
python -m src.simulator simple --problem problem1 --steps 25 --masking percentage --ratio 0.8
```

**Arguments:**
- `--problem`: Problem name (e.g., `problem1`, `problem3`)
- `--steps`: Number of trajectory steps (default: from config)
- `--masking`: Masking strategy (`percentage` or `random`)
- `--ratio`: Masking ratio for percentage strategy (0.0-1.0)
- `--probability`: Masking probability for random strategy (0.0-1.0)

### Full Mode

Run the complete pipeline with cross-validation for multiple problems:

```bash
python -m src.simulator full --domain blocks --problems problem1 problem3 problem5 --steps 50 --masking random --probability 0.3
```

**Arguments:**
- `--domain`: Domain to use (`blocks` or `hanoi`, default: `blocks`)
- `--problems`: List of problem names
- `--steps`: Number of steps per trajectory (default: from config)
- `--masking`: Masking strategy (`percentage` or `random`)
- `--ratio`: Masking ratio for percentage strategy
- `--probability`: Masking probability for random strategy

---

## Masking Strategies

### 1. Percentage Masking

Masks a fixed percentage of predicates in each state.

```bash
# Mask 80% of predicates
python -m src.simulator simple --problem problem1 --masking percentage --ratio 0.8

# Mask 30% of predicates
python -m src.simulator full --problems problem1 problem3 --masking percentage --ratio 0.3
```

**Parameters:**
- `--ratio`: Percentage of predicates to mask (0.0 = no masking, 1.0 = mask all)

### 2. Random Masking

Each predicate has an independent probability of being masked.

```bash
# 30% probability per predicate
python -m src.simulator simple --problem problem1 --masking random --probability 0.3

# 50% probability per predicate
python -m src.simulator full --problems problem1 problem3 --masking random --probability 0.5
```

**Parameters:**
- `--probability`: Probability that each predicate is masked (0.0-1.0)

---

## Configuration File

The `config.yaml` file controls default settings:

```yaml
# OpenAI API Configuration
openai:
  api_key: "your-api-key-here"

# Default Paths
paths:
  domains_dir: "src/domains"
  experiments_dir: "experiments"
  
# Domain-specific configurations
domains:
  blocks:
    domain_name: "PDDLEnvBlocks-v0"
    domain_file: "src/domains/blocks/blocks.pddl"
    problems_dir: "src/domains/blocks/problems"
    problem_prefix: "problem"
  
  hanoi:
    domain_name: "PDDLEnvHanoi-v0"
    domain_file: "src/domains/hanoi/hanoi.pddl"
    problems_dir: "src/domains/hanoi/problems"
    problem_prefix: "problem"

# Masking Strategy Defaults
masking:
  default_strategy: "percentage"
  percentage:
    default_ratio: 0.8
  random:
    default_probability: 0.3

# Trajectory Generation Defaults
trajectory:
  default_num_steps: 25
  default_seed: 42

# Experiment Settings
experiment:
  cross_validation_folds: 5
  default_negative_precondition_policy: "hard"
```

---

## Examples

### Example 1: Running Cross-Validation Directly from simulator.py

The simplest way to run cross-validation is to execute the simulator.py file directly:

```bash
# Run cross-validation with LLM-based masking
python -m src.simulator
```

This will:
1. Load configuration from `config.yaml`
2. Run cross-validation on problems [problem1, problem3, problem5]
3. Use LLM-based object detection and fluent classification
4. Save results to the experiments directory

To customize:
1. Open `src/simulator.py`
2. Modify the configuration variables in the `if RUN_CROSS_VALIDATION:` block:
   - `problems`: List of problem names (e.g., `["problem1", "problem3", "problem5", "problem7", "problem9"]`)
   - `num_steps`: Number of trajectory steps (e.g., `25`, `50`)
   - `experiment_name`: Name for the experiment (e.g., `"llm_cv_test"`)

To run the simple single-problem example instead:
1. Set `RUN_SIMPLE_EXAMPLE = True` and `RUN_CROSS_VALIDATION = False`
2. Run `python -m src.simulator`

### Example 2: Quick Test with Blocks Domain (CLI)

```bash
# Run simple pipeline with percentage masking (deterministic detection)
python -m src.simulator_cli simple \
  --problem problem1 \
  --steps 10 \
  --masking percentage \
  --ratio 0.5

# Run simple pipeline with LLM-based detection and masking
python -m src.simulator_cli simple \
  --problem problem1 \
  --steps 10 \
  --masking llm
```

### Example 3: Full Cross-Validation Experiment (CLI)

```bash
# Run full pipeline with deterministic detection and random masking
python -m src.simulator_cli full \
  --domain blocks \
  --problems problem1 problem3 problem5 problem7 problem9 \
  --steps 50 \
  --masking random \
  --probability 0.3

# Run full pipeline with LLM-based detection and masking
python -m src.simulator_cli full \
  --domain blocks \
  --problems problem1 problem3 problem5 \
  --steps 25 \
  --masking llm
```

### Example 4: Using Custom Config File

```bash
# Create custom config
cp config.yaml my_experiment_config.yaml

# Edit my_experiment_config.yaml with custom settings

# Run with custom config
python -m src.simulator_cli full \
  --config my_experiment_config.yaml \
  --problems problem1 problem3
```

### Example 5: Custom Experiment Directory

```bash
# Save experiments to specific directory
python -m src.simulator_cli full \
  --experiment-dir /path/to/my/experiments \
  --problems problem1 problem3 problem5 \
  --masking llm
```

---

## Output Structure

### Simple Mode Output

```
experiments/
└── PDDLEnvBlocks-v0_problem1_steps=25___<timestamp>/
    ├── problem1.pddl
    ├── problem1.trajectory
    ├── problem1.masking_info
    └── problem1_temp_output/
        └── images/
            ├── state_0000.png
            ├── state_0001.png
            └── ...
```

### Full Mode Output

```
experiments/
└── pisam_cv_percentage__steps=50__<timestamp>/
    ├── blocks.pddl
    ├── problem1.pddl
    ├── problem1.trajectory
    ├── problem1.masking_info
    ├── problem3.pddl
    ├── problem3.trajectory
    ├── problem3.masking_info
    └── results_directory/
        ├── sam_learning_blocks_combined_semantic_performance.csv
        ├── fold_1/
        ├── fold_2/
        └── ...
```

---

## Programmatic Usage

You can also use the simulator programmatically in your own Python scripts:

```python
from pathlib import Path
from src.simulator import (
    Simulator,
    mask_observation_with_percentage,
    mask_observation_with_random,
    run_full_pipeline_with_cross_validation,
    load_config
)

# Load configuration
config = load_config()

# Create simulator
simulator = Simulator(
    domain_name="PDDLEnvBlocks-v0",
    openai_apikey=config['openai']['api_key'],
    pddl_domain_file=Path("src/domains/blocks/blocks.pddl"),
    pddl_problem_dir=Path("src/domains/blocks/problems"),
    experiment_dir_path=Path("experiments")
)

# Run simple pipeline
results = simulator.run_simple_pipeline(
    problem_name="problem1",
    num_steps=25,
    masking_strategy="percentage",
    masking_ratio=0.8
)

# Access results
learned_domain = results['learnt_domain']
learning_report = results['learning_report']
```

---

## Masking Strategy Functions

### Percentage Masking

```python
from src.simulator import mask_observation_with_percentage

masked_obs, masking_info = mask_observation_with_percentage(
    domain=domain,
    observation=observation,
    masking_ratio=0.8
)
```

### Random Masking

```python
from src.simulator import mask_observation_with_random

masked_obs, masking_info = mask_observation_with_random(
    domain=domain,
    observation=observation,
    masking_probability=0.3
)
```

### Strategy Selection

```python
from src.simulator import mask_observation_with_strategy

# Automatically selects the right function
masked_obs, masking_info = mask_observation_with_strategy(
    domain=domain,
    observation=observation,
    strategy="percentage",  # or "random"
    masking_ratio=0.8      # strategy-specific parameters
)
```

---

## Full Pipeline with Cross-Validation

```python
from src.simulator import run_full_pipeline_with_cross_validation

working_dir = run_full_pipeline_with_cross_validation(
    domain_name="PDDLEnvBlocks-v0",
    openai_apikey=api_key,
    pddl_domain_file=Path("src/domains/blocks/blocks.pddl"),
    pddl_problem_dir=Path("src/domains/blocks/problems"),
    experiment_dir=Path("experiments"),
    problems=["problem1", "problem3", "problem5"],
    num_steps=50,
    masking_strategy="percentage",
    masking_ratio=0.8
)

print(f"Results saved to: {working_dir}")
```

---

## Troubleshooting

### API Key Issues

**Error:** `ValueError: Please set your OpenAI API key in config.yaml`

**Solution:** 
1. Ensure you've copied `config.example.yaml` to `config.yaml`
2. Add your actual OpenAI API key to `config.yaml`

### Configuration File Not Found

**Error:** `FileNotFoundError: Configuration file not found`

**Solution:**
```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your settings
```

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'yaml'`

**Solution:**
```bash
pip install pyyaml
# Or install all requirements:
pip install -r requirements.txt
```

---

## Advanced Usage

### Custom Trajectory Handler

```python
from src.trajectory_handlers.llm_blocks_trajectory_handler import LLMBlocksImageTrajectoryHandler

# Create custom trajectory handler
trajectory_handler = LLMBlocksImageTrajectoryHandler(
    domain_name="PDDLEnvBlocks-v0",
    openai_apikey=api_key
)

# Use in simulator
simulator.image_trajectory_handler = trajectory_handler
```

### Custom Masking Info

```python
# Load existing masking info
from src.utils.masking import load_masking_info

masking_info = load_masking_info(
    experiment_path=Path("experiments/my_experiment"),
    domain=domain,
    problem_name="problem1"
)

# Apply custom masking
from src.utils.pddl import mask_observation

masked_obs = mask_observation(grounded_obs, masking_info)
```

---

## Support

For issues or questions:
1. Check the configuration in `config.yaml`
2. Verify all paths in the configuration are correct
3. Ensure API keys are valid
4. Check experiment output directories for logs

---

## Reference

### Command-Line Arguments

**Global:**
- `--config PATH` - Configuration file path
- `--domain {blocks,hanoi}` - Domain selection
- `--experiment-dir PATH` - Experiment output directory

**Simple Mode:**
- `--problem NAME` - Problem name (required)
- `--steps NUM` - Trajectory steps

**Full Mode:**
- `--problems NAME [NAME ...]` - Problem list (required)
- `--steps NUM` - Trajectory steps per problem

**Masking (both modes):**
- `--masking {percentage,random}` - Masking strategy
- `--ratio FLOAT` - Percentage masking ratio (0.0-1.0)
- `--probability FLOAT` - Random masking probability (0.0-1.0)
