# Quick Start: Running Cross-Validation with LLM

This guide shows you how to quickly run cross-validation experiments with LLM-based masking.

## Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up OpenAI API key:**
   ```bash
   # Copy the example config
   cp config.example.yaml config.yaml

   # Edit config.yaml and add your OpenAI API key
   # Replace "your-api-key-here" with your actual API key
   ```

## Method 1: Run Directly from simulator.py (Recommended for Testing)

This is the easiest way to run a cross-validation experiment:

```bash
python -m src.simulator
```

### What this does:
1. Loads configuration from `config.yaml`
2. Runs cross-validation on 3 problems: [problem1, problem3, problem5]
3. Uses LLM-based object detection and fluent classification (GPT-4 Vision)
4. Generates 25-step trajectories for each problem
5. Extracts masking info from unknown predicates identified by the LLM
6. Runs 5-fold cross-validation using PI-SAM learning
7. Saves all results to the `experiments/` directory

### Customizing the experiment:

Open `src/simulator.py` and modify these variables (around line 397-401):

```python
# Configuration for cross-validation experiment
domain = 'blocksworld'
problems = ["problem1", "problem3", "problem5"]  # Add more: problem7, problem9
num_steps = 25  # Increase for longer trajectories
experiment_name = "llm_cv_test"  # Change experiment name
```

Available problems in blocks domain:
- problem1
- problem3
- problem5
- problem7
- problem9

### Example configurations:

**Quick test (3 problems, 10 steps):**
```python
problems = ["problem1", "problem3", "problem5"]
num_steps = 10
experiment_name = "quick_test"
```

**Full experiment (5 problems, 50 steps):**
```python
problems = ["problem1", "problem3", "problem5", "problem7", "problem9"]
num_steps = 50
experiment_name = "full_llm_cv"
```

## Method 2: Use the CLI (For Advanced Usage)

For more control, use the command-line interface:

```bash
# LLM-based cross-validation
python -m src.simulator_cli full \
  --domain blocksworld \
  --problems problem1 problem3 problem5 \
  --steps 25 \
  --masking llm

# Deterministic detection with percentage masking
python -m src.simulator_cli full \
  --domain blocksworld \
  --problems problem1 problem3 problem5 \
  --steps 25 \
  --masking percentage \
  --ratio 0.8
```

## Understanding the Output

After running, you'll see output like:

```
================================================================================
CROSS-VALIDATION COMPLETE!
================================================================================
Total time: 450.32 seconds
Results directory: experiments/llm_cv_test__steps=25__20250129_143022

Results CSV: experiments/llm_cv_test__steps=25__20250129_143022/results_directory/sam_learning_blocks_combined_semantic_performance.csv
```

### Output structure:

```
experiments/llm_cv_test__steps=25__20250129_143022/
├── blocks.pddl                           # Domain file
├── problem1.pddl                         # Problem files
├── problem1.trajectory                   # Generated trajectories
├── problem1.masking_info                 # Masking information
├── PDDLEnvBlocks-v0_problem1_steps=25... # Individual problem outputs
│   ├── problem1_temp_output/
│   │   └── images/
│   │       ├── state_0000.png           # State images
│   │       ├── state_0001.png
│   │       └── ...
├── results_directory/
│   ├── sam_learning_blocks_combined_semantic_performance.csv  # Results!
│   ├── fold_1/                          # Cross-validation folds
│   ├── fold_2/
│   └── ...
```

### Analyzing results:

The main results are in:
```
results_directory/sam_learning_blocks_combined_semantic_performance.csv
```

This CSV contains:
- Precision, recall, F1 scores for learned action models
- Performance metrics for each fold
- Aggregated statistics

Each fold directory contains:
- Learned domain files
- Training/test splits
- Detailed learning reports

## Switching Between Simple and Cross-Validation

In `src/simulator.py`, toggle these flags:

```python
# Choose which example to run
RUN_SIMPLE_EXAMPLE = False   # Set to True for single-problem test
RUN_CROSS_VALIDATION = True  # Set to True for cross-validation
```

You can run both by setting both to `True`.

## Troubleshooting

### Error: "Please set your OpenAI API key in config.yaml"
- Make sure you've copied `config.example.yaml` to `config.yaml`
- Add your actual OpenAI API key (starts with "sk-...")

### Error: "Configuration file not found"
```bash
cp config.example.yaml config.yaml
```

### Experiment runs but no results CSV
- Check the `results_directory/` folder in your experiment output
- Look for partial results or error logs in individual fold directories

### Out of memory
- Reduce `num_steps` (try 10 or 15 instead of 25)
- Reduce number of problems in cross-validation

### OpenAI API rate limits
- Add delays between problems (modify the code)
- Use fewer problems or shorter trajectories
- Check your OpenAI account usage limits

## Next Steps

1. **Analyze results:** Open the CSV file in Excel or pandas
2. **Compare methods:** Run with different masking strategies (percentage, random, llm)
3. **Visualize:** Check the generated state images in the `images/` directories
4. **Scale up:** Add more problems or increase trajectory length

For more detailed documentation, see `SIMULATOR_USAGE.md`.
