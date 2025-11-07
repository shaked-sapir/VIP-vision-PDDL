# Relevance Evaluation: Comparing Prompt Variants

This module compares LLM performance with and without an "uncertain" option for relevance judgments.

## Overview

When using LLMs to classify predicates from images, we can allow the model to express uncertainty:
- **Score 0**: Definitely false
- **Score 1**: Uncertain / might be true
- **Score 2**: Definitely true

**Research Question**: Does allowing the model to express uncertainty (score 1) improve the precision of its confident predictions (scores 0 and 2)?

**Hypothesis**: If the model can say "I'm uncertain" for ambiguous cases, its definite predictions should be more accurate.

## Experiment Design

### Two Prompt Variants

1. **WITH Uncertain** - Allows scores 0, 1, 2
   - Model can express uncertainty
   - Encourages the model to use score 1 when unsure

2. **WITHOUT Uncertain** - Allows scores 0, 2 only
   - Forces binary decisions
   - Model must commit to definite predictions

### Evaluation Methodology

1. Run both prompts on the same set of images
2. For the "WITH uncertain" variant, **filter out all score 1 predictions**
3. Compare precision/recall on the **certain predictions only** (scores 0 and 2)
4. Measure: Does the "WITH uncertain" variant have better precision on its certain predictions?

## Usage

### Basic Usage

```python
from pathlib import Path
from src.llms.relevance_eval import RelevanceComparator

# Setup
comparator = RelevanceComparator(
    openai_apikey="your-key",
    block_colors=["red", "blue", "green", "cyan"]
)

# Load images and ground truth
image_paths = list(Path("images").glob("state_*.png"))
ground_truth = load_ground_truth("ground_truth.json")

# Run experiment
metrics_with, metrics_without = comparator.run_comparison_experiment(
    image_paths=image_paths,
    ground_truth=ground_truth
)

# Save results
comparator.save_results_to_csv(
    metrics_with, metrics_without,
    output_path=Path("results")
)
```

### Single Image Extraction

```python
from src.llms.relevance_eval import PromptVariant

# Extract with uncertain option
preds_with = comparator.extract_predicates_with_relevance(
    Path("state_0001.png"),
    PromptVariant.WITH_UNCERTAIN
)

# Filter to certain predictions only
preds_certain = comparator.filter_to_certain_predictions(preds_with)

# Extract without uncertain option (already certain)
preds_without = comparator.extract_predicates_with_relevance(
    Path("state_0001.png"),
    PromptVariant.WITHOUT_UNCERTAIN
)
```

## Ground Truth Format

Ground truth should be a JSON file mapping image names to predicate truth values:

```json
{
  "state_0000": {
    "clear(red:block)": 1,
    "on(red:block,blue:block)": 0,
    "ontable(red:block)": 1,
    "holding(red:block)": 0
  },
  "state_0001": {
    ...
  }
}
```

Values: `0` (false) or `1` (true)

## Metrics

The system computes the following metrics for both positive and negative classes:

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)

### Positive Class Metrics
- Model predicts "2" (definitely true)
- Ground truth is "1" (true)

### Negative Class Metrics
- Model predicts "0" (definitely false)
- Ground truth is "0" (false)

## Output

### Console Output

```
=============================================================
Metrics Comparison: Overall Results
=============================================================

--- WITH Uncertain Option (allows '1') ---
Positive Class:
  Precision: 0.9500, Recall: 0.9200, F1: 0.9348
  TP: 92, FP: 5, TN: 150, FN: 8

Negative Class:
  Precision: 0.9800, Recall: 0.9400, F1: 0.9596
  TP: 150, FP: 3, TN: 92, FN: 10

--- WITHOUT Uncertain Option (only '0' or '2') ---
Positive Class:
  Precision: 0.9200, Recall: 0.9500, F1: 0.9348
  TP: 95, FP: 8, TN: 147, FN: 5

Negative Class:
  Precision: 0.9600, Recall: 0.9200, F1: 0.9396
  TP: 147, FP: 6, TN: 95, FN: 12

--- Delta (WITH - WITHOUT) ---
Positive Class:
  Precision: +0.0300
  Recall:    -0.0300
  F1:        +0.0000

Negative Class:
  Precision: +0.0200
  Recall:    +0.0200
  F1:        +0.0200
=============================================================
```

### CSV Output

Results are saved to `comparison_summary.csv`:

```csv
variant,class,precision,recall,f1,tp,fp,tn,fn,total_samples
with_uncertain,positive,0.95,0.92,0.9348,92,5,150,8,255
with_uncertain,negative,0.98,0.94,0.9596,150,3,92,10,255
without_uncertain,positive,0.92,0.95,0.9348,95,8,147,5,255
without_uncertain,negative,0.96,0.92,0.9396,147,6,95,12,255
```

## Analysis

### Key Questions

1. **Does allowing uncertainty improve precision?**
   - Compare positive precision: WITH vs WITHOUT
   - Compare negative precision: WITH vs WITHOUT

2. **What is the trade-off?**
   - WITH uncertain: Higher precision, but fewer predictions (some filtered as uncertain)
   - WITHOUT uncertain: All predictions counted, but may be less accurate

3. **Is the model well-calibrated?**
   - Are score 1 predictions truly uncertain?
   - Do they fall between clear true/false cases?

### Interpreting Results

**If precision is HIGHER with uncertain option:**
- ✅ Model is well-calibrated
- ✅ Expressing uncertainty helps avoid false positives/negatives
- ✅ Score 1 is being used appropriately for ambiguous cases

**If precision is LOWER with uncertain option:**
- ⚠️ Model may be using score 1 too liberally
- ⚠️ Or: model is less confident when forced to make binary decisions
- ⚠️ May need prompt tuning or different threshold

## Advanced Usage

### Per-Predicate Analysis

```python
from src.llms.relevance_eval import MetricsCalculator

# Compute metrics by predicate type
metrics_by_type_with = MetricsCalculator.compute_metrics_by_predicate_type(
    predictions_with, ground_truth
)

metrics_by_type_without = MetricsCalculator.compute_metrics_by_predicate_type(
    predictions_without, ground_truth
)

# Compare "on" predicates
on_metrics_with = metrics_by_type_with["on"]
on_metrics_without = metrics_by_type_without["on"]

print(f"'on' precision WITH:    {on_metrics_with[0].precision:.4f}")
print(f"'on' precision WITHOUT: {on_metrics_without[0].precision:.4f}")
```

### Custom Temperature

```python
# Lower temperature = more deterministic
preds = comparator.extract_predicates_with_relevance(
    image_path,
    PromptVariant.WITH_UNCERTAIN,
    temperature=0.1  # Very deterministic
)

# Higher temperature = more random/creative
preds = comparator.extract_predicates_with_relevance(
    image_path,
    PromptVariant.WITH_UNCERTAIN,
    temperature=0.7  # More exploratory
)
```

## Implementation Details

### Prompt Design

**WITH Uncertain:**
```
- 2 → definitely holds
- 1 → might hold (unclear/partial/occluded)
- 0 → definitely does not hold

Use 1 when uncertain.
```

**WITHOUT Uncertain:**
```
- 2 → definitely holds
- 0 → definitely does not hold

🚫 DO NOT use score 1.
Make a definite decision.
```

### Filtering Logic

```python
def filter_to_certain_predictions(predictions):
    # Only keep scores 0 and 2
    return {p: s for p, s in predictions.items() if s in [0, 2]}
```

### Metrics Computation

For each variant:
1. Collect all predictions across images
2. Filter to certain predictions (0 and 2)
3. Compare against ground truth (0 and 1)
4. Compute TP, FP, TN, FN
5. Calculate precision, recall, F1

## Files

- `relevance_comparator.py` - Main comparison class
- `metrics_calculator.py` - Precision/recall computation
- `example_usage.py` - Usage examples
- `README.md` - This file

## Future Enhancements

1. **Calibration analysis**: Plot score 1 predictions to see if they're truly ambiguous
2. **Multiple trials**: Run each prompt multiple times to measure variance
3. **Threshold tuning**: Experiment with score thresholds (e.g., treat 1 as 0.5)
4. **Domain-specific prompts**: Customize prompts for different PDDL domains
5. **Confidence intervals**: Add statistical significance testing

## References

- **Relevance Judgments**: Binary classification with uncertainty
- **Precision-Recall Trade-off**: Balancing false positives vs false negatives
- **Model Calibration**: Do predicted probabilities match actual probabilities?
- **GPT-4 Vision**: Multimodal language model for image understanding
