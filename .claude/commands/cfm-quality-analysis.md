Analyze whether later CFMs improve over earlier ones for a completed experiment.

**Experiment directory**: $ARGUMENTS

## Steps

1. Run the following command from the project root with the venv activated:

```
source venv11/bin/activate && python -m benchmark.evaluation.cfm.cfm_quality_analysis "$ARGUMENTS"
```

2. If the command succeeds, report:
   - Output directory: `<experiment_dir>/evaluation_results/CFM_quality/`
   - Number of instance directories scanned
   - Number of instances with >= 2 CFMs
   - CFM count distribution (from stdout)
   - List of 5 PNG files generated

3. If no instances have >= 2 CFMs, say so explicitly.

4. If it fails, show the error and investigate (missing testing/, bad JSON, matplotlib issues).
