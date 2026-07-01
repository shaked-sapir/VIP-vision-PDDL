Generate a fully-detailed Excel report from a completed experiment directory.

**Experiment directory**: $ARGUMENTS

## Steps

1. Run the following command from the project root with the venv activated:

```
source venv11/bin/activate && python -m benchmark.evaluation.experiment_report "$ARGUMENTS"
```

2. If the command succeeds, report the output path and a brief summary of what was generated (number of CFMs found, number of folds, sheets produced).

3. If it fails, show the error and investigate.
