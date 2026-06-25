Aggregate all conflict-free model PDDL files under an experiment into merged domain files.

**Experiment directory**: $ARGUMENTS

## What it produces

Scans every `testing/**/conflict_free_models/conflict_free_model_*/model.pddl` under the experiment and writes:

1. **`union_pre_intersect_eff.pddl`** — union preconditions · intersect effects
2. **`intersect_all.pddl`** — intersect preconditions · intersect effects
3. **`vote_0p25.pddl`** — literals in **≥ 25%** of CFMs
4. **`vote_0p50.pddl`** — literals in **≥ 50%** of CFMs
5. **`vote_0p75.pddl`** — literals in **≥ 75%** of CFMs
6. **`vote_0p90.pddl`** — literals in **≥ 90%** of CFMs

Default output directory: `<experiment_dir>/aggregated_domains/`

## Steps

1. Run from the project root with the venv activated:

```
source venv11/bin/activate && python -m benchmark.evaluation.cfm_domain_aggregate "$ARGUMENTS"
```

2. Present:
   - Number of CFM `model.pddl` files aggregated
   - Output paths for all six merged domains
   - Path to `aggregation_summary.json` (per-action literal breakdown)

3. If no CFM models are found, say so explicitly.
