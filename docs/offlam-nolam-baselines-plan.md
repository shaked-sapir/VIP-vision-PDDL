# OffLAM and NOLAM as regime-specific baselines

Two learners by Lamanna et al., both packaged as AMLGym dependencies, each
built for exactly one of our two degradation axes:

| Arm | Paper | Built for | Cells where it is a fair competitor |
|---|---|---|---|
| NOLAM | Lamanna, Serafini. *Action model learning from noisy traces: a probabilistic approach.* ICAPS 2024 | full but noisy states | `masking_p == 0`, any `noising_p` |
| OffLAM | Lamanna, Serafini, Saetti, Gerevini, Traverso. *Lifted action models learning from partial traces.* AIJ 339 (2025) | partial states, full actions (the paper's setting 1) | `noising_p == 0`, any `masking_p` |

Neither is a competitor in the image cells, where the classifier both abstains
and errs, so both are gated out of `source: image`.

The document has three parts. Part I is the foundation both arms share and is
built once. Part II is NOLAM end to end (code, runs, reporting). Part III is
OffLAM end to end. Parts II and III are independent of each other and are
meant to be worked one at a time, NOLAM first because it is the smaller one.

**Status (2026-09-16).** Part I is built and gates G1–G4 passed (G1 on all
five domains, hyphens intact; G3 scored a `?param_i` depot model at 1.0/1.0).
Parts II and III are built, registered (`nolam`, `offlam`), wired into
`benchmark_runner`, `experiment_runner`, `backfill_baseline` and
`dashboard_config.yaml`, and verified by backfilling copies of two
`simulation-final-run` hanoi cells in a scratch directory: NOLAM on
`mask=0.0__noise=0.1` (realised `e` 0.078 against nominal 0.1, model in 3.5 s)
and OffLAM on `mask=0.1__noise=0.0` (realised mask 0.085, model in 3.4 s at 8
traces, precision 0.84 against CDPS's 0.73 in the same cell); each was gated
out of the other's cell with the results file left byte-identical.

Both arms were then backfilled into every cell of the current 3 x 3
small-data sweep (`simulation-final-run`, five domains, 900 learner runs in
15 minutes, no errors). The audit of that pass found two defects, both fixed
and the pass repeated with `--force`:

- **Untyped parameters.** Depot declares `(clear ?x)` with no type. Both
  packages' regex domain parsers take arity from typed parameters, so they
  read it as `(clear)`, every learned depot model declared it zero-arity, and
  no depot problem was solvable for either arm. The learners now receive a
  typed copy of the domain (`lamanna/domain_dialect.py`: untyped parameters
  as `- object`, the flat type list rooted at `object`); evaluation still uses
  the untouched reference. The other four domains are fully typed and only
  gain the root declaration.
- **Empty operators.** NOLAM emits an operator it never observed with empty
  preconditions and effects. Fast Downward fails on that with an internal
  error, which AMLGym's solving metric counts in no bucket, so the row read
  solving 0 with every other ratio 0. `nolam_glue` now drops operators left
  with nothing, as PI-SAM does (AMLGym's syntactic metrics score a missing and
  an empty operator alike), and the row records `unobserved_operators`,
  `operators_dropped_empty` and `n_operators_emitted`. OffLAM's cautious model
  gives an unobserved operator its full possible-precondition set, so it was
  never affected.
- **Grouped parameter types.** The first version of the typed copy treated
  `?from` in npuzzle's `(?tile - tile ?from ?to - position)` as untyped, so
  NOLAM saw `move(tile, object, position)` and lost most candidate atoms, and
  OffLAM's models failed to parse in unified planning. The typing now respects
  PDDL's shared-type runs, and a domain with no untyped parameter is passed
  through unchanged, so only depot is rewritten.

The third pass (all 45 experiments, `--force`) is the one on disk: 900 rows,
no errors, every expected row present and completed, none in a gated cell,
no empty operator and no zero-arity `clear` in any emitted model, every
solving outcome classified. Sanity reads from it: both arms are near-perfect
in the clean anchor cell of every domain; OffLAM is flat across masking 0 to
0.1 (its paper's ceiling region); NOLAM degrades with noise and beats CDPS on
gripper and npuzzle while trailing it on blocksworld and depot at noise 0.1.
The "Order of work" step 4 (grid extension) has not been started.

Decisions already taken, so they are not re-opened below:

- **Corruption stays `percentage`.** Both papers corrupt each literal
  independently (i.i.d. Bernoulli). Our `percentage` strategy hides or flips
  an exact `round(N * p)` of a state's `N` grounded atoms, drawn uniformly, so
  every atom's marginal rate is `round(N * p) / N`, i.e. `p` up to rounding,
  with lower variance. Every arm in a cell reads the same files, and
  within-cell comparability is the claim the thesis makes. Existing CDPS,
  MILP and ROSAME rows were produced this way and the new rows must sit
  beside them. The `random` strategies reproduce the papers' process exactly
  and stay available as a config switch, but rows from the two strategies are
  separate experiments.
- **Realised rates are recorded, not assumed.** `max(1, round(N * p))` makes
  a nominal 0.01 about 0.02 on depot-sized states, and frame-axiom
  propagation from the clean init state un-masks a few literals per trace.
  Each arm's row records the nominal `p` and the rate measured on the fold's
  frozen data.
- **OffLAM is compared in setting 1 only** (partial states, full actions).
  Our pipeline never hides an action, and no other arm could consume a trace
  with one missing; the paper's settings 2 and 3 are a different question.
- **No negative preconditions anywhere.** CDPS runs under
  `NegativePreconditionPolicy.hard`; none of the five sweep domains has one
  (blocksworld's `put_down` requires only `(holding ?x)`, its
  `(not (handempty))` is an effect; only `hiking` has any, and it is not in
  the sweep). NOLAM is therefore run as `MAP_pre+`.
- **At `noising_p == 0`, CDPS is PI-SAM.** No wrong literal means no
  conflict, so the search returns PI-SAM's model, and the MILP arms likewise.
  The OffLAM comparison is a comparison of base learners under partial
  observability and is framed that way.
- **New grid points are new cells, not backfills.** `backfill_baseline.py`
  adds rows to cells whose `original_observations/` exist. A new noise or
  masking level is generated by `benchmark_runner` under the same `run_name`
  with `--only-noise` / `--only-mask`; the simulated source is seeded per
  fold, so old cells are untouched and new ones land beside them.
- **The grid is fixed.** Every current sweep (the list is the simulation
  entries of `dashboard_config.yaml`, never a directory listing of
  `running_results/`) grows to

  ```
  masking_ps: [0.0, 0.01, 0.1, 0.2, 0.3, 0.4]     # 0.2, 0.3, 0.4 new
  noising_ps: [0.0, 0.1, 0.2, 0.3, 0.4]           # 0.3, 0.4 new
  ```

  as a full cross for every arm already in the sweep: 30 cells per domain,
  of which the existing 3 x 3 = 9 stay as they are and 21 are generated. No
  other value is added now. NOLAM lives in the `mask = 0` row and OffLAM in
  the `noise = 0` column of this grid, both enforced by the gate; nothing
  about the grid is arm-specific.

---

## Part I. Shared foundation

Both learners share one trace grammar, one domain parser lineage and the
same operational hazards, so one adapter package serves both.

### I.1 What the learners consume and emit

**Trace grammar** (`offlam/src/Learner.py:parse_trace`,
`nolam/algorithm/Learner.py:parse_trace`, identical):

- only lines starting with `(:state` or `(:action` are read; any wrapper
  such as `(:trajectory` is ignored;
- one state per line, `(:state (p a b) (not (q c)) ...)`; zero-arity atoms
  as `(p)`;
- one action per line, `(:action (move a b))`;
- states and actions alternate, first and last lines are states;
- an atom absent from a state is **unknown** (open world). A state with no
  explicit negative literal triggers a warning; a malformed line calls
  `exit()`.

Our `.trajectory` convention is the opposite: unmasked positives only, absent
means false or masked, the masked set with its sign lives in `.masking_info`
(`src/utils/pddl_trajectory.py:observation_to_trajectory_file`).

**Domain parser** (`ActionModel.read` + `clean_pddl_domain_file` in both
packages): regex-based, drops `;` comment lines, lowercases, renames each
operator's parameters to `?param_1 ... ?param_n`, requires `(:types` before
`(:predicates`. All `src/domains/*/*.pddl` satisfy this (flat typed domains,
`:strips :typing`). The learned model comes back as PDDL text with `?param_i`
names and `(:requirements :typing)`.

**Operational hazards, shared:** neither learner takes a time budget; both
call `exit()` on a parse error; both write `tmp/` and `PDDL/` relative to
the cwd, where AMLGym's `problem_solving` also writes a *file* named `./tmp`
(the reason `run_fold` already chdirs per fold).

**Not used:** `amlgym.algorithms.OffLAM` / `.NOLAM` adapters. Their
`_preprocess_trace` reads AMLGym's own trajectory format and adds `(not l)`
for every atom missing from the positives, which turns a masked atom into an
explicit false. Importing `amlgym.algorithms` also imports every adapter in
the package, torch-backed ROSAME included. The `nolam` and `offlam` packages
are called directly.

### I.2 Modules

```
benchmark/algorithm_adapters/lamanna/
    __init__.py
    traces.py            write_partial_trace(observation, out_path) -> Path
    subprocess_learn.py  run_learner(fn, kwargs, cwd, timeout_s, log_path) -> LearnOutcome
    realised_rates.py    realised_rates(prepared_trajectories, gt_lookup, domain) -> RealisedRates
```

`traces.py`, from the masked grounded `Observation` that CDPS and the
symbolic ROSAME runner already consume (`load_masked_observation` in
`src/utils/masking.py`): per state, unmasked positive -> `(p a b)`, unmasked
negative -> `(not (p a b))`, masked -> omitted; one state per line; the
component's action as `(:action (name o1 o2))`; wrapped in `(:trajectory
... )`. Negatives are the full closed-world set (the papers' definition of a
complete state). In a `masking_p == 0` cell this yields complete states; in
a `noising_p == 0` cell every written literal is true.

`subprocess_learn.py`: runs a learner function in a child process with
`cwd = work_dir / "<arm>_workspace"`, a wall-clock kill at `timeout_s`,
stdout and stderr to `<workspace>/learner.log`, and returns
`LearnOutcome(model_text | None, terminated_by in {completed, timeout,
error}, wall_seconds, detail)`. A child process is the only construct that
gives the `timeout_seconds` the base class promises, survives the learners'
`exit()`, isolates their relative paths, and keeps `nolam`'s module-global
configuration from leaking.

`realised_rates.py`: diff the fold's frozen `original_observations/` (plus
`.masking_info`) against the GT trajectories, the same comparison the
`compare-original-observations` skill performs, and return the fraction of
grounded atoms hidden and the fraction flipped over all states of the fold.
Both arms record these; NOLAM also consumes one of them (II.2).

### I.3 Regime gate

```python
@dataclass(frozen=True)
class DegradationRegime:
    source: str                 # "image" | "simulated"
    masking_p: Optional[float]  # None for image
    noising_p: Optional[float]  # None for image
```

`BaselineRunner.supports(regime) -> Tuple[bool, str]`, default
`(True, "")`. `resolve_algorithms(names, regime=..., **runner_kwargs)` drops
a runner whose `supports` says no, prints the reason once, and returns the
dropped `{row_name: reason}` map, which `experiment_runner.main` writes to
the cell's `run_params.json` as `skipped_baselines`. The regime also reaches
runner constructors through the existing `_instantiate` filter.

Call sites:

- `benchmark_runner.execute_run`: `_build_main_kwargs` resolves algorithms
  once per run today; the `resolve_algorithms` call moves into the per-cell
  loop, where `masking_p` / `noising_p` are known. Everything else stays
  run-wide.
- `backfill_baseline.py`: builds the regime from the cell's
  `run_params.json` (`p_mask`, `p_noise`, `source`).
- `baseline_regime_gate: strict | off` in `shared:` (default `strict`)
  disables the gate for a deliberate misapplication study.

`BaselineRunner.paper` is documented today as `'24'` / `'26'`; its docstring
widens to "publication tag" and the two arms use `"icaps24"` / `"aij25"`.

### I.4 Verification gates for Part I

Run by hand in the scratchpad on two finished folds (a depot
`mask=0.0__noise=0.1` cell for hyphenated identifiers and a hanoi
`mask=0.1__noise=0.0` cell), before any runner is written:

| Gate | Pass criterion |
|---|---|
| G1 domain parsing | both packages' `ActionModel(domain_reference.pddl)` load depot and hanoi without error; operator and predicate counts match the reference |
| G2 trace round trip | a file written by `write_partial_trace` is parsed by both packages' `parse_trace` with the same number of states, actions, positive and negative literals as the source observation; masked atoms are absent |
| G3 scoring | a `?param_i` model text is scored by `evaluation.evaluate_model` with the same precision/recall as the same model rewritten with the reference parameter names (ROSAME output already goes through this path; the gate pins it for these emitters) |
| G4 isolation | `run_learner` reports `timeout` on a sleeping function and `error` on a function that calls `exit()`; the calling process survives both |

Fallback if G1 fails on hyphens: `src/utils/pddl_naming.py` (`canonical` in,
`rewrite_symbols` out), the pair `benchmark/baselines/image_fold_inputs.py`
already uses for the same purpose.

Tests: `traces.py` (masked omitted, negatives explicit, one state per line,
zero-arity atoms, repeated objects in an action, round trip through
`nolam`'s parser); `subprocess_learn.py` (G4 as a test);
`realised_rates.py` (a hand-built fold with known counts); the gate
(`test_benchmark_runner_filters.py`, `test_backfill_runner_kwargs.py`).

---

## Part II. NOLAM

### II.1 The learner

`nolam.algorithm.Learner().learn(domain_path, trace_paths, e) -> ActionModel`,
`str(model)` is the PDDL text. Per (operator, atom) it counts the four
before/after observed-value combinations across that operator's transitions,
computes the likelihood of nine precondition/effect hypotheses under
"each observed value is wrong with probability `e`, independently", and takes
the MAP hypothesis (`Configuration.SAMPLING = False`). Sub-second.

`e` is a **learning parameter, not a data parameter**: it is passed at
learning time and decides how much disagreement between transitions NOLAM
attributes to sensor error rather than to the model. Too small and it learns
the noise as effects; too large and the prior dominates. The paper hands it
the generating value.

`Configuration.ALLOW_PREC_NEG` (module global): `True` is the paper's `MAP`,
`False` is `MAP_pre+`. MAP ties are broken with `np.random.choice`;
`Configuration.RANDOM_SEED` is declared but never applied.

Handles full states only: an atom missing from a state simply does not enter
the counts and silently biases the posterior. Hence `masking_p == 0` only.

### II.2 Why exact-fraction noise is valid input

NOLAM reads per-atom marginal counts and assumes independence between atoms
by construction. Under `percentage` each atom's marginal flip probability is
`round(N * p) / N` (depot, `N = 52`, `p = 0.2`: `0.192`); the only departure
from Bernoulli is a weak negative dependence between atoms of one state,
which NOLAM ignores whichever way the data was made. The data match the
model at the level it reads it; the lower variance is shared by every arm.

The clean init state contributes zero flips to the first transition's
"before" values; the realised rate averaged over all states is slightly
below the rate of states `t >= 1`. Second-order, identical for every arm,
stated in the write-up.

### II.3 Runner

`benchmark/baselines/nolam_runner.py`, `NOLAMRunner(BaselineRunner)`,
registry key `nolam`, row name `NOLAM`, `input_kind = "symbolic"`,
`paper = "icaps24"`, `uses_milp = False`,
`supports(regime)`: `source == "simulated" and masking_p == 0`.

`learn()`:

1. workspace `work_dir / "nolam_workspace"`; write the reference domain and
   one trace per prepared trajectory via `write_partial_trace`;
2. resolve `e` (II.4); `realised_rates` for the record;
3. `run_learner(learn_nolam, ...)` where `learn_nolam` sets
   `Configuration.ALLOW_PREC_NEG`, seeds NumPy, calls `Learner().learn`,
   returns `str(model)`;
4. return `(model_text | None, report)`.

### II.4 Knobs

Threaded through `run_config.yaml` `shared:` via `_RUNNER_KWARG_KEYS`,
mirrored as `backfill_baseline.py` flags, recorded by `run_params()` in
`baseline_params` and in every row's `algorithm_specific`:

| Key | Default | Meaning |
|---|---|---|
| `nolam_noise` | `oracle` | `oracle` passes the fold's **realised** flip rate from `realised_rates` (the paper's protocol, "NOLAM is told the true noise rate", made exactly true under `percentage`). A float pins one `e` for every cell, for a misspecification study |
| `nolam_allow_neg_precs` | `false` | `false` = `MAP_pre+`, matching CDPS's `hard` policy and the GT domains |
| `nolam_seed` | `0` | NumPy seed for MAP tie-breaking |

Row fields (`algorithm_specific`): the three knobs, `nolam_e_source`
(`realised` | `nominal` | `pinned`), `nolam_e_used`, `nominal_mask_rate`,
`nominal_noise_rate` (the cell's `p_mask` / `p_noise`), `realised_mask_rate`
(0 by gate), `realised_noise_rate`, `n_traces`, `terminated_by`,
`wall_seconds`, `learner_log`.

### II.5 Runs

The NOLAM row of the grid is `masking_p = 0`, and the fixed grid gives it
`e in {0, 0.1, 0.2, 0.3, 0.4}`, the paper's exact range. The grid extension
is one operation for the whole sweep, done once, and serves Part III too:

```bash
# 1. in the sweep's config: the fixed grid above, and `nolam` (and later
#    `offlam`) in shared.algorithms
# 2. generate every cell not yet on disk; resume: true skips the existing 9
python -m benchmark.benchmark_runner --config benchmark/<sweep>.yaml
# 3. retrofit NOLAM into the 3 pre-existing mask=0 cells
python -m benchmark.backfill_baseline \
    --experiment-dir benchmark/running_results/*/<sweep>__mask=0.0__noise=* \
    --baselines nolam --workers 8
```

If a sweep is run cell by cell on the cluster, `--only-mask` / `--only-noise`
select single cells off the same config, as `scripts/cluster/` already does.
The L-sweep's 21 new cells times five sizes at a 1 h cap is a cluster job.

### II.6 Reporting

`dashboard_config.yaml`: `- key: NOLAM, modes: [simulation]` with a colour.
Heatmap cells outside `mask=0` render "–", which is what the gate produces.

Write-up points: NOLAM receives the true (realised) noise rate, which no
other arm is told; exact-fraction noise, same expectation as the paper's;
init state clean; `MAP_pre+`; `noise=0, mask=0` as the anchor cell where
NOLAM, PI-SAM and ROSAME should all be near-perfect.

### II.7 Done when

- Gates G1–G4 pass; the tests in I.4 and a `test_nolam_runner.py` (knobs
  recorded, gate refuses a `mask > 0` regime, row fields present) pass.
- The `mask=0` row of at least one small-data sweep has NOLAM rows at all
  five noise levels and the dashboard shows the series.

---

## Part III. OffLAM

### III.1 The learner

`offlam.algorithm.learn(domain_path, trace_paths) -> str`. Applies the
completion rules (definite and possible preconditions and effects, inertia,
fictitious-predicate splitting) over all traces to a fixpoint,
`greedy=True` hard-coded, and writes the **cautious** model the paper
evaluates: every precondition not yet ruled out, only certain effects. No
parameter of any kind. Post-fix copied from AMLGym's adapter:
`(:requirements)` -> `(:requirements :typing)`.

Assumes no negative preconditions and **no noise**: a wrong literal is truth,
and if `p` and `not p` ever meet in one observation the whole predicate is
deleted from the model (`check_inconsistency`). Hence `noising_p == 0` only.

Run time grows with the trace count: the paper reports ~1000 s CPU at 80 to
100 traces of 10 transitions. Ours are longer (depot: 16 to 19 states).

### III.2 Setting 1 and our masking

The paper (section 8.1) makes each state complete, cuts a 10-transition
window, and for setting 1 removes each literal of each state with
probability `1 - omega`, keeping every action with its name and arguments.
Settings 2 and 3 additionally replace whole actions with a wildcard, which
rule 17 handles by enumerating candidate ground actions; we do not use them.

Our `percentage` masking hides `round(N * p)` of a state's grounded atoms,
both polarities, for `t >= 1`, then propagates frame axioms from the clean
init state. So `omega = 1 - masking_p` up to the rounding and propagation
effects recorded as `realised_mask_rate`. OffLAM's own inertia rule derives
much of what our propagation adds, so the propagation is close to a no-op
for it.

### III.3 Runner

`benchmark/baselines/offlam_runner.py`, `OffLAMRunner(BaselineRunner)`,
registry key `offlam`, row name `OffLAM`, `input_kind = "symbolic"`,
`paper = "aij25"`, `uses_milp = False`,
`supports(regime)`: `source == "simulated" and noising_p == 0`.

`learn()`: workspace `work_dir / "offlam_workspace"`; domain and traces via
`write_partial_trace`; `realised_rates` for the record;
`run_learner(learn_offlam, ..., timeout_s=timeout_seconds)` where
`learn_offlam` calls `offlam.algorithm.learn` and applies the requirements
fix; a timeout yields `(None, report)` with `terminated_by = "timeout"`, the
same shape as a CDPS fold that exhausts its budget.

No knobs. Row fields (`algorithm_specific`): `omega_nominal` (`1 - p_mask`),
`nominal_mask_rate`, `nominal_noise_rate`, `realised_mask_rate`,
`realised_noise_rate` (0 by gate), `n_traces`, `terminated_by`,
`wall_seconds`, `learner_log`.

Fallback if OffLAM times out at the corpus sizes we care about: an
`offlam_relevant_negatives_only` knob that restricts explicit negatives to
atoms over the current action's objects, as AMLGym's adapter does. Neither
learner can lift any other atom to an operator, so it is a size optimisation
with no semantic effect. Not built unless needed.

### III.4 Runs

OffLAM's column of the grid is `noising_p = 0`, which the fixed grid gives
at `masking_p in {0, 0.01, 0.1, 0.2, 0.3, 0.4}`, i.e. `omega` from 1 down to
0.6. The paper reports OffLAM at its full-observability performance for
`omega >= 0.4` in all but three domains, so on this grid it is expected to be
near ceiling throughout, and the comparison is chiefly about PI-SAM's and the
other arms' behaviour as observability drops, with OffLAM as the reference
that should not degrade. Lower observability is deliberately not added now.

If Part II's grid extension has already run with `offlam` in
`shared.algorithms`, OffLAM rows exist in every new `noise = 0` cell and only
the 3 pre-existing ones need a backfill; otherwise the extension is run here
(same command as II.5) and the backfill covers all six:

```bash
python -m benchmark.backfill_baseline \
    --experiment-dir benchmark/running_results/*/<sweep>__mask=*__noise=0.0 \
    --baselines offlam --learn-timeout 600 --workers 8
```

PI-SAM (as CDPS), the MILP arms and ROSAME are in every cell of the grid
from the extension itself, which is what makes the within-cell comparison
exist. Budget: the cell's `learning_timeout_seconds`, enforced by the
child-process kill. Expect timeouts at L >= 500 in the L-sweep; they are
rows, not failures.

### III.5 Reporting

`dashboard_config.yaml`: `- key: OffLAM, modes: [simulation]` with a colour.

Write-up points: setting 1 only; cautious model, as the paper evaluates;
`omega = 1 - masking_p` with the realised rate stated; at `noise = 0` CDPS
equals PI-SAM, so this is a base-learner comparison; run time bounded by our
timeout where the paper used 1000 s CPU; low-observability OffLAM rows carry
extra preconditions by design (the paper reports the same).

### III.6 Done when

- Gates G1–G4 pass on the hanoi fold; `test_offlam_runner.py` (gate refuses
  `noise > 0`, timeout row shape, requirements fix applied) passes.
- The `noise=0` column of at least one small-data sweep has OffLAM rows at
  all six masking levels and the dashboard shows the series.

---

## Order of work

Code first, runs last. Nothing is launched until both arms are wired in and
verified on existing data.

1. Part I, gated by G1–G4 (hand checks on two finished folds, no runs).
2. Part II (NOLAM) code: runner, registry, knobs, tests. Verified by a
   `backfill_baseline --dry-run` and then a real backfill into one existing
   `mask=0.0__noise=0.1` cell of a small-data sweep, which exercises the
   whole path (gate, workspace, child process, realised rate, row merge,
   dashboard) on frozen data at no cost.
3. Part III (OffLAM) code: same, verified on one existing
   `mask=0.1__noise=0.0` cell.
4. Only then the runs: the grid extension on every current sweep with
   `nolam` and `offlam` already in `shared.algorithms`, so the 21 new cells
   per domain get every arm in one pass; then one backfill of both arms into
   the 9 pre-existing cells per domain; then the dashboard rebuild per the
   `results-dashboard` skill.
