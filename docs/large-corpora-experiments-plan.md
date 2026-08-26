# Large-corpora experiments: PI-SAM+MILP vs the symbolic ROSAME arms

Task 3. Head-to-head on large **symbolic** corpora, with both families stopped by
a convergence rule rather than by a fixed budget.

Status: **plan only — nothing implemented.** Every number below that is called
"measured" comes from the 3060 `milp_loop_rounds.json` logs already on disk under
`benchmark/running_results/*/simulation-final-run__mask=*__noise=*`; every number
called "projected" is an extrapolation from folds of at most 8 traces.

---

## 0. Scope

**In:** symbolic arms only — `pisam_milp_loop` (two GT variants) against
`rosame_milp_24` and `rosame_24`.

**Out:** the imaged arms. There is no symbolic ICAPS-26 arm — `rosame_i_26` and
`rosame_i_milp_26` are imaged, and `rosame26_data.py` is built on
`image_fold_inputs.py`, whose job is walking image frames. "ROSAME-26 symbolic"
in earlier discussion means the symbolic ICAPS-24 arms, with and without MILP.

The ICAPS-26 UNSAT investigation (`docs/rosame-i-26-failure-suggestions.md`) is
deferred: understanding it needs large *image* corpora, to separate trace count
from epoch count, and this task is symbolic.

---

## 1. Experiment shape

| knob | value |
|---|---|
| domains | blocksworld, depot, gripper, hanoi, npuzzle |
| L (training-set size) | 10, 50, 100, 500, 2000 |
| folds | 5, `cv_scheme: kfold` — disjoint test stripes |
| train/test split | 80/20 of the WHOLE corpus, per fold |
| mask x noise grid | reuse the existing 3x3 (mask 0 / 0.01 / 0.1, noise 0 / 0.1 / 0.2) |
| arms | `pisam_milp_loop` with `gt_anchoring: init_only` and `gt_anchoring: none`; `rosame_milp_24`; `rosame_24` |
| cell timeout | 1 h ceiling, configurable |
| subset size | **fixed 4** (not `half`) |

### 1.0 What L is, and what the folds are

L is a **training-set size**, not a fold size. The harness splits the corpus 80/20
per fold and then draws L problems from that fold's 2000-problem *training pool*:
`prepare_fold_trajectories` takes `selected_problem_dirs[:num_trajectories]`, a
prefix, so every smaller L is a strict subset of every larger one (nested
sampling). One corpus therefore serves the whole L sweep — L is a fraction of the
training pool: 0.5 %, 2.5 %, 5 %, 25 %, 100 %.

So a single 2500-problem corpus per domain covers L = 10 .. 2000. No per-L
corpora are needed. (The per-L *run configs* remain, but only because patience
depends on L and one YAML scalar cannot express that — see SS3.)

**An earlier version of this section said "5 folds, disjoint partition of the
corpus, 1600 train / 400 test per fold". Both halves were wrong.** The split is
over the whole corpus, giving 2000 train / 500 test at n=2500; and the folds were
not disjoint at all — see SS1.2.

### 1.2 Monte-Carlo vs k-fold

The splitter (`run_fold.cv_split`) offers two schemes:

- `montecarlo` — an independent 80/20 split per fold, seeded `42 + fold`. **This
  is what every result on disk used.** The five test sets overlap: measured at
  n=2500, any two share 84-119 problems. Fold results are therefore correlated
  and the across-fold std understates the true variance.
- `kfold` — one shuffle seeded `42` for all folds, then disjoint stripes
  (`indices[fold::n_folds]`). Every problem is tested exactly once.

Task 3 uses **`kfold`**; `montecarlo` stays the default so existing results stay
reproducible. The two are not comparable — rows from different schemes are
measured on differently-constructed test sets, and nothing in the dashboard
distinguishes them today.

### 1.1 Why `subset_size: 4` and not `half`

`SubsetSizeKind.HALF` is the current default, and the logs confirm the subset is
exactly `ceil(N/2)`:

| fold traces | subset used |
|---|---|
| 3, 4 | 2 |
| 5, 6 | 3 |
| 7, 8 | 4 |

Under `half`, L=2000 would hand CP-SAT 1000-trace MILPs. Measured solve cost
tracks *subset* size, not fold size — median 0.21 s at subset 2, 0.49 s at
subset 4 — so pinning the subset at 4 keeps per-round solve cost flat in L.

**This is the change that makes large L affordable at all.** An earlier claim
that "MILP feasibility breaks at large L" was wrong: it is a property of the
`half` policy, not of the corpus size.

---

## 2. The stop-rule problem

### 2.1 What the logs show

Across all 3060 loop runs:

```
stop_reason:  fixpoint 2145 | perfect_fit 900 | sampler_exhausted 15 | no_improvement 0
best_round:   median 1, mean 2.0
rounds after best: median 6, mean 16.9, max 70
```

`no_improvement` fired **zero** times — and not because it lost a race:
`no_improvement_rounds` is blank in `benchmark/run_config.yaml:54`, i.e. the rule
was switched **off** for every run on disk. So these logs say nothing about how
the rule behaves; they only establish what the other rules did.

Rounds per run grow steeply with fold size:

| traces | median rounds | max |
|---|---|---|
| 3 | 4 | 11 |
| 4 | 7 | 18 |
| 5 | 11 | 25 |
| 6 | 21 | 53 |
| 7 | 36 | 70 |
| 8 | 71 | 151 |

### 2.2 Why `fixpoint` cannot survive

`fixpoint` fires when every admissible `(subset, model)` pair has been solved.
At L=2000, m=4 that is C(2000,4) ~ 6.6e11 rounds. `fixpoint` and
`sampler_exhausted` between them ended 2160 of 3060 runs; both become unreachable.
With `no_improvement` off, the only rule left is `budget_seconds` — and a
timeout is not a convergence criterion. The arm's result would become a function
of the clock, which is exactly the protocol artifact this task exists to avoid.

### 2.3 The rule: rounds since best

`fixpoint` asks "have I tried everything?" — combinatorial in N.
Rounds-since-best asks "have I stopped finding anything?" — independent of N.

This is the direct analogue of ROSAME's `has_converged` window
(`src/milp/rosame26_budget.py:332`), which stops when the training loss has not
improved the running best by more than 0.2 % for 3 consecutive 40-epoch windows.

**What it gives up:** at N=8 the loop genuinely exhausted C(8,4)=70 subsets and
*proved* no better one existed. At L=2000 patience gives up that guarantee for
"stopped when sampling stopped paying." That is the right trade at this scale,
but it changes what a `stop_reason` means and the paper must say so.

The state this needs already exists: `_LoopState.rounds_without_improvement`
(`src/plan_denoising/milp_denoiser/loop.py:674`) is incremented on a
non-improving round and reset to 0 on an improving one
(`loop.py:893-905`). `_stop_reason` already reads it (`loop.py:451-454`).

### 2.4 A relative-improvement threshold is NOT needed

The earlier proposal was to port ROSAME's 0.2 % relative threshold to V. The logs
refute it. Relative gains at improving rounds (n=1115):

| percentile | gain |
|---|---|
| p10 | 0.35 % |
| p25 | 0.70 % |
| **p50** | **1.87 %** |
| p75 | 5.59 % |
| p90 | 22.1 % |
| min observed | 0.13 % |

Nothing below 0.1 %; only 2.7 % of gains fall below 0.002. V is integer-valued
over a frozen finite pool, so the loop does not converge asymptotically the way a
gradient descent does. A plateau threshold answers a question this loop never
asks. **Do not add `min_relative_improvement`.**

---

## 3. Choosing the patience value

### 3.1 The measured quantity

Patience must exceed the **gap between consecutive improvements**. Measured:

| N | median gap | p90 | p99 | max |
|---|---|---|---|---|
| 3 | 1 | 3 | 3 | 3 |
| 4 | 1 | 4 | 6 | 6 |
| 5 | 1 | 5 | 9 | 10 |
| 6 | 1 | 8 | 15 | 20 |
| 7 | 1 | 9 | 21 | 35 |
| 8 | 1 | 12 | 29 | 70 |

The median gap is **1 at every N** — improvements, when they arrive, arrive at
once. The *tail* is what grows: p90 goes 3 -> 12, p99 goes 3 -> 29.

(The "rounds after best" column in §2.1 is a measurement artifact — those runs
were terminated by `fixpoint` at exactly C(N,4) rounds, so the tail equals the
space size and carries no signal. The gap distribution is the real evidence.)

### 3.2 The model

Each round samples a subset; let `p` be the probability that one sampled subset
yields an improvement, `0 < p < 1`. Rounds are independent samples, so the gap
until the next improvement is Geometric(p) and

```
P(miss after k barren rounds) = (1 - p)^k
```

To miss a real improvement with probability at most `alpha`:

```
(1 - p)^k <= alpha
k * ln(1 - p) <= ln(alpha)        # ln is increasing, direction holds
k >= ln(alpha) / ln(1 - p)        # ln(1-p) < 0, so dividing FLIPS it
```

The flip is the step to watch: `0 < 1-p < 1` makes `ln(1-p)` negative, and
dividing an inequality by a negative number reverses it. Both `ln(alpha)` and
`ln(1-p)` are negative, so the quotient is positive. Patience is a **floor**,
not a ceiling — more patience means fewer missed improvements, so "miss rarely"
puts a lower bound on `k`. (`k <= ...` would read as "be impatient enough", which
`k = 0` would satisfy while missing everything.)

Worked example, `p = 0.1`, `alpha = 0.01`:

```
k >= ln(0.01) / ln(0.9) = -4.605 / -0.1054 = 43.7  ->  k = 44
check: 0.9^44 = 0.0097 <= 0.01   OK
       0.9^43 = 0.0108 >  0.01   too small
```

**From this to the `ln C` form.** For small `p`, `ln(1-p) ~ -p`, so

```
k >~ ln(1/alpha) / p
```

i.e. patience scales like `1/p`. As the subset space `C(N,4)` grows, the fraction
of subsets that still buy an improvement shrinks, and the measured p99 gap
(SS3.1) grows close to logarithmically in `C(N,4)`. Substituting that empirical
`1/p ~ ln C(N,4)` relation gives the form used below:

```
patience = ceil(a * ln C(N, 4))
```

The `ln C` dependence is **empirical, not derived** — the geometric bound gives
`k ~ 1/p`, and only the measurements in SS3.1 tie `1/p` to `ln C(N,4)`. That tie
rests on four points at N <= 8; SS3.4 is the caveat.

### 3.3 Fitting `a`

Solving `a = p99 / ln C(N,4)` at each measured N:

| N | C(N,4) | ln C | p99 gap | required a |
|---|---|---|---|---|
| 5 | 5 | 1.61 | 9 | 5.59 |
| 6 | 15 | 2.71 | 15 | 5.54 |
| 7 | 35 | 3.56 | 21 | 5.91 |
| 8 | 70 | 4.25 | 29 | **6.83** |

**`a = 3` is refused by the data** (it was proposed before this fit and would
give patience 13 at N=8, against an observed p99 of 29). The required
coefficient is 5.5-6.8, and it *drifts upward* with N over the four points
available — so a fitted constant is itself an extrapolation, not a law.

Taking `a = 7` (above every observed requirement, with margin for the drift):

| L | C(L,4) | ln C | **patience** |
|---|---|---|---|
| 10 | 210 | 5.35 | **38** |
| 50 | 2.3e5 | 12.35 | **87** |
| 100 | 3.9e6 | 15.18 | **107** |
| 500 | 2.6e9 | 21.67 | **152** |
| 2000 | 6.6e11 | 27.22 | **191** |

Growth is slow — 200x the corpus buys 5x the patience — so cost stays bounded.

**Simpler alternative, if defending a formula in the paper is unattractive:**
patience 40 for L <= 100, 150 for L >= 500. Same shape, two numbers, states in
one sentence. The formula is more principled; the two-value table is more
readable. Either is defensible; this is a presentation choice.

### 3.4 The caveat that decides whether any of this holds

Every number in §3.1 comes from folds of at most **8** traces, where the sampler
could see most of C(N,4). At L=2000 each subset is a vanishing sample, and the
improvement-arrival process may stop being geometric with a comparable `p` —
improvements could arrive sporadically rather than front-loaded, in which case
even patience 191 stops early.

**This is a gate, not a footnote.** See §6.

---

## 4. ROSAME-side convergence

`BudgetMode.CONVERGE` already exists and is wired
(`src/milp/rosame26_budget.py:71-83`; `rosame26_runner.py:394-405` passes a
`stop_check` calling `has_converged`, and `_BestModelTracker` activates only
under CONVERGE). Nothing needs designing — it needs switching on.

Its constants (`rosame26_budget.py:86-110`) were tuned on 3-9-trace folds:
`CONVERGE_WINDOW=40`, `CONVERGE_MIN_IMPROVEMENT=0.002`,
`CONVERGE_PATIENCE=3`, `CONVERGE_MIN_EPOCHS=60`. The tuning rationale is
per-epoch loss noise (median epoch-to-epoch change ~0.065 against a final-vs-best
gap of ~0.16). Longer traces change that ratio, so the constants need
**validation at large L** — tracked in `future-tasks.md`, not blocking.

Note the asymmetry that remains by design: ROSAME stops on a *relative-plateau*
of training loss; the loop stops on *rounds since best*. They are both
"stop when the search stops paying", measured in each family's natural unit
(epochs vs rounds). §2.4 explains why the loop must not borrow ROSAME's
threshold form.

---

## 5. Changes to make

### 5.1 `StopRules.no_improvement_rounds` — no code change

The field, the counter and the `_stop_reason` branch all exist. The large-corpora
runs set it in `run_config.yaml` per L. **Verify only** that
`effective_stop_rules` (`config.py:427`) leaves it untouched — it disables
`stop_on_fixpoint` under a non-frozen pool and nothing else, and the pool is
frozen here.

### 5.2 `subset_size: 4` — config only

`run_config.yaml:46`, from `half`. No code change; `SubsetSizeKind.FIXED` already
parses an int.

### 5.3 Factor columns — the one real code change

Add to `BaselineRunner` (`benchmark/baselines/base_runner.py`) three properties,
written into each result row beside `algorithm`:

| field | values |
|---|---|
| `input_kind` | `symbolic` / `imaged` |
| `paper` | `24` / `26` |
| `uses_milp` | bool |

Per-arm assignment is the table in `ENDPOINT-ANCHORING.md` §1.

Rationale: today an arm is one opaque string, so any question about a *dimension*
means parsing it. `rosame_milp_24_tag` already breaks the pattern with a fourth
component on one arm, and `migrate_arm_names.py` exists precisely because
semantics encoded in strings had to be rewritten on disk. With factor columns the
next rename is cosmetic.

**Not adding `comparable`** — dropped at the user's direction.

Touch points: `base_runner.py` (three properties, defaulted or abstract),
each of the seven runners, `result_schema.py:15` (`BASE_IDENTITY_FIELDS`),
and the row build at `run_fold.py:105`. Land this **before** generating rows so
the schema does not change mid-grid.

### 5.4 Cell timeout

1 h ceiling, configurable. It now does double duty: `StopRules.budget_seconds`
inherits `learning_timeout_seconds` (`run_config.yaml:53`), so the same knob caps
both the loop and the CDPS-fair comparison. With patience in place,
`budget_seconds` should become the *rare* stop reason rather than the usual one —
that is the check in §6.

---

## 6. Pilot gate, before the full grid

Run **one cell** — one domain, L=500, one mask/noise point — and read its
`milp_loop_rounds.json` before committing cluster time:

1. **Is the gap still memoryless?** Plot gap-vs-round. If improvements arrive
   front-loaded as at small N, §3 holds. If they arrive sporadically, the
   patience formula needs refitting on this data instead.
2. **What is the stop reason?** Mostly `no_improvement` means the rule works.
   Mostly `budget_seconds` means the bottleneck is elsewhere — most likely
   V-scoring, see 3.
3. **What does a round cost at scale?** V is scored against the whole frozen
   pool, not the subset, so per-round evaluation is O(L) even at fixed subset
   size. This is the one cost that still grows with L, and no run on disk has
   paid it at L>8. If it dominates, the answer is a cheaper V or a sampled V,
   not a different stop rule.

L=500 rather than 2000 for the pilot: large enough that C(L,4) is unreachable and
the O(L) V cost is visible, small enough to fail fast.

---

## 7. Order of work

1. Factor columns (§5.3) — schema first.
2. `subset_size: 4` + per-L `no_improvement_rounds` in `run_config.yaml` (§5.1, §5.2).
3. Generate one corpus: 5 x 500 = 2500 traces, one domain.
4. **Pilot gate (§6).** Stop here and re-read §3 against real data.
5. Generate the remaining corpora, per-L (`length_min == length_max`).
6. Run the grid.

---

## 8. Decisions already settled

| decision | value | where |
|---|---|---|
| endpoint anchoring | stay faithful to both upstreams, caveat carried | `ENDPOINT-ANCHORING.md` |
| length axis | per-length corpora, `length_min == length_max`; no `generation_info.json` reader | §1 |
| ROSAME convergence | `BudgetMode.CONVERGE`, already wired | §4 |
| relative-improvement threshold on V | rejected — gains are large, V is integer | §2.4 |
| `comparable` factor column | not added | §5.3 |
| simulation results on disk | complete and current, 270 rows x 10 arms x 5 domains; no re-run | — |
| image results on disk | `ROSAME-I_MILP_24` postdates the grounding fix; gaps are in *our* arms (CDPS in 2/5 domains, `PISAM_MILP_SR` in 1/5) | — |
| stale `ROSAME` / `ROSAME-I` labels | 120 files, all `.bak`; the dashboard reads an exact filename, never a glob — inert, leave them | — |
