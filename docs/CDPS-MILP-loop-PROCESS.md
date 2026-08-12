# CDPS-MILP loop — mid-process log

> Written 2026-08-11, mid-session, as a resume point.
> Updated 2026-08-12 (P4 complete).
> **Authority:** `docs/cdps-milp-loop-plan.md` (execution plan) and
> `docs/cdps-milp-denoiser-design.md` (encoding details). This file is a
> *status snapshot*, not a spec — if it disagrees with those, they win.

**Branch:** `cdps-with-milp-implmenetation` (see §5)
**Status:** P1 + P2 **DONE and validated** (`4600b5b76`). P3 **DONE**, both exit
gates passed (`b19fb8d69`). Q6–Q9 **answered** (§4). P4 **DONE** — loop driver
+ benchmark integration, 74 unit tests, both MILP arms verified side by side in
one fold (§4ter), and the `milp_repair_cost` scope collision fixed (§4ter.5).
Next: P5, then P6.

---

## 1. What this is

Replace CDPS's *search* over trajectory repairs with a single CP-SAT solve.
PI-SAM stays the learner, so the safety story survives:

```
noisy observations ──► MILP (one solve) ──► T′ ──► re-mask ──► PI-SAM ──► model
                       (minimal repair)
```

vs CDPS, which explores repairs node-by-node, paying one full PI-SAM run per
node (~1s) — it starves on npuzzle.

Two algorithms, no hybrid:

| key | what | status |
|---|---|---|
| `cdps_milp_single_round` | ONE joint MILP over ALL fold trajectories | **implemented** (P2) |
| `cdps_milp_loop` | homogeneous rounds, each samples a subset | **implemented** (P4) |

---

## 2. P1 + P2 — what was built (done)

### 2.1 Module move
All MILP code moved out of `benchmark/algorithm_adapters/rosame_milp/` into
**`src/pi_sam/plan_denoising/milp_version/`** (git-tracked renames), because
it is now shared between the `rosame_milp*` baselines and our CDPS variant,
and `src/` must not depend on `benchmark/`.

| file | role |
|---|---|
| `encoder.py` | CP-SAT encoding (moved) |
| `encoding_config.py` | `MilpEncodingConfig` + presets `upstream()` / `tag()` / **`cdps_dialect()`** |
| `converter.py` | pddl_plus → vendor `planning_structs`; `GtAnchoring`; `RepeatedArgsInstance` |
| `config.py` | **new** — `CdpsMilpConfig`, the `cdps_milp:` YAML surface, validated |
| `trajectory_extraction.py` | **new** — solved MILP → flips → re-masked T′ |
| `single_round.py` | **new** — the driver |
| `test_cdps_milp.py` | **new** — 12 unit tests |
| `vendor/` | upstream ROSAME code (see `vendor/UPSTREAM.md`) |

### 2.2 The CDPS dialect (the key technical point)
`cdps_dialect()` **drops three constraint families** that the ROSAME upstream
encoding has, because each of them can exclude the ground-truth model:

1. `del ⇒ pre` (paper eq. 18)
2. `PreIsNotEmpty` / `AddIsNotEmpty` (upstream extra, not in the paper)
3. `stepadd + hol ≤ 1` (redundant-add ban)

Each has a dedicated unit test that turns *exactly that one family* back on
via `dataclasses.replace` and shows the model flips SAT → UNSAT — so a
failure names one guilty constraint, not three.

### 2.3 Other pieces
- **GT anchoring**, config-selectable: `init_only` | `all_gt_states`
  (`gt_states_by_obs`, 0-based state index → 1-based encoder time index).
- **`RepeatedArgsInstance`** — widens the vendor's `permutations` grounding to
  `product` so reflexive groundings (`stack(a,a)`) exist. ON for
  `cdps_milp_*`, OFF for `rosame_milp*`. Needed because the vendor *skips*
  repeated-type schemas, which would emit a T′ never constrained on those
  transitions → PI-SAM could raise conflicts on a "feasible" T′, breaking the
  design doc's §4.1 proposition.
- **Re-masking (design §4.2)** — T′ = originals + flips **on observed fluents
  only**; masked slots stay masked. `repair_cost` excludes masked slots.
- **eq16** flag (`+λ_pre·pre[α,ℓ]`, λ=0.4) present but **off** by default.
- **Gurobi** is a stub raising `NotImplementedError` (no license; CP-SAT only).
- Registered as an algorithm; dispatched through `run_fold.run_cdps_phase(...,
  milp_config=...)` so both denoisers share one evaluation path. Artifacts land
  in `<fold>/cdps_milp_single_round/` in CDPS's shape, plus
  `milp_repair_log.json` / `milp_masked_completion.json`.

### 2.4 Validation results (blocksworld smoke, 2 folds, 30 s CDPS budget)

| fold | CDPS cost | CDPS time | MILP cost | MILP time | MILP status |
|---|---|---|---|---|---|
| 0 | 115 | 30 s (timeout) | **91** | 0.226 s | OPTIMAL |
| 1 | 244 | 30 s (timeout) | **161** | 0.323 s | OPTIMAL |

Both folds: `pisam_conflicts_on_feasible = 0`, `n_unmapped_fluents = 0`,
`n_traces_dropped = 0`. **The standing lower-bound check
(`cost(MILP) ≤ cost(best CDPS CFM)`, valid only with `eq16: off`) holds.**
Two orders of magnitude faster, and cheaper repairs.

### 2.5 Tests
`src/pi_sam/plan_denoising/milp_version/test_cdps_milp.py` — **12 tests, all
passing**; `rosame_milp/test_rosame_milp.py` — 5 tests, still passing.
Coverage: the 3 dropped families, minimal-repair exactness, masked-is-free,
repeated-args grounding + binding aggregation, both GT-anchoring modes +
out-of-range indices, config validation + derived settings, flip→patch index
mapping.

---

## 3. Decisions taken this session (P3 design)

All recorded into `docs/cdps-milp-loop-plan.md` §7.

### 3.1 The existing evaluation stack measures the wrong thing for the loop
`evaluate_predictive_power` (`benchmark/evaluation/predictive_metrics.py:31`)
scores against the **GT domain**, over **planner-generated** states,
**one step at a time with no chaining**. The loop needs the opposite on all
three axes: reference = the **frozen original noisy observations**, states =
**ours**, dynamics = **rollout**. Consequences: (a) selecting on GT is
circular and impossible in real image mode; (b) it's a generalization measure
where we need in-sample *fit*; (c) no chaining hides drift; (d) it's far too
slow to call every round.
**It is not replaced** — it stays the headline metric in the results table.
V is the internal selection signal and the P5 anytime y-axis.

### 3.2 Reusable primitive
`CompatibleUPEnv.apply()` (`benchmark/evaluation/upenv_compat.py:87`) already
returns `None` on precondition failure — but it *refuses* to apply effects
through a failed precondition, which is exactly what apply-anyway needs.
**Decision (Q1): implement effect application directly** — ground the schema,
precondition = subset test under CWA, successor `= (s \ del) ∪ add`. ~40
lines, pure, no new dependency, unit-testable with hand-built models per the
§2.1 contract.

### 3.3 Apply-anyway, and why
On `pre ⊄ s_t`: count one `inapplicability_event`, **apply the effects anyway**,
continue. Rationale: PI-SAM is a *safe* learner that deliberately
over-approximates preconditions, so spurious inapplicability is the expected
failure mode of every model we produce. Under stop-on-failure nearly every
trace dies at step 1–2 and V becomes near-constant across candidates —
useless as a selector, and it measures "how early did it break" rather than
model quality. Apply-anyway grades effects *independently* of preconditions.
`w₁ = w₂ = 1`, both counters logged separately so reweighting is post-hoc.

### 3.4 Location — deviates from plan §4
`src/pi_sam/plan_denoising/evaluator.py`, **not** under `milp_version/`,
because P5 uses the same Evaluate to score `cdps` and `rosame_milp` snapshots
and must not drag in CP-SAT.

### 3.5 Execution vs grading (Q3)
- **Grading**: unmasked fluents only (already settled, unchanged).
- **Execution**: a masked slot has *no truth value*, so `pre ⊆ s` and `s \ del`
  are undefined on it. The **rollout never hits this** (s₀ complete, all later
  states model-computed). The **one-step secondary metric does**, and resolves
  it by **skipping transitions whose base state has a masked slot occurring in
  that action's `pre`/`del`**, logging `skipped_transitions`. **V is
  unaffected either way.**

### 3.6 `success_rate` (Q4 — answered: yes)
A transition counts as mismatched if it has ≥1 effect mismatch **or** an
inapplicability event (binary per transition). `success_rate ∈ [0,1]` — the
P5 y-axis.

### 3.7 Correlation check moved P5 → P3, as a second exit criterion (Q5)
`M_best = argmin V` is sound only if lower V ⇒ better model in the GT sense.
If the proxy is uncorrelated the loop optimises noise and P5 draws a rising
curve on a meaningless axis — so this must gate P4, not follow it.
**It is free:** CDPS already emits many `conflict_free_model_{idx}/model.pddl`
per fold from identical data with differing quality — exactly the population
needed — across every finished run. Offline script, no re-runs.
**Pass:** within-fold Spearman ρ ≥ +0.4 between `success_rate` and
`precision_overall` / `recall_overall`, sign-consistent across domains.
**On failure:** fix V (w₁:w₂ off 1:1, select on `success_rate`, add the
one-step term) *before* building P4.

### 3.8 Finding: image-mode s₀ is NOT GT+unmasked today (Q2)
- Simulated mode **enforces** it — `noise_injection.py:7,40`: t=0 untouched,
  no masking, no noise.
- Image mode **does not** — `image_trajectory_handler.py:174-178` turns the
  VLM's frame-0 `unknown` set into masked slots in s₀.

So "init is GT" means two different things per mode. In image mode CDPS treats
a partially-masked state as GT (= *unpatchable*, so those slots are never
resolved) and the MILP admits it as a **hard** state — a hard constraint
derived from an uncertain classification.

**Recommended fix: take s₀ from the problem file's `(:init ...)`** — real GT,
invents nothing. Rejected alternative: forcing UNCERTAIN→false, which invents
values and then pins them unrepairably.
**Deferred out of P3** because it changes s₀ for *all* algorithms and
invalidates every existing image-mode result. Scheduled before P5's benchmark.
P3 only asserts-and-logs.

### 3.9 P6 added to the plan (structural review, post-P5)
Agreed that `src/pi_sam/plan_denoising/` nested inside `src/pi_sam/` is
backwards: the denoiser *consumes* the learner, ROSAME vendor code now sits
two levels deep in a learner package, and the learner-agnostic evaluator has
no correct address there. Honest counter recorded: CDPS *is* PI-SAM-coupled
today (conflict detection is `NoisyLearnerMixin.handle_effects`).
Proposed target — `src/learning/`, `src/denoising/`, `src/model_evaluation/`
with a one-way import rule, as **one mechanical import-only commit after P5**.
**Nothing restructured yet.**

---

## 4. Q6–Q9 — ANSWERED (the decision record that governs P4)

| # | question | decision |
|---|---|---|
| **Q6a** | pool policy — do repaired traces replace their noisy originals? | `pool_policy: frozen \| replace \| frozen_with_hints`, **default `frozen`**. All three implemented. `replace` **auto-disables** dedup and the fixpoint stop rule (the pool is no longer a fixed set, so neither is well-defined). |
| **Q6b** | "with/without replacement" — the ambiguity the user flagged | Resolved as *neither of the naive readings*: a subset is a **set** (no repeated trajectory inside one round), and the same subset **may** be drawn again in a later round — but only if `M_best` has changed since. |
| **Q6c** | dedup key | **`(subset, M_best)`**. Skip a round only when that exact pair was already solved. This is what makes the same subset re-drawable *and* gives an exact fixpoint rule (`math.comb(len(pool), subset_size)` admissible pairs per incumbent). |
| **Q6d** | determinism | **Pin** `random_seed` **and** `num_workers` in `encoder.solve()`. CP-SAT's portfolio is otherwise nondeterministic across worker counts, which would make a round's own dedup key lie. |
| **Q7a** | budget | The loop, `single_round` **and** `rosame_milp` all inherit the **CDPS per-fold timeout**, not the plan's stale literal `3600`. That is what makes the head-to-head fair. |
| **Q7b** | is online V charged? | Yes, and it is **negligible**: measured ~10 ms/round. Recorded rather than engineered around. |
| **Q8a** | incumbent update rule | Greedy **strict** `V_r < min_V`. **No tolerance band** — §4bis.5(1) shows accuracy rises smoothly with the gap, so a band forgoes real improvement. |
| **Q8b** | the vacuous exit criterion | The plan's "M_best ≥ round-1 on V" is true by construction. **Replaced** by reading the per-round logs (`rounds_improved`, `rounds_tied`, `best_round`, `stop_reason`), which say what actually happened. |
| **Q8c** | tie-break | Option **C — incumbent wins**, which is exactly what strict `<` already does. No extra machinery; tie *events* are counted and logged so a degenerate run is visible. |
| **Q9a** | subset size | `half` = `max(2, ceil(n/2))`, the default. |
| **Q9b** | how it is configured | **Named policies** (`half \| all \| <int>`) — **not** the proposed `ast.parse` expression whitelist. A three-value enum needs no parser. |

Q6's original framing ("the loop has nothing to show at `num_trajectories: 3`")
still stands as an *experiment-design* note, not a code question: at n=3,
`half` → m=2, every subset is 2/3 of the data and the loop is a jittery
`single_round`. **P5 must run the loop at ~10–20 trajectories** for its
behaviour to be distinguishable at all. The smoke runs deliberately use n=3
because they test wiring, not behaviour.

*(§7.4 withdraws the bolded sentence. n=3 is genuinely cramped — 4 rounds
against `C(3,2)=3`, then fixpoint — but "cramped" is not "indistinguishable":
a later round still beat round 1 in 3 of 5 real blocksworld folds at n=3.
Meanwhile npuzzle is indistinguishable at **every** n tested. The requirement
was on the wrong axis.)*

---

## 4bis. P3 — DONE (evaluator + both exit gates passed)

Authorised by **D1 = (a)**, **D2 = recommendation**, **D3 = commit first**.

### 4bis.1 What was built

`src/pi_sam/plan_denoising/evaluator.py` (~470 lines) + `test_evaluator.py`
(**17 tests, all green**).

```python
observations_reconstruction_score(domain: Domain,
                                  observations: Sequence[Observation],
                                  weights: Optional[EvaluationWeights] = None) -> EvaluationResult
```

- **D1 (a):** the input is a parsed `Domain`, so the *written* `model.pddl`
  artefact is what gets scored — one code path for CDPS CFMs, PI-SAM output
  and every ROSAME baseline alike.
- **D2:** every `TraceEvaluation` carries both `v_raw` and
  `v_per_transition`; `EvaluationResult` aggregates both.
- `V = w₁·effect_mismatches + w₂·inapplicability_events`, w₁ = w₂ = 1.
- The reference is the frozen **original (noisy) observations**, unmasked
  slots only. No GT domain, no planner, no simulator.

### 4bis.2 Three upstream facts that shaped the implementation

Verified by reading `pddl_plus_parser`, not assumed:

1. **`grounded_effect._apply_discrete_effects`** — delete effects `discard()`
   the predicate. States are **positive-only / CWA**.
2. **`grounded_precondition._validate_predicates_hold`** — applicability is a
   *substring* test, `condition.untyped_representation in state.serialize()`.
   Since `(on a b)` is a substring of `(not (on a b))`, CWA-completing the
   rollout state with explicit negative literals would have silently satisfied
   positive preconditions. **This reversed the planned design** — the rollout
   state is now built by `_project_to_positive_state`, and there is a NOTE in
   the file so nobody "fixes" it back.
3. **`Operator.apply(prev, allow_inapplicable_actions=True)`** already *is*
   apply-anyway, with the library's own precondition semantics.
   → **Q1's recommendation is superseded**: the ~40 hand-rolled lines of
   effect application I proposed earlier were not written; we call upstream.

Also guarded: `mask_state` leaves `is_masked=True` predicates holding their
**true** polarity, and `GroundedPredicate.copy()` silently **drops
`is_masked`**. The evaluator never reads a masked slot's polarity and never
grades a masked slot; both are pinned by tests.

### 4bis.3 Exit A — V(GT model) == injected-noise count

**150 / 150 folds exact**, blocksworld, over five cells:

| cell | folds matching | V on fold 0 |
|---|---|---|
| `mask=0.0 noise=0.0` | 30/30 | 0 |
| `mask=0.0 noise=0.1` | 30/30 | 46 |
| `mask=0.0 noise=0.2` | 30/30 | 90 |
| `mask=0.01 noise=0.2` | 30/30 | 91 |
| `mask=0.1 noise=0.2` | 30/30 | 76 |

`inapplicability_events = 0` and `init_masked_slots = 0` everywhere,
confirming the simulated path leaves t=0 GT and unmasked as documented.

### 4bis.4 Exit B — V ↔ GT-metrics correlation (**gates P4 — PASSED**)

Every candidate model on disk per fold (`conflict_free_models/*/model.pddl`,
`learned_domain_PISAM_*`, `baseline_models/*`, `cdps_anchored/*`) scored twice:
V on the frozen originals, and syntactic precision/recall vs
`domain_reference.pddl` (f1 = harmonic mean of the two `mean` fields).

```
folds scored              : 120        (2519 (fold, model) pairs)
mean   Spearman rho       : -0.862
median Spearman rho       : -0.883
rho < 0                   : 120/120 folds
rho <= -0.5               : 119/120 folds
argmin-V picks the GT-best: 109/120 folds
mean f1 regret of argmin-V: 0.0024      max: 0.1084
```

### 4bis.5 Three limits of V that P4/P5 must respect

All three fall out of the exit-B data and are **inherent to a GT-free
metric**, not defects:

1. **V's failure mode is exact ties, not small gaps.**
   *(Corrected. An earlier revision of this section claimed "V has a resolution
   floor of a few points" and asked for a tolerance band. That was inferred from
   the 11 argmin-V misses alone — a selection-effect view, since argmin is by
   construction the decision most exposed to near-ties. The full pairwise
   measurement below does not support it.)*

   Over all **19,852** within-fold pairs with distinct V, accuracy rises
   *smoothly* with the gap — there is no threshold:

   | V gap ≥ | pairs | P(V correct) | mean f1 gain |
   |---|---|---|---|
   | 1 | 19,852 | 0.898 | +0.115 |
   | 5 | 17,588 | 0.912 | +0.125 |
   | 10 | 15,118 | 0.930 | +0.136 |
   | 20 | 10,883 | 0.954 | +0.160 |
   | 50 | 3,725 | 0.975 | +0.211 |

   A 1-point gap is already right ~90% of the time. A band would be actively
   harmful: the 4,734 pairs in gap `[1,10)` are decided correctly 79.5% of the
   time for a **mean f1 gain of +0.047** — refusing them forgoes real
   improvement. → **no tolerance band**; greedy `V < V_best` is correct.

   What *does* need handling is exact ties — but the tie is **smaller than an
   earlier revision of this file claimed**. Those numbers counted textually
   identical models as separate candidates: CDPS writes several
   `conflict_free_model_*/model.pddl` per fold that are byte-for-byte clones,
   and a "tie" between a model and its own copy is not a decision. Deduplicating
   by PDDL-text hash over the same 2519 (fold, model) pairs:

   | measure | folds | mean tie size | max |
   |---|---|---|---|
   | raw argmin tie (clones counted) | 150/150 | 6.7 | 13 |
   | **argmin tie, distinct models** | **117/150** | **3.4** | **5** |

   Of those 117, the tied models actually differ in f1 (>0.01) in **96/150**
   folds — mean spread **0.0367**, max **0.1485**.

   So the tie is real and consequential in ~2/3 of folds, but it is a choice
   among ~3 genuinely different models, not ~7. → the loop still needs a
   **deterministic GT-free tie-break**, and Q8's exit criterion must not rest on
   "V improved" alone. (Q8c settled this as *incumbent wins*, which strict `<`
   already gives for free.)
2. **V cannot rank at all when the data is clean.** In the
   `noise=0.0` cell *every* candidate scores V=0 while f1 spreads 0.969–1.000.
   That residual is generalisation beyond the observed data, which nothing
   computed *on* that data can see. The loop degenerates to its tie-break
   under zero noise — acceptable, but it must be stated, not discovered later.
3. **V is not comparable across folds.** Within-fold rho −0.862 vs pooled
   rho −0.454; normalising recovers much of it
   (`rho(v_per_transition, f1) = −0.660`) — note *much*, not all.

   *(Scope corrected, §7.1bis. This point previously ended "cross-fold
   aggregation in P5 must use `v_per_transition`, never `v_raw`", which
   over-reached: P5's headline y-axis is `success_rate`, already a bounded
   ratio, so it never touches `v_raw`. The accurate statement is that **V is a
   within-fold ranking signal** — it is what the loop selects with, not
   something to pool. The same non-comparability also holds **across
   trajectory counts**, since `v_raw` sums over transitions.)*

---

## 4ter. P4 — DONE (loop driver + benchmark integration)

Split into **P4a** (the driver, in `src/`) and **P4b** (wiring, in `benchmark/`).

### 4ter.1 P4a — what was built

| file | lines | role |
|---|---|---|
| `milp_version/loop.py` | 902 | `run_loop` — the round loop, samplers, dedup, stop rules, per-round log + per-round model (§7.2) |
| `milp_version/model_prior.py` | 171 | `LearnerDomain` → vendor `ObservationM`, the reference-model channel |
| `milp_version/config.py` | *extended* | the loop-only YAML keys (`sampler`, `subset_size`, `learner_input`, `pool_policy`, `co_sample_conflicts`, `w_prior`, `seed`, `stop:`, `eval:`) |
| `milp_version/encoder.py` | *edited* | Q6d — `random_seed` + `num_workers` pinned |
| `milp_version/test_loop.py` | 672 | **65 tests** across 11 classes (hash, samplers, dedup, stop rules, budget, learner input, subset GT, reporting, round models, subset size, prior) |

**73 tests green** in `milp_version/` (61 loop + 12 P2).

### 4ter.2 Five design decisions taken inside `loop.py`

1. **Structural model identity.** The dedup key needs a stable hash of
   `M_best`, and `to_pddl()` is **not** stable — it rebuilds the
   `:requirements` line from a **set** on every call, so the same model hashes
   differently across calls. The hash is therefore computed from the model's
   *structure* (sorted action → sorted pre/add/del literal names), not its text.
   Without this, dedup silently never fires and the fixpoint rule never trips.
2. **`_TraceCache` — convert once per fold.** pddl_plus → vendor
   `planning_structs` conversion is not free and the pool is frozen by default,
   so each trace is converted once and reused across rounds.
3. **V is evaluated against ALL original observations, never the round's
   subset.** Scoring a candidate on the data it was fitted to would reward
   overfitting the sample and make rounds incomparable — the whole point of
   V is that it is one fixed yardstick.
4. **Contradictory observations are dropped, not fatal.** A trace the converter
   rejects is skipped with a log line and counted; one bad trace must not kill
   a fold that has nine good ones.
5. **Exact fixpoint via `math.comb`.** With a frozen pool, the number of
   admissible `(subset, M_best)` pairs for a fixed incumbent is exactly
   `C(len(pool), subset_size)`; once all are solved, no further round can do
   anything, so the loop stops with `stop_reason = "fixpoint"`.

### 4ter.3 P4b — benchmark integration

- **Algorithm key beats the YAML `variant:` key.** `milp_config_for(key, cfg)`
  pins `MilpVariant` from the *selected algorithm key* via
  `dataclasses.replace`. This is what lets **one** `cdps_milp:` block serve
  **both** arms in a single run — a single `variant:` field cannot express
  "run both". The YAML `variant:` is consequently ignored for these two keys.
- **`cdps_budget_seconds` was one parameter doing two jobs.** The fold's
  denoiser budget is the fallback for two *independent* caps: `stop.budget_seconds`
  (the whole loop) and `time_limit_seconds` (one solve). Split into
  `loop_budget` / `solve_limit`; previously a 600 s fold budget was handed to a
  single solve.
- **`cdps_family_names()`** (`benchmark/algorithms.py`) — the run banner and
  `run_params["algorithms"]` previously built the same label list two different
  ways (one off a hardcoded constant, one off `cdps_milp_algorithm_name`). With
  a second MILP arm they would have drifted; both now derive from this one
  helper. Baselines stay the caller's business (the two callers read different
  attributes: `.name` vs `.display_name`).
- **`resolve_algorithms` is now a 5-tuple**
  `(cdps, cdps_anchored, milp_single_round, milp_loop, baselines)`; all three
  call sites updated.
- **Per-arm artefact dirs** — `<fold>/cdps_milp_single_round/` and
  `<fold>/cdps_milp_loop/`, so both arms run in the same fold without
  overwriting each other. The dispatch is a data-driven loop over
  `[(key, subdir, selected)]`, not two copies of the same 15-line call.
- **`run_config.yaml`** — the `cdps_milp:` block documents every loop key inline.

### 4ter.4 Verification

Smoke: blocksworld, `mask=0.01 noise=0.2`, 1 fold, 3 trajectories, both arms,
60 s budget, `max_rounds: 5` → **both arms completed in one fold in 60.2 s**,
2 result rows, report written. `fold_result.json`:

| | `CDPS_MILP_SR` | `CDPS_MILP_LOOP` |
|---|---|---|
| learning time | 0.315 s | 0.679 s |
| `precision_overall` / `recall_overall` | 0.94 / 1.00 | 0.92 / 0.96 |
| `milp_repair_cost` (fold-wide) | 91 | `None` — by design, see §4ter.5 |
| `milp_loop_best_round_repair_cost` / `..._subset_size` | `None` | 83 over **2** of 3 traces |
| loop keys | all `None` | rounds 5, solved 5, improved 2, tied 1, best_v 91.0, best_round 2, stop `max_rounds` |

Both arms share one result vocabulary; the 8 loop-only keys are populated for
the loop and `None` for the single round, so one table holds both.

### 4ter.5 `milp_repair_cost`'s two scopes — FIXED

**The problem as found.** `milp_repair_cost` meant two different things in one
column. For `single_round` it was the cost over **all** observations (91 above);
for the loop, the winning round's cost over its **subset** (83 over 2 of 3
traces). `83 < 91` reads as "the loop repaired more cheaply" when it only
repaired *less*. Under `pool_policy: replace` there was a third meaning: a round
extracts against the already-repaired pool, so its cost is an *increment*.

**Decision (user, 2026-08-12): option B — the fold-wide column stays
single-round-only.** The field's only job is the design §7.1 check
`cost(MILP) <= cost(best CDPS CFM)`, which requires one solve that certified
every trace jointly. The loop structurally never performs one, so it reports
`None` rather than a number that looks like it can carry the check.

The alternative considered and rejected was to recompute the loop's cost over
the full pool (original → `state.repaired`). That is dishonest the other way:
traces no round ever touched would contribute 0, and the union of per-round
repairs was never jointly certified by any single solve.

**What changed.**

| | |
|---|---|
| `LoopResult.repair_cost` | renamed → `best_round_repair_cost`; `best_round_subset_size` added next to it, so the scope is never implicit |
| `LoopResult.as_report` | emits `repair_cost: None`, `best_cost: None` |
| `run_fold._milp_specific` | new `milp_loop_best_round_repair_cost` / `..._subset_size`; `milp_repair_cost` unchanged for the single round |
| `RoundLog.repair_cost` | unchanged (per-round, in `milp_loop_rounds.json`) — now carries a comment stating its scope |

Nothing is lost: `milp_loop_rounds.json` already stored per-round `repair_cost`
and `subset`. Covered by `test_the_fold_wide_cost_keys_stay_empty`.

### 4ter.5bis One known issue, deliberately not fixed

1. **`--resume` regression.** Adding `run_cdps_milp_loop` to `run_params` makes
   resuming a *pre-existing* experiment dir report a spurious conflict on that
   key (absent vs `False`), because `RESUME_IGNORED_PARAMS` is only
   `{timestamp, num_trajectories_list, gt_rate_percentages, folds}`. Left
   consistent with the precedent set when `run_cdps_milp` and
   `run_cdps_anchored` were added, rather than special-cased.

### 4ter.6 Two upstream findings surfaced during P4

- **Depot is broken for every algorithm that loads observations without a
  problem file**, which includes CDPS and both MILP arms. Root cause:
  `GroundedPredicate` violates the `__eq__`/`__hash__` contract — `__eq__` walks
  the type hierarchy, `__hash__` is over `str` — so two spellings of one fluent
  compare equal but hash apart, the membership test in
  `ground_all_predicates_in_state` probes the wrong bucket, and CWA-completion
  appends a contradictory negative beside the present positive. `96353c4ff` did
  **not** fix this; it moved which call site suffers. §7.8 has the measurement
  and the correction to what this section previously claimed. **Fixed since**, by
  `normalize_predicate_types_in_state` — §7.8's Resolution. The
  `__eq__`/`__hash__` violation itself is still there; it lives in
  `pddl_plus_parser` and is now merely unreachable from this repository's states.
- **`model_prior._binding`'s distinctness guard is unreachable.**
  `Predicate.signature` is a **dict**, so `(on ?x ?x)` collapses to arity 1
  upstream and the guard can never see repeated parameters. Harmless, but it
  should not be read as protection that exists.

---

## 5. Repo state (branch `cdps-with-milp-implmenetation`)

- **Committed — `4600b5b76`** (P1 + P2, 32 files, +2441/−252): the
  `milp_version/` move, the CDPS dialect, `cdps_milp_single_round`,
  and the `docs/` + `CLAUDE.md` updates.
- **Committed — `b19fb8d69`** (P3): `evaluator.py` +
  `test_evaluator.py` — `observations_reconstruction_score`, 17 tests.
- **Committed — `fa4c0fc1c`** (P4, 15 files): `milp_version/loop.py`,
  `milp_version/model_prior.py`, `milp_version/test_loop.py`, the pinned CP-SAT
  seed/workers, and the benchmark integration for both MILP arms.
- **Committed — `641cb97a3`** (unrelated, kept separate): the dashboard legend
  hover-highlight. It predates this work but earns its place beside it — every
  non-CDPS series shares one dash pattern, so colour is the only discriminator,
  and the two MILP arms take the baseline count past what is readable by eye.
- **Smoke artefacts — deleted** (decision, 2026-08-12): the `milp-dispatch-smoke`
  and `milp-loop-smoke` manifests and their `running_results/` cells. They were
  1-fold / 3-trajectory / 60 s throwaways whose only informative numbers are
  already in §4ter.4, and `finished_run_configs/` should hold real runs only.
- **Name collision — resolved.** The new GT-free entry point is
  `observations_reconstruction_score`; `evaluate_model` stays the name of the
  **GT-based** reporting function in
  `benchmark/experiment_running_helpers/evaluation.py:62` (syntactic
  precision/recall + `problem_solving` + predictive power, one call site at
  `result_builders.py:61`). The two answer opposite questions and importing the
  wrong one in the loop would select on ground truth and **silently** invalidate
  the run, so the distinction is spelled out in the new function's docstring.
  The dead `evaluate_model` import at `run_fold.py:31` was dropped.

`source venv11/bin/activate` to run the tests.

---

## 6. Next actions, in order

~~1. **P3** — evaluator + unit tests.~~ **done** (§4bis.1, 17 tests green)
~~2. **P3 exit A** — V(GT model) == injected-noise count.~~ **done**, 150/150
~~3. **P3 exit B** — V↔GT-metrics correlation. **Gates P4.**~~ **PASSED**,
mean rho −0.862
~~4. Answer **Q6–Q9**.~~ **done** (§4)
~~5. **P4a** — loop driver + 61 unit tests.~~ **done** (§4ter.1)
~~6. **P4b** — benchmark integration + smoke run.~~ **done** (§4ter.3–4)
~~7. **Fix `milp_repair_cost`'s two scopes.**~~ **done** (§4ter.5, option B)
~~8. **Decide the smoke artefacts.**~~ **deleted** (§5), P4 committed.

Remaining, in order:

1. **P5** — anytime performance profile. Decisions in §7.1, steps in §7.1ter.
2. In parallel/background — the **eq16 on/off comparison** on
   `single_round` (already authorized under P2; cheap; it is the
   "does PI-SAM cover for Eq. 16" experiment, a claim in its own right).
3. **P6** — the `src/` structural review (§3.9), one mechanical import-only
   commit after P5.

**Deferred, decoupled from P5** (decision D2, 2026-08-12):

- **Image-mode s₀ fix** (§3.8). Previously listed as gating P5's benchmark. It
  is not: P5 runs on simulated data, where s₀ is already GT+unmasked, so the
  bug cannot touch the anytime result. It does change s₀ for *all* algorithms
  and invalidates existing **image-mode** results, so it needs its own run —
  which is a reason to schedule it separately, not a reason to block on it.

---

## 7. P5 — anytime performance profile (in progress)

### 7.1 Seven decisions taken before implementation (2026-08-12)

Each was checked against the tree before being put to a decision, which is how
D1's blocker and D2's non-blocker were told apart.

| | question | decision |
|---|---|---|
| **D1** | The loop was to be run at ~10–20 trajectories; both P5 domains ship exactly **10** training problems, so `num_trajectories` caps at 8. | The 10–20 note was an inference of mine, not a requirement — see §7.1bis. The loop must work at 3–8 (backfilled onto existing cells) **and** at much larger n later, with no special-casing. Two work items fall out: an `--algorithm` selector for `backfill_cdps.py`, and verification across n. |
| **D2** | Does the image-mode s₀ fix gate P5? | No. Decoupled; recorded in §6. |
| **D3** | `rosame_milp` emits no per-epoch snapshots, so it cannot sit on an anytime axis. | Build the per-epoch snapshot callback — the plot is 3-way (CDPS / loop / ROSAME), not 2-way. |
| **D4** | Round models are not persisted, so no round can be re-scored offline. | Write `round_{i}/model.pddl`. **Done — §7.2.** |
| **D5** | Is the harness a reader, or a reader plus a scorer? | Both. CDPS emits `model.pddl` but no `success_rate`, so the harness must score what it reads. |
| **D6** | Which ablation values run? | Explicit config values, chosen per run, exactly like the benchmark runner. Never a default full sweep. |
| **D7** | Harness location. | Accepted as planned; the name may change later. |

### 7.1bis Correcting two of my own constraints

Both had been written into this document as rules. Neither survived being
checked.

- **"Run at ~10–20 trajectories."** The basis was real but narrow: at n=3 the
  subset is `max(2, ceil(3/2)) = 2`, so there are only `C(3,2) = 3` distinct
  subsets and the loop hits its fixpoint after ~3 rounds. That describes a
  **finding to report**, not a precondition to satisfy. At n=8 it is `C(8,4) =
  70`, which is ample. The loop is correct at both; only its room to explore
  differs, and saying so is more useful than hiding the small-n case.
  *(Superseded by §7.4, which measured this: the `C(3,2)=3` fixpoint is real —
  n=3 reaches it in 4 rounds — but n is not what decides whether the loop beats
  its first round. The domain is. Blocksworld's gains **peak at n=8**, the
  largest real n available, so the 10–20 advice pointed away from where they
  are.)*
- **"Aggregate on `v_per_transition`, never `v_raw`."** Stated
  unconditionally, it was too broad. The anytime plot's y-axis is
  `success_rate`, which is `1.0 - mismatched/num_transitions`
  (`evaluator.py:114`) — already a bounded ratio, so the `v_raw`
  non-comparability never reaches it. The sharper rule: **V is a within-fold
  ranking signal.** It is what the loop selects with, not something to pool.
  This has a consequence that matters directly for D1: `v_raw` sums over
  transitions, so V at n=3, n=8 and n=50 are **different quantities**, and
  `v_per_transition` only partly repairs that (ρ recovers −0.454 → −0.660, not
  to the within-fold −0.862 of §4bis.4).

### 7.1ter The seven steps

Ordered so that nothing has to be re-run later: everything that changes what a
run *emits* lands before the run itself.

| # | step | why here |
|---|---|---|
| 1 | **D4** — `round_{i}/model.pddl` per round | Must precede any run, or those runs cannot be re-scored. **Done, §7.2.** |
| 2 | **D1** — `--algorithm` selector in `backfill_cdps.py` | Unlocks the existing 3–8-trajectory cells without regenerating data. `run_cdps_phase` already takes `milp_config`, so this is CLI wiring, not new machinery. **Done, §7.3.** |
| 3 | **D1** — verify the loop at small and large n | Confirms no special-casing is needed; produces the small-n finding of §7.1bis as data. **Done, §7.4** — and it corrected the framing: the deciding variable is the domain, not n. |
| 4 | **D3** — per-epoch snapshot callback in the ROSAME adapter | Makes the plot 3-way. Changes what a run emits, so it precedes the run. **Done, §7.5.** |
| 5 | **D5/D7** — the anytime harness: snapshot reader + offline scorer + curves | Consumes 1 and 4. Offline, so it can be iterated on after the run. **Done, §7.6.** |
| 6 | **D6** — config-driven ablation selection | Determines *which* cells the run covers. **Done, §7.7.** |
| 7 | the benchmark run, with the eq16 on/off comparison riding along | Everything above is a precondition. |

### 7.2 D4 — per-round models (done)

`run_loop(fold_work_dir=...)` now also writes
`milp_loop_round_models/round_{i}/model.pddl`, a sibling of
`milp_loop_rounds.json`: the log says what each round scored, the directory
holds the model it scored.

Every round that produced a model gets a file, **including rounds that lost to
the incumbent**. The loop keeps only the winner, so without this the losers are
gone — and an anytime curve is a statement about what was on the table at each
point in time, not only about what survived.

Verified end-to-end on a real fold (blocksworld, `mask=0.1 noise=0.2`, n=8, 5
rounds), because the unit tests cover `save_round_model` in isolation and
cannot cover the wiring `fold_work_dir` → `_LoopState.round_models_dir` → the
write site in `_learn_and_score`:

- all 5 rounds had a model, all 5 files present, no gaps and no extras;
- rounds 4 and 5 shared round 2's `model_hash` and wrote byte-identical text;
- **the winner's file was not byte-identical to the returned model** — the diff
  is `:requirements` ordering only, and re-parsing the file yields identical
  preconditions and effects for all four actions.

That last point is the `model_hash` docstring's claim — `to_pddl` is not a
stable function of the model — now demonstrated on real data rather than
asserted. Two consequences for the P5.5 scorer: the artifact is faithful (it
re-parses to the model that was scored), and **model identity must be read from
the log's `model_hash`, never from the file text**.

### 7.3 D1 — backfilling the MILP arms (done)

`backfill_cdps.py --algorithm {cdps_anchored,cdps_milp_single_round,cdps_milp_loop}`,
with `--milp-config` pointing at a run_config (`shared.cdps_milp`), a file
holding just that block, or the bare block. `cdps_anchored` stays the default,
so every existing invocation is unchanged.

**The arm decides where its input comes from.** That is the substance of the
change, not the flag:

- `cdps_anchored` rebuilds init+final-anchored trajectories from the cell's
  frozen degraded files plus the data dir's GT trajectories.
- the MILP arms consume the frozen degraded files **unchanged**. That is what
  makes their rows comparable to the CDPS row sitting beside them in the same
  `fold_result.json`: same observations, same GT map, only the denoiser differs.
  Reusing the anchored staging path would have produced rows *labelled*
  `CDPS_MILP_*` that were actually anchored — a different algorithm, silently
  non-comparable, and it would have broken design §7.1's
  `cost(MILP) <= cost(best CDPS CFM)` check.

`anchor_endpoints` is therefore set from the arm, not hardcoded: anchoring is a
property of the trajectories, not of the denoiser.

**`gt_rate != 0` is refused, not guessed.** A cell records problem names,
masking files and test problems — but *not* which state indices had GT
injected. Choosing a different set than the original run would protect
different states while claiming to be the same experiment. All 6405 existing
cells are `gtrate0`, and `SimulatedDataSource.prepare` raises for `gt_rate > 0`,
so the refused branch is unreachable today; it exists so that it stays
unreachable when that changes.

Verified on a throwaway copy of a real blocksworld cell (`mask=0.1 noise=0.2`,
n=3), running both MILP arms into a cell that already held four rows:

- six distinct rows coexist — `ROSAME`, `ROSAME_MILP`, `ROSAME_MILP_TAG`,
  `CDPS_ANCHORED`, `CDPS_MILP_LOOP`, `CDPS_MILP_SR` — confirming the row-name
  check uses the *computed* arm-suffixed name, not a constant;
- the re-serialised inputs are set-identical to the cell's frozen originals for
  all three problems (the byte difference is predicate ordering only);
- P5.1's `milp_loop_round_models/round_{1..4}/model.pddl` landed in the
  backfilled cell, so a backfilled loop row is re-scorable exactly like a live
  one;
- the two scopes of §4ter.5 held: `CDPS_MILP_SR` reported
  `milp_repair_cost=75` (fold-wide, 3 traces) with every loop key `None`;
  `CDPS_MILP_LOOP` reported `milp_repair_cost=None` and
  `milp_loop_best_round_repair_cost=69` beside `..._subset_size=2`.

One pre-existing wart left alone: `--dry-run` on `cdps_anchored` still writes
its anchored trajectories into the cell before reporting that it would do
nothing. It is unchanged from before this step, and the MILP arms' prep is
read-only, so their dry runs are genuinely dry.

### 7.4 D1 — the loop at small and large n (done)

The question was whether the loop needs special-casing at either end of D1's
range ("3–8 now, many more later"). **It does not** — but measuring it replaced
two of my own claims with better ones, including one from §7.1bis.

**Scaling.** Blocksworld, one loop run per size, default stop rules, 600 s
budget. Sizes above 8 are pooled: every distinct trajectory across the cell's
folds, deduped by content hash — folds share problems, and counting those copies
would fake a large pool out of a small one.

| n | m | C(n,m) | rounds | best round | seconds | stopped by |
|---|---|---|---|---|---|---|
| 3 | 2 | 3 | 4 | 1 | 0.5 | **fixpoint** |
| 4 | 2 | 6 | 6 | 1 | 0.6 | no_improvement |
| 5 | 3 | 10 | 8 | 3 | 1.2 | no_improvement |
| 8 | 4 | 70 | 11 | 6 | 3.0 | no_improvement |
| 12 | 6 | 924 | 6 | 1 | 1.9 | no_improvement |
| 20 | 10 | 184 756 | 6 | 1 | 3.4 | no_improvement |
| 40 | 20 | 1.4·10¹¹ | 6 | 1 | 8.4 | no_improvement |
| 80 | 40 | 1.1·10²³ | 6 | 1 | 18.0 | no_improvement |

Every size produced a model; time grows roughly linearly in n for a fixed round
count; `math.comb` returning a 24-digit integer at n=80 is harmless because
Python integers are unbounded, so the fixpoint rule needs no guard. n=3 is the
only size that reaches its fixpoint, and it does so in 4 rounds against
`C(3,2)=3` — §7.1bis's inference, now measured.

**At large n the loop is patience-bound, never budget-bound**: 18 s of a 600 s
budget at n=80. That looks like under-exploration, so it was tested rather than
assumed — n=20 re-run with `no_improvement_rounds: null, max_rounds: 40`. Round
1 scored V=480 and **no round in the next 39 beat it** (V was 480 in 25 of 40
rounds, worse in the rest). The default patience of 5 is not costing quality
here; the V landscape is flat.

**What actually decides whether the loop beats its own first round is the
domain, not n.** Running every real fold of a cell (5 folds per n, n=3…8):

| cell | n=3 | n=4 | n=5 | n=6 | n=7 | n=8 | total |
|---|---|---|---|---|---|---|---|
| blocksworld `sim_run__mask=0.1__noise=0.2` | 3/5 | 3/5 | 3/5 | 1/5 | 1/5 | **5/5** | **16/30** |
| npuzzle `simulation-cluster-run-cpu256-10min__mask=0.1__noise=0.2` | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | **0/30** |

(cells are "folds where some later round improved on round 1". Neither row comes
from `simulation-final-run__*`, the authoritative set — §7.8.) Blocksworld's
mean V gain over round 1 peaks at n=8 — 6.6 % mean, 20.8 % best fold — which is
the *largest* real n available, not the smallest. npuzzle never improves at any
n: its traces exercise the same four move actions in the same shape, so every
subset yields the same model and V is constant across rounds.

Three corrections to earlier claims in this document, all mine:

1. **§7.1bis said the small-n case was the one worth reporting.** The sharper
   statement is that n is not the deciding variable at all. The advice "run at
   ~10–20 trajectories" was not merely narrow — it pointed away from where the
   measured gains are. The existing 3–8-trajectory cells are a *good* operating
   point for blocksworld, and 10–20 would not have rescued npuzzle.
2. **A single fold at each size suggested headroom shrinks as n grows**
   (best_round 3 at n=5, 6 at n=8, then 1 at every n≥12). Across 5 folds per
   size that reading does not hold: n=8 is the best case, not the worst. The
   n≥12 rows are also pooled across problems, so they are weaker evidence than
   the real folds and should not carry a conclusion on their own.
3. The loop being "correct at both ends" is now measured rather than argued, so
   **no special-casing is added**.

**Depot is excluded, by a live code defect rather than by the loop.** Every depot
trace is rejected with `State 1 maps multiple grounded predicates to the same
proposition(s) ['clear_p1', 'clear_p3'] — the observation is contradictory`. With
no encodable trace the loop raises `No usable traces for the MILP loop`, which is
the correct behaviour — it cannot repair what it cannot encode. This affects the
whole MILP family, single round included, and CDPS as well.

The sentence that stood here originally called this "a pre-existing data bug"
already fixed in `96353c4ff`. Both halves were wrong; §7.8 measures it. It is not
a property of the cell this sweep happened to read, so re-running the sweep
against `simulation-final-run__mask=0.1__noise=0.2` would reproduce it exactly.

**The exclusion is now lifted** — §7.8's `normalize_predicate_types_in_state`
landed and depot encodes. The headroom table above is therefore missing a depot
row that is measurable rather than impossible; P5.7 supplies it. Nothing else in
§7.4 changes, since depot never contributed a number to it.

### 7.5 D3 — ROSAME per-epoch snapshots (done)

Written with the code but recorded here late; this section is retrospective.

CDPS gets intermediate models for free — one per round, and §7.2 now keeps the
losers too. ROSAME produces exactly one model, at the end, so on an anytime axis
it would be a single point no matter how long it trained. `SnapshotWriter`
(`benchmark/algorithm_adapters/anytime_snapshots.py`) supplies the missing half:
`snapshot_{i:04d}.pddl` every `interval` steps plus a `snapshots.json` index,
under `anytime_snapshots/{arm}/`. Rendering is injected as a callable, so the
writer knows nothing about ROSAME and the ROSAME adapter owns `rosame_to_pddl`.

**The recorded clock excludes the cost of snapshotting.** That is not fussiness.
Measured on real folds: one blocksworld epoch costs ~6.4 ms against ~0.8 ms per
snapshot, and depot 8.9 ms against 1.5 ms — **12 % and 17 %**. Charging that to
the learner would shift ROSAME's curve right by more than the gaps the plot
exists to show, and it would penalise ROSAME exactly in proportion to how
densely *we* chose to measure it. `elapsed_seconds()` therefore subtracts
accumulated overhead, and `snapshots.json` carries both the total and a
`timing_note` saying so.

**The asymmetry with CDPS is left in place rather than modelled away.** CDPS and
the loop do *not* subtract the cost of writing their own models. Those writes are
milliseconds against solves measured in seconds, so the correction would be
noise; adding it would mean maintaining two timing conventions to move a curve by
less than its line width. Stated here so a later reader finds it recorded rather
than discovers it.

### 7.6 D5/D7 — the anytime harness (done)

`benchmark/evaluation/anytime/`: `checkpoints.py` (reader), `score.py` (offline
scorer), `curves.py` (profiles + figure), `run_anytime.py` (CLI),
`test_anytime.py` (9 tests). Run it at a cell or a single fold:

    python -m benchmark.evaluation.anytime.run_anytime \
        benchmark/running_results/blocksworld/simulation-final-run__mask=0.1__noise=0.2

Each fold gets `anytime_scores.json` and `anytime_curve.png`. Nothing re-runs a
learner, so it is safe to point at a finished run and safe to re-run after
changing how the score is computed — which is the whole reason D5 put scoring
here instead of inside each arm.

**Four artifact shapes, one checkpoint stream.** Every arm already wrote
intermediate models; no two wrote them the same way. Plain CDPS, the anchored
variant and single-round MILP all use `conflict_free_solutions_log.json` +
`conflict_free_models/conflict_free_model_{i}/model.pddl`; the loop uses
`milp_loop_rounds.json` + `milp_loop_round_models/round_{i}/`; ROSAME uses
§7.5's snapshot index. The reader normalises all four into `Checkpoint(arm,
index, elapsed_seconds, model_path)` and nothing downstream knows the difference.

Two decisions inside it that a future reader would otherwise have to rediscover:

1. **All arms are scored against the *fold-level* `original_observations/`.**
   `cdps_anchored/` keeps its own copy, and using each arm's own would have been
   the easy default. But anchoring is part of the arm, not part of the yardstick
   — scoring an arm against inputs it improved for itself measures the wrong
   thing. One reference, recorded in the output JSON.
2. **`single_round` gets a `final_model/` fallback.** When it ends *with*
   conflicts it writes an **empty** solutions log and parks its model under
   `final_model/`. Reading an empty log as "no checkpoints" would erase the arm
   from the plot precisely in the runs where it did worst — the one direction of
   error a performance profile must not have. It contributes one point, not a
   curve, which is the honest shape for a one-shot solver.

**Scoring is ground-truth-free**, via `observations_reconstruction_score`, not
`evaluate_model`. The latter scores against the GT domain and exists for offline
reporting; using it here would make the plot a different claim than the one the
loop's own selection rule makes. The y-axis is `success_rate`, already a bounded
ratio, so §4bis.5's `v_raw` non-comparability never reaches it — `v_raw` and
`v_per_transition` are recorded alongside but not plotted.

**A model that will not parse is carried as a point with no score**, never
dropped. Dropping it would quietly flatter whichever arm emitted it; carrying it
means it appears in the scatter but can never advance the running best.

**The dedup cache keys on file text with the `:requirements` line stripped.**
This sits close to §7.2's warning that model identity must come from the log's
`model_hash`, never from file text, so the distinction matters: this digest is a
*cache key*, never an identity claim in a reported number. `to_pddl` rebuilds
`:requirements` from a set, so without the strip one model renders as several and
the cache misses — a false negative, which costs only time. A false *positive*
would need two models whose text agrees on everything except that line, i.e.
identical actions. Real effect on real data: the anchored arm's 17 checkpoints in
one fold collapsed to **3 distinct models**.

**The load-bearing test is `test_density_does_not_change_the_curve`.** It is what
licenses drawing ROSAME's hundreds of snapshots against the loop's handful of
rounds on one axis: a dense arm seeing the same models at the same times as a
sparse one, plus repeats in between, produces the identical profile and the
identical staircase. Without that property the plot would reward instrumentation
rather than learning. The tests build fold layouts in temp dirs rather than
pointing at a real run, because `benchmark/running_results/` is gitignored and a
test needing a fold on disk is a test that passes on one machine.

Verified on all 30 folds of blocksworld `sim_run__mask=0.1__noise=0.2`: every
fold read, scored and plotted; one fold hand-checked at 3 observations / 15
transitions, scored in 0.1 s. That cell predates §7.5 and the MILP arms, so only
`cdps` and `cdps_anchored` appear in it — the other two shapes are covered by the
tests until P5.7 produces runs that hold them. `simulation-final-run__*` would
have been the cell to read (§7.8), but it predates the MILP arms equally, so it
would have exercised the same two shapes; this check is about the reader parsing
artifacts, and blocksworld carries no ancestor-typed predicate, so nothing here
turns on which of the two cells it ran against.

### 7.7 D6 — config-driven ablation selection (done)

D6 verbatim: *"i dont want to run ALL configurations by default: i would want to
choose what value(s) i am going to use for those runs, just like we have in the
benchmark runner"*. The shape chosen was **per-knob value lists**, i.e. the same
cross-product `simulation.grid` already uses:

```yaml
  cdps_milp:
    eq16: off
    pool_policy: frozen
    ablations:
      eq16: [on, off]
      pool_policy: [frozen, replace]
```

**Override, not additive.** A knob listed under `ablations:` *replaces* its value
in the surrounding block; the list is the complete set of values. The example is
4 loop arms, not 5. The alternative reading — the list adds to the block's own
value — would silently run a configuration nobody named, which is D6's complaint
in miniature. `simulation.grid` has no scalar sibling next to `masking_ps: [...]`
for the same reason. The surrounding block keeps its job: supplying every knob
the ablation block does *not* mention.

**Two checks, because there are two ways to lose a result.** They look like one
question and are not:

| check | question | what it prevents |
|---|---|---|
| `CdpsMilpConfig.arm_identity()` | would these produce the same model? | running one solve twice under two names |
| label distinctness in `milp_configs_for` | would these land in the same row? | two algorithms averaged into a row naming neither |

The first is why `ablations: {pool_policy: [frozen, replace]}` can coexist with
`algorithms: [cdps_milp_single_round, cdps_milp_loop]`: `pool_policy` is inert
under single-round, so SR collapses to one arm while the loop gets two. Plain
dataclass equality is too strict for that question, so `arm_identity` drops the
loop-only fields under `single_round` and normalises `lambda_pre` to 0.0 when
`eq16` is off — the same rule `as_stats` already applies.

The second is the one that would have cost a run. `cdps_milp_algorithm_name`'s
suffix does not carry `seed`, `stop.*`, `eval.*`, `solver`, `obs_weights` or
`time_limit_seconds`; ablating `seed` would produce two genuinely different
models both reporting as `CDPS_MILP_LOOP`. That now raises before the run rather
than being discovered in a report. `stop`, `eval` and `variant` are rejected up
front as unablatable; the rest are caught by the label check, which is
**self-maintaining** — add a knob to the label function and it becomes ablatable
with no second list to keep in sync.

**Directories follow labels.** `milp_work_subdir` reuses the label's suffix, so
`cdps_milp_loop__pool=replace/` is readable from its results row and vice versa,
and two arms in one fold cannot overwrite each other's artifacts. An arm with no
suffix keeps the bare `cdps_milp_loop/` it always had, which is what lets
`backfill_cdps._WORK_SUBDIRS` and `anytime.checkpoints.ARM_SUBDIRS` stay
untouched. (Teaching those two to discover suffixed dirs is only needed once a
run actually ablates; it is not needed for P5.7's eq16 ride-along, which can be
two separate cells.)

Expansion happens in `benchmark_runner._build_main_kwargs`, early, because the
run banner and `run_params["algorithms"]` have to name every arm *before*
anything runs — a manifest that disagrees with its own data is worse than no
manifest. It is validated even when both MILP arms are off, so a typo cannot sit
unnoticed in a config that later enables one.

Verified: the shipped `run_config.yaml` still expands to exactly **one** config,
so nothing about existing runs changes. 10 new tests in
`benchmark/test_milp_ablations.py` (on the benchmark side because the feature
spans both layers and `src` must not import `benchmark`), plus the pre-existing
12 + 65 MILP tests still passing.

### 7.8 Provenance audit — which cell each P5 claim rests on, and the depot defect

Raised as: *"the relevant data files (of already-run experiments) for all domains
are in `simulation-final-run__*` data! the sim_run of the depot is indeed broken.
did those data affect any other problem along the way which made you make
assumptions/decisions because of that? if so - list all of them"*, then *"correct
the process.md and tell on what data it should have been tested against. are
those re-runs necessary?"*

Two separate questions turned out to be tangled here: **which cell a measurement
read** (provenance) and **whether depot is broken** (a defect). They are not the
same question and the answers point opposite ways.

#### What exists on disk

| cell family | domains covered | folds/cell |
|---|---|---|
| `simulation-final-run__*` | blocksworld, depot, gripper, hanoi, npuzzle — **all five**, 9 mask/noise combos | 30 |
| `sim_run__*` | blocksworld, depot, gripper only | 30 |
| `simulation-cluster-run-cpu256-10min__*` | npuzzle (among others) | 30 |

`simulation-final-run__*` is the authoritative set because it is the only one
that covers every domain at every grid point. `sim_run__*` is a partial earlier
set; npuzzle and hanoi have no `sim_run` cell at all.

#### The depot defect is live code, not stale data

The working hypothesis in the previous revision of §4ter.6 and §7.4 was that
depot's `sim_run` cell was a stale snapshot written before `96353c4ff`, and that
reading `simulation-final-run__*` instead would make depot encode. That was
wrong, and measuring it took one probe:

| depot cell | fluent in both polarities in the **file text** | ...in the **loaded observation** |
|---|---|---|
| `sim_run__mask=0.1__noise=0.2` | 0/3 traces | 3/3 traces |
| `simulation-final-run__mask=0.1__noise=0.2` | 0/3 | 3/3 |
| `simulation-cluster-run-cpu256-10min__mask=0.1__noise=0.2` | 0/3 | 3/3 |
| `simulation-after-pred-fix__mask=0.1__noise=0.2` | 0/3 | 3/3 |
| `simulation-cluster-run__mask=0.1__noise=0.2` | 0/3 | 3/3 |

Every frozen trajectory is clean. Every load of it, by today's tree, is not. The
contradiction is **manufactured in memory on each run**, so no choice of cell
avoids it.

Walking `load_masked_observation`'s three stages on depot's `s0`: 0 clashes after
`parse_trajectory`, **2 after `ground_observation_completely`**, and masking adds
none. The colliding pair:

    str='(clear p1 - object)'         pos=True   sig={'?x': object}
    str='(not (clear p1 - package))'  pos=False  sig={'?x': package descendant of object}
    a == b ? False    (ignoring polarity: True)

`GroundedPredicate.__eq__` walks the type hierarchy while `__hash__` is
`hash(str(self))`, which embeds the type tag — so the two spellings hash apart
while comparing equal, violating the `__eq__`/`__hash__` contract. The
set-membership probe in `ground_all_predicates_in_state` lands in the wrong
bucket, never consults `__eq__`, and CWA-completion appends the negative beside
the present positive.

The violation is **asymmetric**, which is why it survived this long unnoticed.
`__eq__` compares types with `is_sub_type`, which is one-directional, so with
`a` the `object`-tagged predicate and `b` the `package`-tagged one:

    a == b : False        b == a : True        hash(a) == hash(b) : False

Equality is not symmetric, so `a == b` and `b == a` disagree and the contract is
violated only in one direction. Any probe that happens to put the concrete-typed
operand on the left sees consistent behaviour; the reversed probe does not. This
is also why the "just key the membership test on `(name, args)`" fix was
rejected: it stops the contradiction but leaves both spellings circulating in one
state, and `mask_state`'s linear `==` scan (`src/utils/masking.py:120-124`) would
then silently depend on set iteration order deciding which operand lands left.

#### `96353c4ff` moved the bug, it did not fix it

There are two grounding conventions and two parser call styles, and they pair up
as a clean anti-diagonal. Measured by monkeypatching `get_all_possible_groundings`
between the pre-fix implementation (keep the lifted declared signature) and the
current one (refine to the object's concrete type), crossed with both call styles,
counting both-polarity fluents in depot's `s0`:

| parser call style | lifted (pre-`96353c4ff`) | concrete (current) |
|---|---|---|
| `TrajectoryParser(domain)` | **0** | 2 |
| `TrajectoryParser(domain, problem)` | 2 | **0** |

Without a problem file the parser cannot resolve an object's type and falls back
to the lifted signature, yielding `(clear p1 - object)`; with one it refines to
`(clear p1 - package)`. Grounding must match whichever the parser produced.
`96353c4ff` made grounding always concrete, which fixed the with-problem style and
broke the without-problem style. The two call sites split by algorithm family:

| passes a problem? | call sites | consequence for depot |
|---|---|---|
| **no** | `src/utils/masking.py:219` (`load_masked_observation`), `pddl_trajectory.py:241`, `run_fold.py:71`, `simulated_data_utils.py:45`, `run_simulated_experiment.py:68`, `post_process_gt_metrics.py:28,49` | contradictory — **CDPS and both MILP arms** |
| **yes** | `baselines/rosame_runner.py:122`, `baselines/rosame_i_runner.py:281`, `src/depot-polarity-test/repro.py:70`, `src/domains/hanoi/algorithm.py:79` | consistent — the ROSAME family |

Two consequences worth stating plainly. First, `src/depot-polarity-test/repro.py`
is **not a valid regression test**: it passes a problem, so it exercises the one
call style production code does not use, and it passes while production fails.
The comment block in `get_all_possible_groundings` that cites it as the check is
wrong on both counts and needs correcting with the fix. Second, on depot the
ROSAME baselines and the CDPS/MILP family are currently learning from
**observations that differ**, which makes any depot head-to-head unsound
independently of whether the MILP converter rejects the trace.

#### Blast radius: depot only

Only an *ancestor-typed* predicate — declared over a type that has subtypes —
can exhibit the mismatch, since otherwise both conventions name the same type.
Scanning all seven domain files:

    depot   ancestor-typed predicates: [('clear', {'?x': 'object'})]
    blocks, gripper, hanoi, hiking, maze, n_puzzle: none

So `clear(?x - object)` is the only instance in the repository. Every
blocksworld, npuzzle, gripper and hanoi number in this document is untouched by
this defect; it can only ever have been a provenance question for them.

#### Per-claim: what each P5 result read, and what it should have read

| § | claim | read | should have read | affected by the defect? |
|---|---|---|---|---|
| 7.1 | loop picks a repair by a GT-free score; CP-SAT pinned | blocksworld `sim_run` folds | either — code-path check | no |
| 7.2 | per-round `model.pddl`; identity from `model_hash` not text | blocksworld `sim_run` fold | either — code-path check | no |
| 7.3 | backfill dispatch, two repair-cost scopes, arm-specific input | throwaway copy of a blocksworld cell | either — code-path check | no |
| 7.4 | loop-vs-round-1 headroom table (16/30 blocksworld, 0/30 npuzzle) | blocksworld `sim_run`, npuzzle `simulation-cluster-run-cpu256-10min` | `simulation-final-run__mask=0.1__noise=0.2` per domain | no — **empirical, provenance-affected** |
| 7.4 | depot excluded, every trace unencodable | depot `sim_run` | any depot cell — all identical | **yes, and it is the defect itself** |
| 7.5 | ROSAME per-epoch snapshots | synthetic + unit tests | n/a | no |
| 7.6 | anytime reader/scorer over 30 folds | blocksworld `sim_run` | `simulation-final-run__*` | no |
| 7.7 | ablation expansion, arm identity, label distinctness | config expansion + unit tests | n/a | no |

#### Are the re-runs necessary?

Three groups, three answers.

**The mechanical checks (§7.1, §7.2, §7.3, §7.5, §7.7) — no.** These verify that
a code path does what it says: that a round writes its model, that an arm reads
its own input, that a config expands to the arms it names. A different cell
exercises the same branches with different bytes. Re-running them would consume
time and change no conclusion.

**The blocksworld/npuzzle empirical results (§7.4, §7.6) — no separate re-run.**
These *are* provenance-affected: the headroom table is a claim about data, and it
was taken on cells that are not the authoritative set, so the specific fractions
16/30 and 0/30 are properties of those cells and should not be quoted as
properties of the benchmark. But neither domain carries an ancestor-typed
predicate, so nothing about them is *wrong* — only narrower than it reads. P5.7
runs the whole grid over `simulation-final-run__*` regardless, which
re-establishes both on the authoritative set as a by-product. Commissioning a
separate re-run now would duplicate work P5.7 does anyway.

**Depot — no re-run helps, and one is actively misleading.** The failure is in
the loader, not in the data: re-running the P5.3 sweep against
`simulation-final-run__mask=0.1__noise=0.2` would reproduce the same rejection on
the same three traces and would look like independent confirmation of a data
problem that does not exist. Depot needs the grounding defect fixed first; only
then is a measurement of it worth taking. Until that lands, depot's absence from
every MILP result in this document is a statement about the loader and must not
be read as a statement about the domain's difficulty.

That has now landed (see Resolution below), so the ordering constraint is
discharged and depot becomes a re-run like any other — with the *scope* answer
unchanged: it falls out of P5.7 rather than needing a sweep of its own.

#### Resolution

Fixed by normalising, not by picking a convention. `get_all_possible_groundings`
keeps emitting concrete tags; `normalize_predicate_types_in_state`
(`src/utils/pddl_state.py`) re-tags the *parsed* predicates to match, and
`ground_all_states_in_observation` calls it on every state immediately before
CWA-completion. Both call styles therefore converge on one spelling before
anything hashes, which zeroes **both** rows of the anti-diagonal instead of
swapping which one is broken.

Three alternatives were rejected. Making the without-problem sites pass a problem
touches ten call sites including offline tooling, adds a `KeyError` path, and
leaves `masking._parse_masked_predicate_string` still emitting lifted tags.
Keying the membership probe on `(name, args)` stops the contradiction but leaves
two spellings circulating in one state, which makes `mask_state`'s linear scan
order-dependent given that `Predicate.__eq__` is asymmetric. Repairing
`__eq__`/`__hash__` is the only fix that ends the *class* of bug, but it lives in
`pddl_plus_parser`, outside this repository.

`src/utils/test_pddl_state.py` is the regression check the anti-diagonal asked
for: no contradictory fluent under either call style, plus the stronger claim
that both styles ground to byte-identical typed states. Confirmed to fail on
three of its cases with the normaliser stubbed out. `src/depot-polarity-test/` is
deleted — it only ever exercised the row that already passed.

Verified in four passes, each on a different kind of evidence:

| check | before | after |
|---|---|---|
| unit tests (`test_pddl_state.py` + `test_evaluator.py`) | n/a | 25 passed |
| both-polarity fluents, depot `s0`, both call styles | 2 / 2 | **0 / 0** |
| distinct fluents / literals in depot `s0` | 52 / 54 | **52 / 52** |
| `observation_to_trace` on depot's 3 traces | `REJECTED` ×3 | **`OK` steps 9, 8, 9** |

The 54→52 literal count is the fix seen directly: exactly the two spurious
negatives disappear and no real fluent goes with them.

**Depot reaches the learner for the first time.** Every measurement above stops
at the converter, which was the question that had been open — but it had never
been established that anything *downstream* would accept depot either, since no
depot trace had ever got that far. One fold end-to-end
(`simulation-final-run__mask=0.1__noise=0.2`, `fold0_numtrajs3_gtrate0`, 60 s
budget) settles it: `cdps` returns a model at precision 0.790 / recall 0.750
after exploring 6 conflict-free models, and `cdps_milp_single_round` returns one
at 0.750 / 0.660. Neither number means anything on one fold at n=3 — the claim is
only that the search runs and terminates with a model, not that the model is
good.

Re-run scope, measured rather than assumed: contradictions number 146 across
depot's three traces and **0** on blocksworld, gripper, hanoi and npuzzle, and
the ROSAME family already parsed with a problem. So only depot's `CDPS` and
`CDPS_ANCHORED` rows are affected — 270 folds each. The frozen
`original_observations/` on disk are not corrupt (the corruption was at load
time), so `backfill_cdps.py --force` regenerates exactly those rows without
touching data, other domains, or other algorithms.
