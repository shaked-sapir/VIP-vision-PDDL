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

- **Depot is broken for every algorithm**, not just the MILP arms — the
  documented, unfixed both-polarity corruption in
  `ground_all_predicates_in_state` (`src/utils/pddl_state.py:307`, write-up in
  `src/depot-polarity-test/README.md`). Root cause: `GroundedPredicate` violates
  the `__eq__`/`__hash__` contract.
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
| 3 | **D1** — verify the loop at small and large n | Confirms no special-casing is needed; produces the small-n finding of §7.1bis as data. |
| 4 | **D3** — per-epoch snapshot callback in the ROSAME adapter | Makes the plot 3-way. Changes what a run emits, so it precedes the run. |
| 5 | **D5/D7** — the anytime harness: snapshot reader + offline scorer + curves | Consumes 1 and 4. Offline, so it can be iterated on after the run. |
| 6 | **D6** — config-driven ablation selection | Determines *which* cells the run covers. |
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
