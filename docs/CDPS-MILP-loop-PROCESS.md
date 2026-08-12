# CDPS-MILP loop — mid-process log

> Written 2026-08-11, mid-session, as a resume point.
> **Authority:** `docs/cdps-milp-loop-plan.md` (execution plan) and
> `docs/cdps-milp-denoiser-design.md` (encoding details). This file is a
> *status snapshot*, not a spec — if it disagrees with those, they win.

**Branch:** `cdps-with-milp-implmenetation` (uncommitted; see §5)
**Status:** P1 + P2 **DONE and validated**. P3 designed and agreed, not yet
written. Four P4/P5 questions still awaiting answers (§4).

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
| `cdps_milp_single_round` | ONE joint MILP over ALL fold trajectories | **implemented** |
| `cdps_milp_loop` | homogeneous rounds, each samples a subset | P4, not started |

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

## 4. OPEN — awaiting answers before P4

1. **Q6 — the loop has nothing to show at `num_trajectories: 3`.**
   `subset_size: max(2, ceil(n/2))` → m=2, so every subset is 2/3 of the data,
   rounds are near-duplicates and the loop degenerates into a jittery
   single_round. Bump npuzzle folds to ~10–20 trajectories for the loop
   experiments, or is there a reason to hold at 3?
2. **Q7 — budget accounting.** §3 says Evaluate is charged to no method, but
   the loop needs V *online* to select, so it pays and CDPS doesn't.
   Proposal: `budget_seconds` = the loop's **total** wall-clock including
   online Evaluate (honest — it's part of the algorithm), while the *plot*
   re-scores all snapshots offline for every method.
3. **Q8 — P4's exit criterion is vacuous.** "M_best ≥ round-1 model on V" is
   true by construction (`if V_r < min_V`). Proposal: require a **strict**
   improvement over round-1 V on ≥1 npuzzle fold, else the prior term and
   sampler contribute nothing and we want to know immediately.
4. **Q9 — `subset_size` is an expression language** (`min/max/ceil/floor/round`,
   arithmetic, variable `n_trajectories`). Proposal: `ast.parse(mode="eval")`
   + node/name whitelist, ~25 lines, no raw `eval`, unit-tested, no new
   dependency.

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

1. **V has a resolution floor of a few points.** All 11 argmin-V misses are
   near-ties — the V-winner leads the GT-best by 1–7 points while losing
   0.009–0.108 f1. Worst case: `mask=0.0 noise=0.2 fold2_numtrajs5`,
   V=209 (f1 0.892) beat V=210 (f1 1.000). → the loop must **not** treat a
   1-point V gain as progress. It needs a tolerance band plus a deterministic
   tie-break, and Q8's exit criterion should demand an improvement larger
   than the band.
2. **V cannot rank at all when the data is clean.** In the
   `noise=0.0` cell *every* candidate scores V=0 while f1 spreads 0.969–1.000.
   That residual is generalisation beyond the observed data, which nothing
   computed *on* that data can see. The loop degenerates to its tie-break
   under zero noise — acceptable, but it must be stated, not discovered later.
3. **V is not comparable across folds.** Within-fold rho −0.862 vs pooled
   rho −0.454; normalising recovers much of it
   (`rho(v_per_transition, f1) = −0.660`). D2 was the right call, and cross-fold
   aggregation in P5 must use `v_per_transition`, never `v_raw`.

---

## 5. Repo state (branch `cdps-with-milp-implmenetation`)

- **Committed — `4600b5b76`** (P1 + P2, 32 files, +2441/−252): the
  `milp_version/` move, the CDPS dialect, `cdps_milp_single_round`,
  and the `docs/` + `CLAUDE.md` updates.
- **New, uncommitted (P3):** `src/pi_sam/plan_denoising/evaluator.py`,
  `src/pi_sam/plan_denoising/test_evaluator.py`.
- **Untracked throwaway:** `benchmark/finished_run_configs/milp-dispatch-smoke/`
  and `benchmark/running_results/blocksworld/milp-dispatch-smoke__mask=0.01__noise=0.2/`
  — smoke-test output, **pending a decision: delete or keep as reference.**
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

Remaining, in order:

1. Answer **Q6–Q9** (§4), with Q8 revised in light of §4bis.5(1): the exit
   criterion should demand a V improvement **larger than the resolution
   band**, not merely a strict one.
2. **P4** — loop driver. Must carry a V tolerance band + deterministic
   tie-break (§4bis.5).
3. In parallel/background — the **eq16 on/off comparison** on
   `single_round` (already authorized under P2; cheap; it is the
   "does PI-SAM cover for Eq. 16" experiment, a claim in its own right).
4. **P5** (aggregating on `v_per_transition`, never `v_raw`), then **P6**.
