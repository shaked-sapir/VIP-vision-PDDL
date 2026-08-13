# Design Note: MILP as Denoiser, PI-SAM as Learner (`cdps_milp`)

> **Status: DRAFT — under iteration.** Brainstorm-level design for replacing
> CDPS's conflict search with a MILP that solves the same minimal-repair
> problem globally, while keeping PI-SAM as the model learner (and thus the
> safety story). Motivated by the observed bottleneck: each conflict-search
> node costs one full PI-SAM run (~1 s on npuzzle → ~300 nodes per budget →
> degenerate solutions), while a MILP of our problem size solves in seconds.

---

## 1. The pipeline

```
original_observations (masked + noisy states, s₀ GT, actions observed)
        │  encode (§3)
        ▼
  MILP over (hol, pre/add/del)        — ONE solve over ALL fold trajectories,
        │                               warm-started from the observed values
        │  keep hol restricted to observed fluents; re-mask the rest (§4.2)
        ▼
  T′₁ … T′ₖ   (k best repaired trajectory sets from the solution pool,
        │      ranked by true repair cost)
        ▼
  PI-SAM on each T′ᵢ  (k plain runs, ~1 s each)
        ▼
  M₁ … Mₖ  (CFM analogs; M₁ = returned model)
```

The MILP replaces the *search over repairs*; PI-SAM remains the *learner*.
The model variables inside the MILP exist only to guarantee that T′ is
explicable by some STRIPS model — the same role conflict-freeness plays in
CDPS.

## 2. Settled design answers (from the 2026-07 discussion)

1. **Yes — we take the MILP solution's states and run PI-SAM on them.**
   The MILP's `pre/add/del` assignment is discarded (diagnostics only); the
   returned model is `PISAM(T′)`, so the PI-SAM safety theorem applies to the
   returned model with respect to T′.
2. **Masked fluents are re-masked before PI-SAM — we do NOT feed the MILP's
   0/1 completion to the learner.** Details in §4.2. The MILP *does* assign
   0/1 to masked fluents internally (they are free, objective weight 0), and
   we keep that completion as a diagnostic artifact, but T′ as given to
   PI-SAM has exactly the original masking. Fluent patches := differences
   between `hol` and the observation **on observed fluents only** — the
   exact analog of CDPS fluent patches.
3. **All fold trajectories enter one MILP.** Global consistency across
   trajectories is the point: the lifted `pre/add/del` are the ONLY coupling
   between traces (per-trace `hol` blocks are independent given the model),
   and the joint solve produces one witness model explaining every repaired
   trajectory simultaneously — which is exactly what §4.1 (feasibility ⟹
   PI-SAM conflict-freeness over the WHOLE set) requires. Subset-solving
   would break that: traces repaired in different solves need not be
   mutually consistent, and PI-SAM could raise conflicts between a repaired
   trace and an untouched one.
   Note the '26 paper's subsetting is not a counterexample: their MILP is
   far larger per trace (free `act` variables, permutation symmetry) over
   hundreds of traces, and cross-subset inconsistency is absorbed *softly*
   by gradient training with ψ-aged pseudo-labels — consistency is only
   asymptotic, never a per-solve guarantee. Our per-fold instance (3–8
   traces × ~10 steps, actions fixed ≈ low tens of thousands of binaries)
   makes the joint solve trivially tractable.
   **Fallback if scale ever bites** (100+ traces; not now): (a)
   model-communication rounds — solve subset k, pass its `pre/add/del` as
   hard constraints/priors to subset k+1, finish with a feasibility pass
   over all traces under the fixed model; or (b) lazy expansion — solve a
   subset, verify the model on held-out traces, add violated traces'
   blocks and re-solve. Both restore joint consistency but forfeit global
   optimality (upper bound only, like CDPS).
4. **Single pass, no iteration.** The '26 iterate because the MILP's inputs
   are *neural predictions that improve with training* — the loop co-trains
   predictors and solver. Our inputs are fixed observations; the MILP is
   solved to global optimality once, and a second pass (e.g., feeding
   `PISAM(T′)` back as a model prior) reaches a fixed point immediately:
   `PISAM(T′)` is already consistent with T′, so the prior reinforces the
   same optimum. The only future scenario where iteration earns its cost is
   image mode with re-queried VLM confidences — out of scope.
5. **Model prior off** (their "model" objective term omitted) and **λ-prior
   off** — matching CDPS's `w_m = 0` and leaving precondition inference
   entirely to PI-SAM.

## 3. Formulation (our dialect — deviations from the ROSAME+MILP encoding)

Variables: per trace i, step t, grounded fluent p: `hol[i,t,p] ∈ {0,1}` and
step indicators `stepadd/stepdel/steppre[i,t,p]`; shared lifted
`pre/add/del[α,ℓ] ∈ {0,1}` over parameter-bound literals. **No `act`
variables** — actions are observed; every act-conditional constraint
specializes to the executed action of that step.

Constraints:

- `hol[i,1,p]` **hard-fixed** to the observed initial state (s₀-GT
  assumption). *(Their `InitT/InitF`, kept.)*
- **Final state SOFT** — participates in the objective like any other
  state. *(Deviation: their `GoalT/GoalF` hard-fix the last state; we do
  not have GT there.)*
- Effect propagation per step with executed action a = ⟨α,b⟩: for each
  fluent p touchable by a binding of a, `stepX[i,t,p] = X[α, ℓ(p,b)]` for
  X ∈ {add, del, pre} (equalities — the big-M forms collapse once `act` is
  fixed); for untouchable p: `stepX = 0`. Then
  `hol[t+1] ≥ stepadd`, `1 − hol[t+1] ≥ stepdel`, `hol[t] ≥ steppre`.
- **Frame axioms**: `stepadd ≥ hol[t+1] − hol[t]`,
  `stepdel ≥ hol[t] − hol[t+1]` — every change must be licensed by the
  executed action. Subsumes CDPS's frame-axiom conflicts *and* effect
  consistency, globally.
- Model well-formedness kept: `pre + add ≤ 1`.
  **Dropped** (their extras, not our semantics): `PreIsNotEmpty`,
  `AddIsNotEmpty`, `del ≤ pre` (delete need not be a precondition for us),
  `stepadd + hol ≤ 1` (redundant adds are legal for us).
  Guiding principle: the MILP's model class must equal our true model
  class — extra constraints force unnecessary flips; missing constraints
  risk a T′ that PI-SAM still finds conflicting (see §5).

Objective (maximize):
`Σ_{i,t,p observed} (2·OBS(i,t,p) − 1) · hol[i,t,p]`
with OBS ∈ {0,1} for observed fluents and 0.5 (weight exactly 0 — free) for
masked ones. With unit weights this is precisely **minimize the number of
fluent patches** (Occam / `fluent_patch_cost = 1`, `w_m = 0`). Future hook
(recorded, not now): non-binary OBS from VLM confidences → confidence-
weighted repair, which CDPS's unit costs cannot express.

Warm start: `hol := OBS` (their `warm_start()` pattern), model bits from a
quick PI-SAM pass optional.

Solver: prefer the CP-SAT/CPMpy path (no license) with the Gurobi encoder as
an alternative; both exist in the user's ROSAME clone
(`~/Documents/BGU/thesis/ROSAME`, branch `ROSAME+MILP`,
`constraint_opt/{cp_sat,mip_gurobi,factory}.py`) and can be vendored/adapted.
Note their per-trace uniform-length assumption (`max_t` from trace 0) and
single shared instance — we need per-trace lengths and per-trace groundings
(lifted vars shared; `hol` blocks per trace over that trace's own
proposition space).

## 4. The two subtle correctness points

### 4.1 Feasibility ⟹ PI-SAM conflict-freeness (proposition to prove)

Claim: if (T′filled, M) is MILP-feasible, then PI-SAM on T′ raises no
conflicts. Sketch, per (α, ℓ): `add[α,ℓ]=1` forces ℓ true after every
execution of α (propagation) → cannot-be-add evidence impossible;
`add[α,ℓ]=0` + frame constraint forbids ℓ flipping F→T under α →
must-be-add evidence impossible. Symmetrically for deletes. Frame conflicts:
changes without an actor are infeasible. Hence no conflicting evidence pair
can exist. **Converse**: every CDPS-reachable conflict-free (T′, Φ) is
MILP-feasible ⟹ the MILP optimum is a **lower bound on CDPS's best cost**
(built-in sanity check: `cost(MILP) ≤ cost(best CFM)` must hold wherever
both run; equality certifies CDPS found an optimum). Note what the check can
and cannot catch: an encoding that *drops* a constraint enlarges the feasible
set, makes the MILP cheaper, and passes silently; only an encoding that adds
or mis-states a constraint pushes the MILP above CDPS and fires. **The
converse is false as stated** — CDPS may also add model constraints, which the
MILP cannot, so its conflict-free models are not all MILP-feasible and the
antecedent fails. See §7.1a: restricted to the CDPS models that *are*
MILP-feasible, the check turns out to be vacuous on the whole corpus.

**Open edge case**: multiple bindings / repeated parameter types. PI-SAM's
matcher treats binding ambiguity its own way; the encoding's `unifies`
machinery is another; and `PORosame` *skips* duplicate-parameter actions
(we must NOT skip). The proposition must be checked for this case — add a
unit test on a repeated-type domain before trusting the pipeline.

### 4.2 Masked fluents: re-mask before PI-SAM

The MILP assigns 0/1 to masked fluents (free variables), i.e., it produces a
*completion*. Feeding that completion to PI-SAM would turn arbitrary
imputations into hard evidence — PI-SAM would delete preconditions and infer
cannot-be-effects from values no sensor ever reported, silently breaking the
masking semantics and the safety story. Therefore:

- T′ := observations with `hol`-vs-OBS differences applied **on observed
  fluents only**; originally-masked fluents stay masked.
- Conflict-freeness is preserved a fortiori: PI-SAM's evidence on the
  re-masked T′ is a subset of its evidence on the filled T′ (masking only
  removes evidence), and the filled T′ is conflict-free by §4.1.
- The completion itself is saved as a diagnostic
  (`milp_masked_completion.json`): comparing it to GT gives a free
  *imputation accuracy* metric — how well minimal-repair consistency
  predicts hidden values. Interesting side result; not fed to any learner.

## 5. What to expect vs `cdps` and vs `rosame_milp`

**vs CDPS (search):** same problem, same cost function, same output
semantics — different solver.
- Guaranteed global optimum vs anytime/budget-limited; on cells where the
  search starves (npuzzle: ~300 nodes at ~1 s/node), expect strictly better
  repairs and non-degenerate models. On cells where CDPS already finds
  optima, expect **equal cost** — but possibly a *different* T′ of the same
  cost (tie multiplicity), hence possibly different models with similar
  metrics. Never worse cost, by §4.1.
- What it does NOT fix: systematically biased observations (the depot
  `at`-semantics case). Global optimality finds the cheapest explanation of
  the *given* votes; if the majority votes are wrong, the optimum is the
  same wrong model the search converges to. Expect gains from the *budget*
  pathology, none from the *bias* pathology.
- Costs: lose `search_trace.json` forensics and fine-grained anytime
  behavior (solver incumbents give a coarse version); gain determinism,
  speed, and a provable optimality/lower-bound story.
- Paper framing: a second solver for the same NPO-trajectory repair problem
  (compilation-style, cf. FAMA / SAT-based learners) — run as an ablation:
  same cells, cost(MILP) vs cost(CDPS), metrics, wall-clock.

**vs ROSAME+MILP ('26 adaptation):** different creature entirely. There, the
MILP regularizes a *neural learner's* soft predictions, iteratively, and the
model is read off ROSAME/the MILP — no safety semantics, training dominates
runtime. Here, the MILP *is* the complete solver of the repair problem, no
neural component, deterministic, one shot, and the model comes from PI-SAM
with its safety lineage intact. In simulation mode ours also skips learning
what is already observed (actions, most fluents). `rosame_milp` remains the
external baseline; `cdps_milp` is a variant of *our* method.

**The CFM-set story survives** via the solution pool (Gurobi
`PoolSearchMode=2, PoolSolutions=k`; CP-SAT: iterate with a
"differ-somewhere" cut). Solutions come ranked by true cost — the anytime
plot becomes "cost-ranked solutions are better", a cleaner claim than
DFS discovery order. Distinct T′ may collapse to the same PI-SAM model —
dedup by model hash, report both counts.

## 6. Integration sketch (repo)

- New algorithm key **`cdps_milp`** in `benchmark/algorithms.py` (a sibling
  of `cdps`, NOT a `BaselineRunner` — it produces the full CFM artifact
  suite).
- New module `src/pi_sam/plan_denoising/milp_denoiser.py` (encoder + solve +
  T′ extraction) + `benchmark/experiment_running_helpers/learning_helpers.py`
  entry `learn_cdps_milp(...)` mirroring `learn_cdps(...)`'s signature and
  outputs (cleaned_model, report, patched_observations).
- **Emit the identical artifact schema**: `conflict_free_models/
  conflict_free_model_k/{model.pddl, final_observations/, patch_details.json}`,
  `conflict_free_solutions_log.json` (cost per pool solution),
  `all_solutions_metrics.json` via the existing multi-solution evaluator —
  then the entire dashboard/report stack works with zero changes and
  `cdps` vs `cdps_milp` are directly comparable everywhere.
- Encoder inputs come from the same masked observations `learn_cdps`
  receives (ternary values already available); per-trace grounding via
  existing utils (`ground_observation_completely`).
- Dependencies: `cpmpy` + `ortools` (or `gurobipy` if licensed).

## 7. Validation protocol (first experiment)

On the npuzzle sim grid (the starving domain) + one healthy domain
(blocksworld) as control:

1. **Lower-bound check**: for every cell where CDPS found ≥1 CFM,
   assert `cost(MILP) ≤ best CDPS cost`. **Compare `net_cost`, not `cost`**,
   and see §7.1a before reading any verdict into the result.
2. **Un-degeneration**: on cells where CDPS returned degenerate models,
   report metrics of `PISAM(T′₁)` vs CDPS's returned model.
3. **Tie behavior**: where costs are equal, compare models/metrics
   (expect ties or near-ties; differences map the solution-multiplicity).
4. **Runtime**: MILP solve time + k PI-SAM runs vs CDPS's timeout budget.
5. **Imputation metric** (bonus): masked-completion accuracy vs GT.

### 7.1a Check 1 is vacuous as specified (P5.7 finding)

Measured over the 1080 `simulation-final-run` folds of blocksworld, hanoi,
gripper and npuzzle. **The headline: check 1 has never once been evaluated on
a fold where it could have failed.** Its apparent 887-pass record is not
evidence that the encoding is sound; it is evidence that the check does not
test anything. Three findings, in the order they were established.

**(i) The CDPS side of the inequality was inflated.** `_compute_cost` counts
members of a `Set[FluentLevelPatch]`, and that dataclass is frozen on the
*raw* fluent string, so `(on a b)` and `(not (on a b))` at one
`(obs, comp, state_type)` are two members. They are not two edits:
`flip_fluent_in_state` looks a patch up under `{fluent, negate(fluent)}` and
toggles whatever it finds, so both records issue one identical instruction and
applying both is the identity. The realized edit count is the number of keys
touched an **odd** number of times — see
`src/pi_sam/plan_denoising/patch_accounting.py`, which is the rule
`_dedup_patches` already applies to the `next`/`prev` aliasing of one state,
extended to the case where two records share a key outright. Self-cancelling
pairs appear on 63–67% of blocksworld / hanoi / gripper folds, up to 202
patches' worth on one fold. Corrected, on the 888 folds carrying both numbers:

| CDPS cost read as | pass | violate |
|---|---|---|
| logged `cost` (raw set size) | 887 | 1 |
| realized edits (odd parity) | 629 | **259 (29.2%)** |

`cost` is left alone — it is the search's `g`, and changing it reorders the
frontier — and the corrected figure is emitted alongside as `net_cost` /
`net_fluent_patch_count` by both arms. **Read `net_cost`, never `cost`, when
comparing algorithms.**

**(ii) Those 259 violations do not convict the encoder. The two arms are not
playing the same game.** CDPS can dissolve a conflict two ways: patch the
data, or add a REQUIRE/FORBID model constraint — and at the default
`model_constraint_weight = 0.0` the second move is **free**. The MILP has no
such move at all; it fixes the model semantics and can only pay in fluent
flips. So a CDPS model carrying Φ ≠ {} is not a point the MILP could have
reached, and its cost is not a bound on the MILP's optimum. The evidence is
categorical:

| | folds | best CFM carries model constraints |
|---|---|---|
| violated | 259 | **259 (100%)**, mean 19.1 |
| passed | 629 | 269 (42.8%), mean 7.0 |

Not one violation occurs without model constraints, and violations are zero at
`noise=0.0` for every mask level (0/120 in each of the three cells) — no noise,
no conflicts, no model patches, no violation. Violation rate tracks *noise*
(34–68% at noise > 0) and is flat in *mask*, which also refutes the competing
hypothesis that the MILP was being charged for masked-fluent assignment;
independently, `repair_cost` equals the length of the MILP's own flip list on
all 1080 folds, so it charges for nothing it does not list. **Conclusion: §4.1's
converse is false as stated.** CDPS's reachable set is strictly larger than the
MILP's feasible set, so the MILP is not minimising over a superset and the
lower-bound property does not follow.

**(iii) Restricting to the comparison that *is* legitimate leaves nothing to
compare.** Keep only CDPS models with Φ = {} — the pure-data repairs, the ones
that do live in the MILP's feasible set:

```
pure-data CFM available:  360/1080 folds   (all three noise=0.0 cells, 120/120 each)
                            0/720  folds   at noise > 0, in every cell
bound holds:              360/360         of which milp < cdps:   0
                                          milp == cdps:         360
                                          both costs == 0:      360
```

Every comparable fold is `0 ≤ 0`. At noise = 0 there is nothing to repair; at
noise > 0 CDPS never once produced a Φ = {} model — unsurprisingly, since the
default `node_choosing_strategy` is `MODEL_PATCH_FIRST`, which reaches for the
zero-cost move first. The check therefore has no discriminating power on this
corpus, and never had.

**Making check 1 informative requires a code change, not a re-run.** Either
(a) give CDPS a fluent-patch-only mode so its output lands in the MILP's
feasible set — no such mode exists today; all four `NodeChoosingStrategy`
values are *orderings*, and model patches are always available — or (b) extend
the encoder with model-relaxation variables so the MILP can buy the same move,
and compare full against full. Raising `model_constraint_weight` alone is not
enough: it prices the move but does not put the two arms in the same space.

**Model quality is unaffected by any of this.** The search learns from
`apply_fluent_patches` output, so it learned from the data the cancellation
actually produced; only the price it charged itself was wrong. Precision,
recall and solving ratio for every existing CDPS row stand. What does not
stand is the claim that the MILP arm's optimality has been validated.

### 7.1b Arm comparison on the full grid (P5.7 result)

All three MILP arms backfilled onto the same 1080 `simulation-final-run` folds
(4 domains × 9 cells × 30 folds), so every comparison below is **paired** on
270 folds per domain — same data, same masking, same fold split. `SR` is
`cdps_milp_single_round`; `eq16` is the same arm with the eq-16 objective term
on at 0.4; `loop` is `cdps_milp_loop`.

Δ against `SR`, averaged over each domain's 270 folds:

| domain | arm | Δprecision | Δrecall | Δsolving | time (s) |
|---|---|---|---|---|---|
| blocksworld | eq16 | +0.0003 | +0.0043 | **+0.0204** | 0.61 → 0.62 |
| | loop | +0.0007 | +0.0017 | −0.0019 | 0.61 → **5.66** |
| hanoi | eq16 | +0.0009 | +0.0011 | **+0.0093** | 1.70 → 1.63 |
| | loop | +0.0003 | +0.0001 | +0.0037 | 1.70 → **16.75** |
| gripper | eq16 | +0.0003 | +0.0006 | +0.0037 | 0.83 → 0.83 |
| | loop | +0.0003 | +0.0006 | +0.0037 | 0.83 → **7.84** |
| npuzzle | eq16 | 0.0000 | 0.0000 | 0.0000 | 3.00 → 3.01 |
| | loop | 0.0000 | 0.0000 | 0.0000 | 3.00 → **28.09** |

**eq16 on is a small, consistent, free win.** It never hurts any metric in any
domain, helps `solving_ratio` most (+2.0pp on blocksworld), and is runtime
neutral — twice slightly *faster*, which is within noise but rules out a real
cost. It should be the default.

**The loop does not pay for itself.** It costs a uniform ~9.4× the runtime of
the single round and returns at most +0.0007 precision; on blocksworld it
*loses* 0.19pp of solving ratio. Two reasons this is a real result and not a
budget artifact:

- The loop is not being cut off. Stop reason is `fixpoint` on 720 folds and
  `perfect_fit` on 360 — **zero timeouts**. At noise > 0 it runs a mean of 25
  rounds (median 7, max 76); at noise = 0 it correctly stops at 1.1. It
  exhausts its subset space and still finds nothing better.
- Its selection rule is the GT-free `observations_reconstruction_score`, which
  is sound; the loop is choosing correctly among candidates that are simply not
  meaningfully different.

The honest reading is that **on this corpus one MILP solve over all traces
already finds what the loop's per-subset solves find.** The loop's premise —
that one unrepairable trace drags the joint solve — is not what limits accuracy
here. Precision is capped by something else (0.73 hanoi, 0.78 npuzzle, ~0.92–0.94
gripper/blocksworld, with recall ≈ 1.0 everywhere), and that ceiling is
identical across all three arms. Whatever it is, more MILP is not the lever.

## 8. Open questions for iteration

- Prove/refute §4.1 under repeated parameter types; decide the binding
  treatment in the encoder (sum vs OR over bindings covering the same p).
- **Restore some validation for the encoder.** §7.1a shows check 1 currently
  proves nothing, so the MILP arm has *no* soundness test at all. Cheapest
  route: a fluent-patch-only CDPS mode (suppress model-patch children rather
  than merely deordering them), which puts both arms in one feasible set and
  makes check 1 bite at noise > 0. Failing that, feed a violating fold's CDPS
  `T′` to the encoder as a *fixed* assignment and ask whether it is feasible —
  infeasibility names the over-tight constraint directly.
- Decide whether `_compute_cost` should net out self-cancelling patch pairs.
  Doing so is correct but reorders the frontier, so it changes which models
  CDPS finds and invalidates every result on disk; the reporting-only
  `net_cost` in `patch_accounting.py` is the interim answer.
- **Find what actually caps precision, since it is not the denoiser.** §7.1b
  shows all three MILP arms sitting on an identical per-domain precision ceiling
  (0.73 hanoi, 0.78 npuzzle) with recall ≈ 1.0, and the loop reaching that
  ceiling by fixpoint rather than by timeout. Recall ≈ 1 with precision well
  below it is the signature of *over-general* preconditions — PI-SAM keeping
  preconditions it has not seen refuted — which is a learner/observability
  limit, not a repair-quality one. Test by measuring precision against
  transition count per action: if it rises with coverage, the denoiser was never
  the bottleneck and further MILP variants are wasted effort.
- Whether to keep `cdps_milp_loop` at all. It costs ~9.4× the single round for
  ≤ +0.0007 precision (§7.1b). Retire it, or find the regime its premise
  describes — the obvious candidate is a domain where one trace is genuinely
  unrepairable, which this grid may simply not contain.
- k (pool size) and diversity: cost-ranked pool vs forced-diverse cuts.
- Weighted OBS (VLM confidences) — future, image mode.
- Whether to also expose the MILP's own model as a diagnostic row
  (`algorithm_specific`), never as the returned model.
- Anytime reporting: solver incumbent callback → coarse "cost vs time"
  curve to keep some anytime story for the comparison plots.
