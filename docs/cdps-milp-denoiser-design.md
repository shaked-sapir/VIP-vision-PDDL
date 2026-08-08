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
   trajectories is the point (the lifted `pre/add/del` are shared; per-trace
   `hol` blocks are independent given the model). Our instances are tiny
   (3–8 traces × ~10 steps × a few hundred grounded fluents ≈ low tens of
   thousands of binaries). The '26 paper subsets traces only because their
   MILP also frees the actions and their datasets have hundreds of traces —
   neither applies here. If scale ever bites, decompose then; not now.
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
both run; equality certifies CDPS found an optimum).

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
   assert `cost(MILP) ≤ best CDPS cost`. Any violation = encoding bug.
2. **Un-degeneration**: on cells where CDPS returned degenerate models,
   report metrics of `PISAM(T′₁)` vs CDPS's returned model.
3. **Tie behavior**: where costs are equal, compare models/metrics
   (expect ties or near-ties; differences map the solution-multiplicity).
4. **Runtime**: MILP solve time + k PI-SAM runs vs CDPS's timeout budget.
5. **Imputation metric** (bonus): masked-completion accuracy vs GT.

## 8. Open questions for iteration

- Prove/refute §4.1 under repeated parameter types; decide the binding
  treatment in the encoder (sum vs OR over bindings covering the same p).
- k (pool size) and diversity: cost-ranked pool vs forced-diverse cuts.
- Weighted OBS (VLM confidences) — future, image mode.
- Whether to also expose the MILP's own model as a diagnostic row
  (`algorithm_specific`), never as the returned model.
- Anytime reporting: solver incumbent callback → coarse "cost vs time"
  curve to keep some anytime story for the comparison plots.
