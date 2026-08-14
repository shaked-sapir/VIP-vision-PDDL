# Plan: `cdps_milp_single_round` + `cdps_milp_loop`

> Status: PLAN v2 — agreed across the 2026-08-10 brainstorm sessions, based on
> `docs/cdps-milp-denoiser-design.md` and `docs/rosame-milp-vs-cdps-milp.excalidraw`
> (Evaluate/Learn pseudo-functions + min_V graph), with Eqs. 11–16 verified
> against the ICAPS-26 paper PDF. This document is the execution plan; the
> design doc remains the authority on the MILP encoding details.
>
> **Amended 2026-08-12 after P4.** Points that the implementation settled
> differently are marked **AMENDED** / **SUPERSEDED** in place rather than
> rewritten, so the reasoning stays auditable. P1–P4 are done; §7 carries the
> per-phase status. `docs/CDPS-MILP-loop-PROCESS.md` is the running log.
>
> **Paths below predate 2026-08-14.** The `refactor-seprerate-denoising-from-pisam` branch moved
> `src/pi_sam/plan_denoising/` → `src/plan_denoising/`, lifted the MILP encoder out
> to `src/milp/`, and renamed `milp_version/` → `milp_denoiser/`.
> Left in place under the same amend-don't-rewrite rule as everything else here.
> CLAUDE.md's module map has the current layout.

---

## 0. The two algorithms (separate, no hybrid)

| key | what | assumption | role |
|---|---|---|---|
| `cdps_milp_single_round` | ONE joint MILP over ALL fold trajectories → T′ → PI-SAM. Exactly the settled design doc. | all traces fit one MILP (true at our scale) | Baseline of our method-family; global-optimality story; the `cost(MILP) ≤ cost(best CDPS CFM)` lower-bound check lives HERE and only here. |
| `cdps_milp_loop` | Homogeneous rounds: EVERY round (including the first) samples a subset, solves, learns, evaluates. Round 1 simply has no prior (M_best empty). | none — subsets keep every solve small | Anytime story; rising success-rate plot; scale path. |

**Explicitly ruled out (2026-08-10):** a hybrid where round 0 solves all
traces and later rounds solve subsets. Either everything fits → use
`single_round`; or it doesn't → all rounds subset-sample, no exceptions.
Consequence: the loop has one round function called repeatedly — simpler
code — and the lower-bound check is a `single_round`-only assertion.

## 0.5 Notation — the three objective weights

| symbol | term it weights | role | value |
|---|---|---|---|
| `w_m` | model-edit cost (CDPS's notion) | prices *changing* the model | **0 always**, everywhere (design doc §2.5); keeps the objective = pure minimal repair. |
| `w_prior` | agreement between witness `pre/add/del[α,ℓ]` and **M_best** (best PI-SAM model so far) | prices *disagreeing with the previous best model*; the only carrier of cross-subset consistency in the loop. Soft term, never hard constraints. | 0 while M_best is empty; then **tie-break normalization: `0.9/#model-bits`** (can never buy one flip; selects among cost-optimal repairs). Config-sweepable over {0, tie-break, 1/bit} — the 1/bit arm is the ROSAME-faithful comparison. |
| `λ_pre` | unconditional bonus for `pre[α,ℓ]=1` (Eq. 16 of the '26 paper: `+λ·pre`) | biases literals into preconditions to defeat the steppre=0 escape (mirrors their bias loss Eq. 5, same λ) | eq16-on variant only; **their value λ = 0.4** (Sec. 7 Training). |

The '26 paper's analog of `w_prior` is **Eqs. 13–15** (model-agreement
terms, coefficient `2·ROS_X − 1 ∈ [−1,1]` — weight-1 scaled by the SOFT
model's confidence). We do not copy that scale: our M_best is binary (every
bit would get full weight — their scheme never does that), their model term
is epistemically equal to their soft state/action predictions (ours:
observations = data with proof obligations, M_best = our own inference from
that same data → double-counting), their Fig. 6 admits the term balance is
unsolved future work, and their downstream damping (ψ=0.99 pseudo-label
aging + gradient training) has no analog in our pipeline.

## 1. Decisions settled (2026-08-10, both sessions)

1. **No hybrid rounds** — see §0.
2. **M is always PISAM(re-mask(T′)), never the MILP witness** (steppre=0
   escape). Witness discarded (except as diagnostic in eq16-on).
3. **w_prior**: soft term, tie-break default, 3-arm config sweep (§0.5).
4. **Evaluate reference**: the ORIGINAL noisy observations — the exact
   trajectories the experiment started with, frozen throughout; observed
   fluents only. Never simulator GT (V_GT is a logged diagnostic + the
   noise-floor line only), never a previous round's repairs.
   Two distinct per-round quantities: `cost_r` = T′_r vs originals
   (#observed-fluent flips — a property of the REPAIR) and `V_r` =
   rollout-under-M_r vs originals (a property of the MODEL). V is not
   cost: a model can come from a cheap repair yet roll out badly, and under
   the tie-break prior costs are near-identical across rounds — V is what
   the loop selects on.
5. **Rollout rule: apply-anyway + count.** From each trace's s₀ (GT by
   assumption) execute the observed action sequence under M. Precondition
   failure ⇒ log one inapplicability event, apply effects anyway, continue.
   `V = w₁·(#effect mismatches on observed fluents) + w₂·(#inapplicability
   events)`; w₁ = w₂ = 1. (w₁,w₂ exist ONLY inside Evaluate — they weight
   the two error types: wrong-prediction vs over-caution. Both counts are
   logged separately, so any reweighting is post-hoc from logs, no reruns.)
   One-step variant (from each state apply aₜ, compare t+1) logged as a
   secondary metric.
6. **Two encoder flavors: eq16 off / on** (λ_pre per §0.5). The flag lives
   in the shared encoder, so BOTH algorithms get it. Note: eq16 changes T′
   itself (declared preconditions constrain repaired states → non-minimal
   repairs), so it is meaningful in `single_round` too — and single_round is
   the cheapest, cleanest place to run the "does PI-SAM cover for eq. 16"
   experiment (one solve per variant, no loop dynamics). In eq16-on the
   witness's preconditions become meaningful → report the witness as a
   diagnostic model (`algorithm_specific`), never returned. Lower-bound
   check valid only for eq16-off. Hypothesis: PISAM(T′_off) ≥ PISAM(T′_on)
   ⇒ the denoiser/learner decomposition beats folding model pressure into
   the solver.
7. **Two sampler flavors** (config): `random` (baseline) and
   `hardest_first` (active-learning flavor: the m traces with worst
   per-trace V under M_best; needs a cooldown guard so a
   systematically-biased trace isn't hammered forever).
8. **Two learner-input flavors** (config): `subset_only` and `accumulated`
   — see §2.3. **Default: `subset_only`** (decided 2026-08-10).
9. **k = 1** — one solution per solve; solution-pool machinery out of scope
   for now.
10. **Stop rules pluggable** (config, any-of semantics): wall-clock budget
    (default = CDPS's search budget, for fair anytime comparison); no V
    improvement for j rounds (default j = 5); V(M_best) = 0; max_rounds.
11. **No VLM confidences for now.** The encoder API keeps an optional
    per-fluent-slot weight parameter defaulting to 1.0 — a dormant hook
    with zero behavior change today; image mode may use it later.

## 2. The loop, precisely (homogeneous rounds)

```
Learn(all_trajectories, config):
    M_best ← none; min_V ← ∞; repaired ← {}          # trace_id → latest repaired version
    for r = 1, 2, ... until stop_rule(config):
        S_r  ← sample(all_trajectories, config.sampler, M_best)
        T′_r ← MILP(S_r, prior=M_best, w_prior=config.w_prior, eq16=config.eq16)
        repaired[t.id] ← re_mask(T′_r[t])  for t in S_r
        learner_input ← T′ set per config.learner_input (§2.3)
        M_r  ← PISAM(learner_input)          # on conflict in 'accumulated': fallback §2.3
        V_r  ← Evaluate(M_r, all_trajectories)        # vs ORIGINAL noisy obs
        log round r: subset, cost_r, V_r components, model hash, wall-clock, flags
        if V_r < min_V: min_V ← V_r; M_best ← M_r
    return M_best (+ full round log)
```

Round 1 is not special: M_best is empty so the prior term is simply absent.

### 2.1 Evaluate contract

`Evaluate(M, trajectories) → {V, effect_mismatches, inapplicability_events,
per_trace breakdown, success_rate, one_step_success_rate}` with
`success_rate = 1 − mismatched_transitions/total_transitions`. Pure
function, no solver dependency, unit-testable with hand-built models.

### 2.2 Sampler (config: `sampler`)

- `random`: m traces uniformly without replacement per round.
- `hardest_first`: m traces with worst per-trace V under M_best (random
  until M_best exists); cooldown: a trace sampled in round r is ineligible
  in round r+1.
- Subset size `m`: config. **AMENDED (P4, Q9a):** the default is `half`
  = `max(2, ceil(n/2))`, not the literal 2 written here — this line and §5's
  expression string disagreed, and `half` is the one that was implemented.
- Optional heuristic (with `learner_input: accumulated`): after a mixed-set
  conflict, co-sample the conflicting traces in the next round — forces the
  MILP to resolve their disagreement jointly; turns conflicts into the
  exploration signal.

### 2.3 Learner input (config: `learner_input`)

What PI-SAM sees in round r:

- `subset_only`: exactly re_mask(T′_r) — this round's MILP output.
  **Conflict-impossible** (MILP feasibility ⇒ a witness model explains the
  subset jointly; design doc §4.1). Downside: M_r learns from only m traces
  → weaker, bouncier models.
- `accumulated`: the `repaired` table — the latest repaired version of
  every trace visited SO FAR. **Never include never-repaired (original
  noisy) traces** — noisy traces conflicting is the very problem being
  solved; including them guarantees conflicts. The set grows from m traces
  toward all of them across the first epoch. More data per round, BUT the
  combined set was never certified by a single solve: trace A repaired in
  round 3 (consistent with witness W₃) and trace B in round 5 (W₅) may
  admit no common model → PI-SAM may raise a genuine conflict
  (e.g. must-be-add from A vs cannot-be-add from B).
  **Fallback:** on conflict, learn this round's M_r from `subset_only`
  input instead, set `mixed_set_conflict=true` in the round log, and (if
  enabled) trigger the co-sampling heuristic. The conflict RATE across
  rounds is itself a reported measurement — how mutually consistent the
  loop's repairs are.

Both flavors are cheap; run both, compare. If `accumulated` conflicts
dominate (>~30% of rounds), prefer `subset_only` or add a feasibility
re-certification pass (one fast extra solve per round) — decide from data.

## 3. Plots and the 3-way comparison (vs `cdps` AND `rosame_milp`)

Common protocol so all methods land on one axis system:

- **x-axis: wall-clock**, equal budgets per cell.
- **y-axis: the SAME Evaluate metric** (success_rate) applied to each
  method's current best model against the SAME frozen original noisy
  observations.
- **Checkpoints** (when a method emits a point):
  `cdps` → each newly discovered CFM; `cdps_milp_loop` → each round;
  `cdps_milp_single_round` → its single completion point (a horizontal
  reference line after);
  `rosame_milp` → each training epoch (or each MILP invocation) — extract a
  model by thresholding ROSAME's probabilities at 0.5 (their rule), applied
  uniformly at every epoch.
- **Why different checkpoint definitions still compare (clarified
  2026-08-10):** checkpoints only decide WHEN a snapshot is taken, never
  WHAT is measured. Every method emits the same object — a stream of
  (wall-clock timestamp, candidate PDDL action model) pairs — and every
  snapshot is scored identically (same Evaluate, same frozen originals).
  The plot is each method's step function of best-so-far success rate vs
  elapsed time under equal budgets (anytime performance profile, standard
  in the planning literature). Checkpoint DENSITY doesn't bias the curve:
  a step function's value at time t is "best model available by t" — more
  points don't score higher, only a better model sooner does.
  single_round shows nothing until its solve completes, then one jump to a
  flat line — an honest picture of a one-shot solver.
- **Evaluation is offline/post-hoc:** methods log snapshots during the run
  timestamped at model-availability; a separate harness evaluates all
  snapshots afterward, so Evaluate's runtime is charged to no method.
  Fallback secondary x-axis if wall-clock is contested: fraction of budget.
- Per-method curve = running best; loop additionally shows the per-round
  scatter. Sim-mode reference lines: GT-model success rate (noise-floor
  ceiling); V_GT variants in a diagnostic panel (never used for selection).
- Final-model comparison stays the benchmark's standard metrics table
  (model precision/recall vs GT, solvability, etc.) — Evaluate/V is the
  internal selection + anytime signal, not the headline metric.
- Fairness notes: rosame_milp's checkpoints include neural training time
  (that's honest — it's their runtime); all methods get the same budget per
  fold; same folds, same noise seeds.

## 4. Integration (repo)

- Algorithm keys: **`cdps_milp_single_round`** and **`cdps_milp_loop`** in
  `benchmark/algorithms.py` (siblings of `cdps`, full CFM artifact suite,
  NOT BaselineRunner).
- All MILP-related code under **`src/pi_sam/plan_denoising/milp_version/`**:
  - `encoder.py` — variables/constraints/objective (eq16 flag, w_prior
    term, per-slot weight param defaulting to 1.0), per-trace lengths and
    groundings, warm start `hol := OBS`.
  - `solver.py` — CP-SAT/CPMpy first (no license), Gurobi alternative;
    adapt from the ROSAME clone (`~/Documents/BGU/thesis/ROSAME`, branch
    `ROSAME+MILP`, `constraint_opt/`); mind their uniform-length and shared
    single-instance assumptions.
  - `trajectory_extraction.py` — T′ from `hol`, patches on observed
    fluents only, re-masking, `milp_masked_completion.json` diagnostic.
  - `evaluator.py` — the Evaluate contract (§2.1).
  - `loop.py` — Learn driver: sampler, learner-input policy, stop rules,
    round log.
  - `single_round.py` — thin driver: encode-all → solve → extract → PISAM.
  - `config.py` — the configuration surface (§5).
- `learning_helpers.py`: `learn_cdps_milp_single_round(...)` /
  `learn_cdps_milp_loop(...)` mirroring `learn_cdps(...)`.
- Identical artifact schema (`conflict_free_models/...`,
  `conflict_free_solutions_log.json`, `all_solutions_metrics.json`) so the
  dashboard stack works unchanged; loop additionally emits
  `loop_rounds_log.json` (per round: subset ids, cost_r, V components,
  model hash, wall-clock, eq16/w_prior/sampler settings,
  mixed_set_conflict flag).

## 5. Configuration surface (everything pluggable)

The block below is a VALID example config (each key holds one value); the
allowed options live in the comments so nothing has to be remembered.
`config.py` validates every enum key and, on a bad value, raises an error
listing the allowed options.

```yaml
cdps_milp:
  # --- which algorithm ---
  variant: loop                  # options: single_round | loop

  # --- MILP encoder (shared by both variants) ---
  eq16: off                      # options: off | on.  on = add '+ lambda_pre * pre[α,ℓ]'
                                 #   (ICAPS-26 Eq. 16 precondition bias); NOTE: changes T′ too.
  lambda_pre: 0.4                # used only when eq16: on (their published value)
  w_prior: tiebreak              # options: none | tiebreak | rosame
                                 #   none     = no reference-model term (rounds independent)
                                 #   tiebreak = 0.9 / #model-bits — prior can never buy a flip  [default]
                                 #   rosame   = 1 per literal (ROSAME-faithful arm; ablation only)
  solver: cpsat                  # options: cpsat | gurobi.  DECIDED: cpsat (no license);
                                 #   gurobi path kept as a stub for later.
  obs_weights: uniform           # options: uniform (only one for now). Dormant hook:
                                 #   per-fluent-slot weight = 1.0; image-mode confidences later.

  # --- loop driver (ignored when variant: single_round) ---
  sampler: random                # options: random | hardest_first
                                 #   random        = m traces uniformly w/o replacement  [default]
                                 #   hardest_first = m traces with worst per-trace V under M_best
                                 #                   (+1-round cooldown per sampled trace)
  subset_size: half              # AMENDED (P4, Q9b): NAMED POLICIES, not an expression.
                                 #   half = max(2, ceil(n/2))  [default] | all | <int>
                                 #   The expression language written here originally
                                 #   (min/max/ceil/floor/round over n_trajectories, to be
                                 #   parsed with an ast whitelist) was dropped: three named
                                 #   values need no evaluator to be trusted.
  learner_input: subset_only     # options: subset_only | accumulated
                                 #   subset_only = PISAM on this round's T′ only
                                 #                 (conflict-impossible)  [default]
                                 #   accumulated = latest repaired version of every trace
                                 #                 visited so far (never original noisy traces);
                                 #                 falls back to subset_only on mixed-set
                                 #                 conflict, and logs it
  co_sample_conflicts: false     # options: true | false; meaningful only with accumulated
  pool_policy: frozen            # ADDED (P4, Q6a): frozen [default] | replace | frozen_with_hints
                                 #   frozen  = the pool is always the ORIGINAL noisy traces;
                                 #             repairs are only learned from
                                 #   replace = repairs go back into the pool. Auto-disables
                                 #             dedup AND the fixpoint rule (the pool is no
                                 #             longer a fixed set, so neither is well-defined)
  stop:                          # any-of semantics: first satisfied rule stops the loop
    budget_seconds: null         # AMENDED (P4, Q7a): blank/null = INHERIT the fold's CDPS
                                 #   denoiser timeout. The literal 3600 written here was stale
                                 #   (finished runs use TO=600) and would have made the
                                 #   head-to-head against CDPS unfair by 6x.
                                 #   NOTE: this caps the WHOLE loop. The cap on ONE solve is
                                 #   `time_limit_seconds` above — two independent budgets that
                                 #   were originally one parameter.
    no_improvement_rounds: 5     # stop after this many rounds without V improving (null = off)
    stop_on_perfect_fit: true    # stop if V(M_best) == 0 — the model reproduces ALL original
                                 #   observations exactly. Rare (noise puts V's floor > 0);
                                 #   a free early-exit, nothing more.  (was: v_zero)
    max_rounds: null             # hard cap on number of rounds (null = off)
    stop_on_fixpoint: true       # ADDED (P4, Q6c): with a frozen pool the admissible
                                 #   (subset, M_best) pairs for one incumbent number exactly
                                 #   C(len(pool), m); once all are solved no further round can
                                 #   change anything, so stop.

  # --- Evaluate ---
  eval:
    effect_mismatch_weight: 1    # (was w1) cost per wrongly-predicted observed fluent
                                 #   in the rollout
    inapplicability_weight: 1    # (was w2) cost per action whose preconditions under M
                                 #   failed during rollout (apply-anyway still applies
                                 #   its effects and continues)
```

## 6. Open questions — status after 2026-08-10 review

1. **w_prior — DECIDED**: keep it (only carrier of cross-subset
   consistency; w_prior=0 = random restarts, no trend). Default tie-break
   `0.9/#model-bits`; ablate {0, tiebreak, rosame} via config.
2. **Sampler/m — revisit after smoke runs** under the no-hybrid decision:
   defaults ~~m=2~~ **m=`half`** (Q9a) + random; compare against
   hardest_first on per-round V trend slope. (Coverage-epoch partitioning
   dropped from scope; random + cooldown approximates it at our n.)
3. **Learner input — DECIDED**: implement both (§2.3); **default
   `subset_only`**; `accumulated` available for the ablation (conflict-rate
   + V-trend comparison).
4. **Pool — ~~DECIDED: k=1, pool machinery out of scope~~ SUPERSEDED by Q6a
   (2026-08-11).** The real question turned out not to be k, but whether a
   round's *repairs* replace their noisy originals in the pool the next round
   samples from. Three policies implemented — `frozen` (default) | `replace` |
   `frozen_with_hints`. `frozen` wins by default because feeding repairs back
   creates a ratchet: round r's repair becomes round r+1's evidence, so an
   early wrong repair can never be argued away by the data it overwrote, and
   V is then scored against a pool that has drifted from the observations.
5. **w₁:w₂ — DECIDED**: frozen at 1:1 (live only inside Evaluate, §1.5);
   revisit later only if needed — post-hoc from logs, no reruns.
6. **Claim structure — DECIDED**: single_round claims cost-optimality
   (never loses to cdps on cost); loop claims model quality across the
   near-optimal multiplicity. Framing: MILP = proposal mechanism, V =
   black-box selector, loop = derivative-free search over the tie set.
   Always log cost_r beside V_r. P5 includes the **V ↔ GT-model-metrics
   correlation check** (validates selection-by-V; if uncorrelated, the
   selection signal is broken — know early).
7. **VLM confidences — deferred entirely**; `obs_weights` stays a dormant
   parameter (=1.0).

8. **Solver — DECIDED**: CP-SAT (CPMpy) only for now; no Gurobi license
   dealing. The encoder keeps a solver-abstraction seam so a Gurobi backend
   can be added later without touching the encoding.

Still genuinely open (park): the §4.1 proposition under repeated parameter
types. Background (clarified 2026-08-10): the issue is the grounded-fluent ↔
lifted-literal correspondence becoming AMBIGUOUS when an action schema has
two parameters of the same type — e.g. `drive(?from - place, ?to - place)`
with fluent `at(?p - place)`: under grounded `drive(l1, l1)` the ground
fluent `at(l1)` unifies with BOTH parameter-bound predicates ⟨at, ?from⟩
and ⟨at, ?to⟩, so `stepX[p,t] = X[α, ℓ]` stops being one equality and
becomes an aggregation over all matching ℓ (OR vs sum — the open encoding
choice). The PORosame code in the ROSAME clone sidesteps this by simply
SKIPPING such actions when building constraints — those transitions
contribute nothing. We cannot inherit that skip: our domains contain
repeated-type schemas, PI-SAM handles them with its own matcher semantics,
and a MILP that skipped them would emit T′ never constrained on those
transitions — PI-SAM could then raise conflicts on a "feasible" T′,
breaking §4.1. Hence the P1 unit test on a repeated-type domain, with the
encoder aggregating bindings instead of skipping.

**P4 addendum (2026-08-12) — a related guard that cannot fire.**
`model_prior._binding` carries a distinctness check meant to reject a lifted
literal that binds the same parameter twice (e.g. `(on ?x ?x)`). It is
**unreachable**: `Predicate.signature` in `pddl_plus_parser` is a **dict**, so
`(on ?x ?x)` has already collapsed to arity 1 by the time the code sees it.
Harmless, but nobody should read it as protection that exists. This is a
property of the *prior* channel only; the encoder's own binding aggregation
(above) is unaffected and remains covered by the P1 test.

## 7. Implementation phases

1. **P1 — Encoder + solver + single_round path.** Design-doc §3
   formulation; eq16 flag; w_prior term (inactive without a prior model);
   obs_weights param (dormant); per-trace lengths/groundings; warm start.
   Unit tests: tiny domain with known minimal repair; repeated-parameter-
   type domain. Exit: lower-bound check passes on npuzzle + blocksworld
   cells (eq16-off).
2. **P2 — PI-SAM handoff + artifacts.** Re-masking, patch extraction,
   artifact emission, masked-completion diagnostic. Exit:
   `cdps_milp_single_round` end-to-end, dashboard renders next to `cdps`;
   run design-doc §7 validation; run the eq16 on/off comparison here
   (incl. witness-vs-PISAM diagnostic within eq16-on).
3. **P3 — Evaluate module.** §2.1 contract + unit tests (hand models with
   known mismatch counts; conservative model → inapplicability counted,
   not cascaded). Exit: V(GT model) reproduces injected-noise count on a
   sim fold (valid because `noise_injection.py` leaves t=0 untouched, so
   under the GT model the rollout reproduces the GT state sequence and every
   injected flip on an observed fluent surfaces as exactly one effect
   mismatch, with zero inapplicability events).
   Decided 2026-08-11:
   - **Location deviates from §4**: `src/pi_sam/plan_denoising/evaluator.py`,
     NOT under `milp_version/` — P5 uses the same Evaluate to score `cdps`
     and `rosame_milp` snapshots, so it must not drag in the CP-SAT package.
   - **Execution vs grading are separate concerns.** Grading: unmasked
     fluents only (settled). Execution: a masked slot has no truth value, so
     `pre ⊆ s` and `s \ del` are undefined on it. The rollout never hits this
     (s₀ is complete, all later states are model-computed); the *one-step*
     secondary metric does, and resolves it by **skipping transitions whose
     base state has a masked slot occurring in that action's `pre`/`del`**,
     logging `skipped_transitions`. V itself is unaffected.
   - **The V↔GT-metrics correlation check moves here from P5** as a second
     exit criterion. `argmin V` is only sound if lower V ⇒ better model in
     the GT sense; if the proxy is uncorrelated the loop optimises noise and
     P5 draws a rising curve on a meaningless axis. The check is free: CDPS
     already emits many `conflict_free_model_{idx}/model.pddl` per fold from
     identical data with differing quality — exactly the population needed —
     across every finished run. Offline script, no re-runs. Pass criterion:
     within-fold Spearman ρ ≥ +0.4 between `success_rate` and
     `precision_overall`/`recall_overall`, sign-consistent across domains.
     On failure, fix V (w₁:w₂ off 1:1, select on `success_rate`, add the
     one-step term) BEFORE building P4.
   - **Open item raised here, executed before P5**: in image mode s₀ is NOT
     currently GT+unmasked — `image_trajectory_handler.create_masking_info`
     turns the VLM's frame-0 `unknown` set into masked slots in s₀. CDPS then
     treats a partially-masked state as "GT" (= unpatchable, so those slots
     are never resolved) and the MILP admits it as a **hard** state. Fix =
     take s₀ from the problem file's `(:init ...)` (real GT, invents nothing)
     rather than forcing UNCERTAIN→false (which invents values and pins them
     unrepairably). Deferred out of P3 because it changes s₀ for *all*
     algorithms and invalidates every existing image-mode result. P3 only
     asserts-and-logs.
4. **P4 — Loop driver. DONE** (2026-08-12). Sampler flavors, learner-input
   flavors + conflict fallback + co-sampling, stop rules, round log.
   `milp_version/loop.py` + `model_prior.py` + 61 unit tests (73 green in
   `milp_version/`). Decided/amended during implementation:
   - **The exit criterion above is vacuous** and was replaced. "M_best ≥
     first-round model on V" is true *by construction* — the incumbent only
     ever moves on strict `V_r < min_V`. What is actually read instead: the
     round log's `rounds_improved`, `rounds_tied`, `best_round` and
     `stop_reason`.
   - **Greedy strict `<`, no tolerance band** (Q8a). Measured over 19,852
     within-fold pairs, V's accuracy rises *smoothly* with the gap (0.898 at
     gap ≥ 1 → 0.975 at gap ≥ 50); a band would refuse the 4,734 pairs in
     `[1,10)` that are still 79.5% correct for a mean f1 gain of +0.047.
   - **Ties break to the incumbent** (Q8c) — which strict `<` already gives,
     so no extra machinery. Tie events are counted so a degenerate run shows.
   - **Round identity is `(subset, M_best)`** (Q6c), which is what lets the
     same subset be re-drawn under a *new* incumbent and gives the exact
     fixpoint rule. It needs a **structural** model hash: `to_pddl()` is not
     stable across calls (it rebuilds `:requirements` from a set), so hashing
     the text would silently disable dedup.
   - **CP-SAT `random_seed` + `num_workers` are pinned** (Q6d). Consequence
     worth stating: `rosame_milp` results produced *before* this pin were not
     bit-reproducible.
   - **The algorithm key beats the config's `variant:` field.**
     `milp_config_for(key, cfg)` pins `MilpVariant` from the selected key, so
     ONE `cdps_milp:` block drives both arms in one run — which a single
     `variant:` value cannot express.
   - **V is scored on ALL original observations**, never on the round's own
     subset: scoring a candidate on its own training sample would reward
     overfitting and make rounds incomparable.
   - **`--resume` caveat**: `run_cdps_milp_loop` is a new `run_params` key, so
     resuming a pre-P4 experiment dir reports a spurious conflict on it.
5. **P5 — Plots + benchmark.** §3 protocol; npuzzle (starving domain) +
   blocksworld (control); 3-way comparison vs `cdps` and `rosame_milp`;
   ablations: eq16 (on single_round), w_prior 3-arm, sampler, learner_input,
   m. (The V↔GT-metrics correlation check moved to P3.)
   Note: most of the anytime snapshot stream already exists on disk —
   `conflict_search.py` writes `conflict_free_model_{idx}/model.pddl`,
   `patch_details.json` (`wall_time_seconds`) and
   `conflict_free_solutions_log.json` (`wall_time_so_far`), and
   `single_round.save_artifacts` mirrors that shape. So P5's harness is
   mostly a reader. The one real gap is `rosame_milp`, which emits no
   per-epoch snapshots today: it needs a callback that thresholds its
   probabilities at 0.5 and writes `(timestamp, model.pddl)` per epoch.

6. **P6 — Structural review of `src/` (post-P5, opinion recorded
   2026-08-11; NOTHING to be changed before P5 is done).**

   Raised by the observation that `src/pi_sam/plan_denoising/` sitting
   *inside* `src/pi_sam/` is odd. I agree, for four reasons:

   - **Inverted dependency.** `plan_denoising` *consumes* the learner: CDPS
     and `milp_version` both call PI-SAM as a black-box subroutine. A
     consumer nested inside its dependency's package is backwards; it reads
     as "denoising is a feature of PI-SAM" when the real relation is
     "PI-SAM is a plugin of the denoiser".
   - **The learner is swappable, the containment says otherwise.** The
     architecture is deliberately learner-agnostic at the seams (the MILP
     produces T′; *some* learner turns T′ into a model). Nesting hard-codes
     one learner into the namespace.
   - **Foreign code is now nested two levels deep in a learner package.**
     `pi_sam/plan_denoising/milp_version/vendor/` holds ROSAME-derived
     upstream code. Vendored ROSAME under `pi_sam/` is a smell.
   - **The evaluator forces the issue.** `evaluator.py` scores models from
     CDPS, the MILP arms and ROSAME alike. It is learner-agnostic by
     construction, so `src/pi_sam/…/evaluator.py` is simply the wrong
     address — P3 already puts it one level up as a stopgap.

   Counter-argument, recorded honestly: CDPS is *not* learner-agnostic
   today. Its conflict detection is `NoisyLearnerMixin.handle_effects`, and
   the §4.1 safety proposition is a statement about PI-SAM specifically. So
   the current nesting reflects a real coupling, not only an accident.

   Proposed target (to be decided at P6, not now) — three siblings under
   `src/` with a one-way dependency rule:

   | package | contents | may import |
   |---|---|---|
   | `src/learning/` | PI-SAM, noisy variant, `masking/`, `noising/` | — |
   | `src/denoising/` | `conflict_search.py`, `frontier.py`, `milp_version/` | `learning` |
   | `src/model_evaluation/` | `evaluator.py` (+ future model-level metrics) | neither |

   Sequencing: do it as **one mechanical, import-only commit on a clean
   branch after P5**, with no behavioural change in the same commit.
   Doing it during P4 would collide with every file under active
   development and would make the diff unreviewable.
