# ROSAME+MILP adapter — implementation walkthrough

A part-by-part map of the implementation: what each file/line does, and which
part of the ICAPS-26 paper (Sec. 6 equations 11–37, Sec. 7 "Training") or of
the upstream repo (github.com/xikaioliver/ROSAME, branch `ROSAME+MILP`,
commit `95c733f`) it corresponds to. Companion docs: `vendor/UPSTREAM.md`
(provenance, hyperparameters, paper/code discrepancies).

## The two registered variants

| Registry key | Class | What it is |
|---|---|---|
| `rosame_milp_24` | `RosameMilpRunner` | **The loop** (paper Sec. 6 "Integrating MILP", Sec. 7): warmup training, then a MILP solve every `mip_interval` epochs whose solution supervises further training as pseudo-labels; output = decode of the last successful solve. |
| `rosame_milp_24_tag` | `RosameMilpTagRunner` | **The loop under the `tag` rule-set** (`MilpEncodingConfig.tag`): identical to `rosame_milp_24`, but the MILP requires only ≥1 add effect per schema (no precondition requirement) and drops the redundant-add ban. Scoped to this arm alone; no other arm has a `tag` variant. |

`RosameMilpBaseRunner` holds the plumbing both share plus the one-shot `learn`
they inherit. It re-declares `name` abstract, so it cannot be instantiated; it is
not an arm. Merely omitting the override would not do — it would inherit
`RosameBaselineRunner`'s and file rows under that arm's label.

Registered in `benchmark/baselines/__init__.py`; all runners live in
`benchmark/baselines/rosame_milp_runner.py`.

### Encoding rule-sets — `MilpEncodingConfig`

The two "undocumented" constraint families (per-schema non-empty rule and the
redundant-add ban) are toggled via an immutable
`MilpEncodingConfig` (`encoding_config.py`), mirroring `CDPSConfig`. Runners
pass it through `encoding_config` (default `upstream()`):

| Preset | `schema_nonempty` | `forbid_redundant_adds` | Used by |
|---|---|---|---|
| `upstream()` | `PRE_AND_ADD` (≥1 pre AND ≥1 add) | `True` | `rosame_milp_24` |
| `tag()` | `ADD` (≥1 add only) | `False` | `rosame_milp_24_tag` |

A future variant = one new preset (+ optional `SchemaNonemptyRule` member) and a
thin runner subclass. The config's fields land in `solve_stats` /
`fold_result.json` via `as_stats()` (plus a derived `enforce_nonempty_schemas`
bool for report continuity).

## File map & data flow

```
src/milp/                                   [SHARED with the pisam_milp_* learners]
├── __init__.py        sys.path bootstrap for vendor/ (upstream absolute imports)
├── vendor/            upstream code, verbatim (planning_structs/, constraint_opt/)
├── converter.py       our pddl_plus world  ->  vendored planning_structs inputs
├── encoding_config.py MilpEncodingConfig — toggle-able MILP rule-set (presets)
└── encoder.py         CPSATObservedActions — the MILP itself ("cp-sat-observed")

benchmark/algorithm_adapters/rosame_milp/    [ROSAME-specific glue only]
├── __init__.py        triggers the vendor bootstrap by importing src.milp
├── model_bridge.py    trained ROSAME <-> MILP (obs_m, pseudo-labels, PDDL decode)
├── milp_loop.py       MilpPORosame — V2's training loop with model-CE channel
└── test_rosame_milp.py  torch-free unit tests

flow (V2): backfill/experiment_runner
  -> RosameMilpRunner.learn (rosame_milp_runner.py:243)
     -> converter: build_ps_domain/instance, observation_to_trace  [inputs]
     -> MilpPORosame.learn_pooled_with_milp (milp_loop.py:110)     [loop]
          every round: milp_round() closure (rosame_milp_runner.py:275)
            -> model_bridge.rosame_to_observation_m               [prior]
            -> encoder CPSATObservedActions.solve                 [MILP]
            -> model_bridge.extract_model_labels / model_agreement [feedback]
     -> model_bridge.solution_to_pddl                             [output]
```

---

## 1. `vendor/` — upstream code, untouched

`planning_structs/{domain,instance,traces}.py` and
`constraint_opt/{factory,util,cp_sat,mip_gurobi}.py` are verbatim from the
upstream commit. Single modification (documented in UPSTREAM.md):
`constraint_opt/__init__.py` guards the `gurobipy` import so the CP-SAT path
runs without a Gurobi license; `"mip-gurobi"` self-registers only when
gurobipy is installed. `__init__.py` of the adapter package inserts
`vendor/` into `sys.path` so upstream's absolute imports
(`from planning_structs...`) work unmodified.

Upstream's own encoders (`cp_sat.py`, `mip_gurobi.py`) are kept for reference
and parity-checking but are **not** used at runtime — they assume one shared
grounding and free `act` variables (see §3).

## 2. `converter.py` — our world → vendored structures

| Lines | What | Notes / upstream correspondence |
|---|---|---|
| 27–42 `build_ps_domain` | pddl_plus Domain → vendored `PSDomain` | Types with parents, predicates and schemas with parameter types **in PDDL signature order**. This ordering is the load-bearing convention: binding tuples `x` everywhere mean *1-based PDDL parameter positions*, so no ROSAME-style type-grouped canonicalization is needed (module docstring, lines 3–8). Upstream builds Domain by hand per experiment script. |
| 45–53 `build_ps_instance` | problem objects (+ domain constants) → `PSInstance` | One instance **per problem** — the basis of our mixed-grounding support (upstream: one global instance). |
| 76–80 `_state_prob` | ternary encoding | masked → 0.5, observed → 1−ε / ε (ε=1e-5). Matches the PO-ROSAME ternary convention; 0.5 makes the fluent objective-neutral in the MILP (see encoder `w()`). |
| 83–144 `observation_to_trace` | grounded+masked Observation → `ObservationT` | `obs_p[t]` = one `ObservationP` per grounded predicate per state (paper's p_{i,t} input to eq. 11); `init` = positive fluents of s0 (hard, GT by assumption); `goal` = GT-final fluents or None (soft). Attaches `.instance` and `.actions` (t → observed grounded `Action`) — the two extensions our encoder consumes. Unmatched action → trace skipped with a warning (line 136). |
| 147–159 `gt_final_state_fluents` | last state of a GT `.trajectory` | Regex-parses the final `(:state ...)` block. Feeds the paper's "final state is ground truth" anchoring (eqs. 21–22). |
| 162–177 `find_gt_trajectory` | problem path → GT trajectory path | Walks up to 4 parents looking for `gt_trajectories/<prob>/<prob>.trajectory` (standard data_dir layout). |

## 3. `encoder.py` — `CPSATObservedActions` (the MILP)

Adapted from vendored `cp_sat.py`; registered as `"cp-sat-observed"`
(line 358) in upstream's own factory. The docstring (lines 1–31) lists the
five deviations; the equation-by-equation map:

**Variables**

| Lines | Variables | Paper |
|---|---|---|
| 115–123 `build_domain_variables` | lifted `pre/add/dele[(schema, predicate, x)]` | the model variables of Sec. 6; shared across ALL traces (the joint-solve point) |
| 125–137 `build_trace_variables` | per-trace `hol[(i,t,p)]` (t=1..steps+1) and `stepadd/stepdel/steppre[(i,t,p)]` | hol = the paper's state variables; step\* = grounded step indicators. Built from `self._instance(i)` — **each trace's own grounding** (deviation #2; upstream uses one `traces.instance` for everything). Per-trace lengths (deviation #3; upstream applies trace 0's length to all). |
| — | **no `act` variables** | deviation #1: actions are observed data, eqs. 12 & 23 and the act side of 24–32 vanish. Upstream `cp_sat.py:147` has exactly-one-action-per-step over the shared instance — removed entirely. |

**Constraints**

| Lines | Constraint | Paper eq. |
|---|---|---|
| 150 | `del ⇒ pre` (only preconditions may be deleted) | eq. 18 |
| 152 | `¬(pre ∧ add)` (preconditions and adds disjoint) | eq. 17 |
| schema non-empty | ≥1 pre and ≥1 add per schema (`upstream`), or ≥1 add only (`tag`) | **not in paper** — upstream `PreIsNotEmpty`/`AddIsNotEmpty` (`mip_gurobi.py:102–103`); selected via `MilpEncodingConfig.schema_nonempty` (`PRE_AND_ADD` default = upstream, `ADD` = tag, `NONE` = off) |
| 179–185 | s0 hard-fixed to observed init | eqs. 19–20 |
| 186–194 | final state hard-fixed **iff** `goal is not None` | eqs. 21–22; deviation #4 — a trace without GT final state stays soft (upstream always hard-fixes) |
| 199–217 | `step* [i,t,p] == OR(model var of each binding of the observed action)` via `_bindings` (162–169: `unifies(x, p.args, action.args)`) | replaces the act-coupled eqs. 24–32 with the observed-action specialization; `unifies` is upstream's own (`constraint_opt/util.py`) |
| redundant-add | `¬(stepadd ∧ hol)` (no redundant adds) | **not in paper** — upstream `StepAddPre` (`mip_gurobi.py:177`); `MilpEncodingConfig.forbid_redundant_adds`, default True (`tag` = False). Can make the GT model infeasible under legal redundant adds — proven by unit test 3, lifted by the tag config in unit test 5. |
| 233–235 | `stepadd ⇒ hol(t+1)`, `stepdel ⇒ ¬hol(t+1)`, `steppre ⇒ hol(t)` | eqs. 33–35 |
| 236–237 | value change ⇒ explaining effect | frame axioms, eqs. 36–37 |

**Objective** (244–278) — paper eqs. 11 + 13–16, both channels selectable via
the `objectives` set (`{"state","model"}`, mirroring upstream):
`w(prob) = round((2·prob − 1) · obj_scale)` (247–249) is the integer-scaled
log-odds surrogate upstream uses; a masked fluent (prob 0.5) gets weight
**exactly 0** — it is free. "state" terms weight `hol` by observation
probability; "model" terms weight the lifted variables by the ROSAME prior
(`obs_m` from `model_bridge.rosame_to_observation_m`). Maximization, like
upstream.

**Solving & extraction**

| Lines | What |
|---|---|
| 283–304 `make_solution_hints` | warm-start hints from observation probs + ROSAME prior (upstream does the same); threshold 0.5 |
| 306–331 `solve` | OR-Tools via cpmpy, `time_limit`, populates `solve_stats` (exit status, time, objective, var counts, flags) — this dict lands verbatim in `fold_result.json` under `algorithm_specific.milp` / per-round entries |
| 335–339 `action_model_sol` | binary `ObservationM` (the learned model) |
| 341–347 `repaired_states` | solved hol sequence per trace (the "repaired trajectory" — available, currently not persisted) |

## 4. `model_bridge.py` — ROSAME ↔ MILP

The subtle part is the **row ↔ binding correspondence** (docstring 13–26):
each ROSAME schema's `forward()` returns one 4-way softmax row per (relevant
predicate, variable grounding), enumerated exactly as
`Action_Schema.pretty_print` does. We rebuild that enumeration:

| Lines | What | Correctness anchor |
|---|---|---|
| 41–44 `_signature_positions` | PDDL var name → 1-based signature position | pairs with the converter's signature-order convention |
| 46–63 `binding_table` | ordered `(pred_name, x)` per forward() row | replicates `pretty_print`'s loop (`schema.predicates` in order × `predicate.ground(var)` with `runner.get_params_names` variable names); verified by unit test 4 and by the runtime row-count guard (94–98: hard error on mismatch) |
| 74–110 `rosame_to_observation_m` | forward() probs → `obs_m` prior | **4-way semantics** from upstream's translator: index 0=irrelevant, 1=add, 2=pre, 3=pre+del ⇒ `pre = P[2]+P[3]`, `add = P[1]`, `del = P[3]` (103–105). Structurally-irrelevant triples get 0.0 (87–89), matching ROSAME's own exclusion semantics. |
| 115–142 `extract_model_labels` | binary solution → 4-way one-hot pseudo-labels per row | precedence add > del > pre > irrelevant mirrors upstream `translator.extract_sol_model` |
| 145–157 `model_agreement` | argmax-match fraction ROSAME vs labels | the loop's stop metric (paper: train until agreement) |
| 162–213 `solution_to_pddl` | binary model → PDDL string | uses the runner's own `_signature_to_pddl`/`precondition_pddl`/`effects_pddl` so ROSAME and ROSAME+MILP outputs are byte-comparable |

## 5. `milp_loop.py` — `MilpPORosame` (V2's trainer)

| Lines | What | Paper / upstream |
|---|---|---|
| 43–67 `_model_ce` | CE of each schema's 4-way distribution vs pseudo-labels, summed over rows, ÷ total rows, **undecayed** | upstream `loss_pseudo_m` (`dl/model.py`). ψ=0.99 decay applies only to state/action channels, which don't exist in our simulation setting (states/actions are data) — see UPSTREAM.md note 2. Deviation note (docstring 48–52): upstream feeds softmax outputs into `F.cross_entropy` (which expects logits); we apply the mathematically intended `−(target·log p)` directly. |
| 69–106 `_train_step` | base ROSAME loss (byte-identical re-implementation: MSE effects + validity + 0.2·precondition prior) **+ CE if labels set** | reimplemented because backward/step live inside the base `_train_step`; λ=0.2 matches AMLGym's vendored ROSAME (paper says 0.4 — UPSTREAM.md note 3). Grounding-mismatch assert at 87–89. |
| 110–178 `learn_pooled_with_milp` | the loop: pooled epochs; after `pre_mip_epochs` warmup, call `milp_round()` every `mip_interval` epochs (gate at 161–162, call at 163); install labels (173), early-stop on `agreement ≥ agreement_stop` (174–176); returns report dict (rounds, final_solution, final_agreement, stop_reason) | paper Sec. 7: warmup 50, interval 1 — our defaults match (see UPSTREAM.md table). Pooled schedule only (paper trains pooled batches). Per-problem re-grounding at 154–156. |

## 6. `rosame_milp_runner.py` — the baseline runners

**Shared plumbing** (on `RosameMilpBaseRunner`):

| Lines | What |
|---|---|
| 87–96 `_goal_fluents_for` | GT final state per problem via `find_gt_trajectory`; missing GT → warning + soft final state (`goal_mode="gt"` default) |
| 98–117 `_build_milp_traces` | prepared (problem, observation) pairs → per-problem `PSInstance` + `ObservationT`; counts `n_gt_goals` (reported — should equal `n_traces`) |
| 119–130 `_solve` | one encoder construction + solve with the configured flags (both undocumented constraints default True) and `mip_time_limit` (default 60 s, upstream value) |

**V1** `learn` (142–194): ROSAME workspace → `PORosame_Runner.learn_full`
(same training as the `rosame` baseline, `epochs=100`) → build traces →
`rosame_to_observation_m` → single `_solve` → `solution_to_pddl` on success /
plain `rosame_to_pddl` fallback with `milp_failed=True`.

**V2** `learn` (243–325): builds the `milp_round` closure (275–286: optional
`mip_traces` subsampling — default None = whole fold, paper's 3-trace
subsampling available; fresh `obs_m` each round; solve; labels + agreement) →
`learn_pooled_with_milp` (289) → if the loop never got a solution, one
last-chance whole-fold solve (304–309) → decode or fallback. Everything
observable is written into `algorithm_specific`: `milp_rounds` (per-round
epoch/agreement/status/time/objective), `stop_reason`, `final_agreement`,
`loop_seconds`, `n_traces`, `n_gt_goals`, flags.

Defaults vs upstream (UPSTREAM.md table): `pre_mip_epochs=50`,
`mip_interval=1`, `mip_time_limit=60` — upstream values; `epochs=100`
(upstream 5000 — VIP regime, matching our `rosame` baseline);
`mip_traces=None` (upstream 3 — our folds are 3–8 traces ≈ their subset).

## 7. `test_rosame_milp.py` — what is verified

1. **GT recovery** (94–110): micro move/at domain, one clean trace → encoder
   returns exactly add at(?to), del at(?from), pre at(?from).
2. **Masked freedom** (113–132): masked middle fluent (0.5) doesn't block the
   consistent solution and is repaired to the value the frame axioms require.
3. **Redundant-add infeasibility** (135–160): flag ON → GT infeasible under a
   legal redundant add (our depot argument, now proven); flag OFF → feasible.
4. **Binding table** (221–228): forward()-row ↔ PDDL-position mapping on a
   two-parameter schema (the (1,) vs (2,) distinction that silently breaks
   everything if wrong).
5. **Tag config** (`test_tag_config_allows_redundant_add`): `MilpEncodingConfig.tag`
   (add-only schema rule + no redundant-add ban) makes the same legal-redundant-add
   scenario of test 3 feasible.

Run: `python -m benchmark.algorithm_adapters.rosame_milp.test_rosame_milp`
(torch-free; needs cpmpy + ortools).

## 8. Known gaps / open items

- **Mixed-grounding unit test** — the encoder supports per-trace instances by
  construction and the blocksworld 4+5-block grid ran clean, but a dedicated
  micro-test (two traces, different object counts, shared lifted recovery) is
  still to be added.
- ~~**Depot polarity corruption**~~ — fixed upstream of this adapter by
  `normalize_predicate_types_in_state` (`src/utils/pddl_state.py`), which puts
  parsed predicates and CWA-completion groundings on one type-tag spelling.
  Regression test: `src/utils/test_pddl_state.py`. The defensive
  duplicate-proposition check in `observation_to_trace` is in place and stays.
- **Repaired traces** — `encoder.repaired_states()` exists but repaired
  trajectories are not persisted anywhere.
- **V2 `retrain_on_repaired`** — deferred design option (retrain ROSAME on the
  MILP-repaired states), not implemented.
