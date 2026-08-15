# Implementation Plan: ROSAME-I + MILP (imaged mode)

> **Audience**: an implementing agent with access to this repo (`VIP-vision-PDDL`)
> and the user's local ROSAME clone at `~/Documents/BGU/thesis/ROSAME`, branch
> **`ROSAME+MILP`** (commit `95c733f1fecd9ddbe9634c8c54ebc85b27ebc076`, the one
> already pinned in `src/milp/vendor/UPSTREAM.md`). You need `dl/model.py` and
> `train_common.py` from that branch as the loss/loop reference.
>
> **Goal**: add `rosame_i_milp` — the ICAPS-26 competitor in its native setting
> (CV state predictor + ROSAME action model + MILP, trained jointly from raw
> visual traces) — as a baseline that runs on our **imaged** experiment cells,
> so it can be compared head-to-head against `pisam_milp_single_round` under the
> identical fold protocol.

---

## 0. Settled design decisions (do not revisit)

1. **One arm only: `rosame_i_milp`, the iterative loop.** No one-shot
   `rosame_i_milp_base` variant. (The existing simulated sibling
   `RosameMilpBaseRunner` is slated for deletion separately; do not extend it.)
2. **The MILP's state channel comes from the CV head** — `clamp(eps, 1 - eps)`
   of the ResNet-18 logits, per proposition per frame. *Not* from our
   VLM-inferred `.trajectory` states. This is what makes the comparison "their
   vision stack + MILP" vs "our vision stack + MILP", with the MILP encoder held
   fixed.

   *(Revised from "sigmoid" after checking upstream. Evidence:
   `main:train.py`'s synth/ResNet-18 paths apply **no** activation — the head
   emits raw logits, which is exactly what our `RosameI_Runner._build_cv_model`
   ports; only `main:models/cv_gridworld.py::CVGrid` and the `ROSAME+MILP`
   AAE-based `dl/model.py:109` apply `sigmoid`, and neither is our architecture.
   Decisively, the DL→MILP boundary itself is
   `convertor/translator.py::trans_obs_tr`, which does
   `state_preds.clamp(eps, 1 - eps)` — a clamp, not an activation. Squashing
   logits through `sigmoid` would also make the channel near-inert: a confidently
   false fluent has a logit near 0, `sigmoid(0) ≈ 0.5`, and the encoder's
   `_w(prob, scale) = round((2*prob - 1) * scale)` weighs `p = 0.5` as exactly 0,
   i.e. masked.)*
3. **The loop follows the `ROSAME+MILP` branch**, not our simulation-mode port.
   Warmup, then a MILP solve every interval; pseudo-labels feed back on **two**
   channels — model (undecayed) **and state** (ψ = 0.99 decayed). The state
   channel does not exist in simulation mode (states are data there, not network
   outputs); in imaged mode it does, and it is the substance of the method.
4. **GT budget: init hard-fixed + GT final state, same as the existing
   `rosame_milp` baselines** (`goal_mode="gt"`). This is *intentionally* more
   ground truth than `pisam_milp_*` receives (init only). The asymmetry favours
   the competitor and is deliberate — do not "fix" it. It is also coherent with
   ROSAME-I's own method, which already γ-anchors the GT final state in its loss.
5. **`pisam_milp_*` is not modified.** No `gt_anchoring` change, no
   `goal_fluents` change, no `run_fold.py` anchored-dispatch work.
6. **No encoder change.** In particular, do **not** add a "soft init" mode to
   `src/milp/encoder.py`. Both arms hard-fix the initial state; that is
   symmetric and requires nothing new.
7. **Simulation-mode cells are skipped**, exactly as `rosame_i` does today:
   print a clear message, return `(None, {})`, let the harness record a null row.

### 0a. Settled in review (2026-08-15)

The plan was reviewed against the code and against the upstream clone before any
was written. What changed, and where:

| Item | Resolution | Section |
|---|---|---|
| CV activation | `clamp(eps, 1-eps)` on raw logits, **not** `sigmoid` | §0.2 |
| `obs_p` frame coverage | Interior frames only; endpoints are hard GT rows | §4 |
| `obs_p` proposition coverage | Every instance proposition; missing CV columns get neutral `0.5` | §4 |
| GT init source | The problem PDDL's `:init` (true GT) | §4 |
| GT final lookup | Resolved once, shared by the γ anchor and the MILP goal | §4 |
| `_goal_fluents_for` | Lifted to a module-level function | §4, §9 |
| ψ exponent | **Epochs** since last labelled, not rounds | §5 |
| State-CE targets | Binary, interior-only | §5 |
| `mip_traces` selection | FIFO prefix of the shuffled epoch order, configurable | §5 |
| Compute budget | `n_seeds = 1` + budget-aware per-solve limits | §5, §7.5 |
| `_trajectory_loss` | Split into forward + loss-from-predictions | §7.6, §9 |
| Parity test | Asserted on the shared proposition subset only | §10 |
| `run_config.yaml` | Deferred — a no-op until an imaged cell exists | §9 |

### 0b. Ratified by the user (2026-08-15)

| Question | Answer |
|---|---|
| CV → MILP state channel | Upstream behaviour: `clamp(eps, 1-eps)`, **not** `sigmoid`. Plan updated (§0.2). |
| Loss scaling / reduction | **"As loyal to the original algorithm as we can."** Where our port and upstream differ, follow upstream (§5a). |
| GT final state | **Fix both** defects — hyphen-normalize the GT trajectory before parsing, *and* fix `_resolve_final_state`'s `trajectories_normalized` blind spot (§7.7). |
| Heterogeneous object universes | Ground the CV head **once on the union** of the fold's object sets; drop the raise (§7.1). |

Still open, and **not** blocking the code: §8.1 (an imaged
`pisam_milp_single_round` run).

---

## 1. Repo context you need

| Path | What it gives you |
|---|---|
| `benchmark/baselines/rosame_i_runner.py` | `RosameIBaselineRunner` — image/action/GT-final input resolution, per-domain hyperparams, n-seed selection. **Copy its input plumbing wholesale.** |
| `benchmark/algorithm_adapters/rosame_i_runner.py` | `RosameI_Runner` — ResNet-18 CV head, the differentiable ROSAME action model, `_trajectory_loss`, the two training schedules, `to_pddl`. |
| `benchmark/baselines/rosame_milp_runner.py` | `RosameMilpRunner` — the MILP plumbing to mirror: `_goal_fluents_for`, `_build_milp_traces`, `_solve`, `_original_problem_paths`, the fallback-on-infeasible contract. |
| `benchmark/algorithm_adapters/rosame_milp/milp_loop.py` | `MilpPORosame` — the *single-channel* loop. Your reference for structure, but it must gain the state channel. |
| `benchmark/algorithm_adapters/rosame_milp/model_bridge.py` | `binding_table`, `rosame_to_observation_m`, `extract_model_labels`, `model_agreement`, `solution_to_pddl`. All reusable as-is. |
| `src/milp/converter.py` | `build_ps_domain`, `build_ps_instance`, `proposition_of`, `observation_to_trace`, `gt_final_state_fluents`, `find_gt_trajectory`. |
| `src/milp/encoder.py` | `CPSATObservedActions` (registered `"cp-sat-observed"`). Note `hard_states()`, `build_objectives()` (consumes `obs.obs_p`), `action_model_sol()`, **`repaired_states(i)`**. |
| `src/milp/vendor/UPSTREAM.md` | Reference hyperparameters and the paper/code discrepancy list. Read §2 before touching ψ. |

---

## 2. What already exists, and what is actually missing

**Exists and works:**

- ROSAME-I end to end (`rosame_i`), reading `state_*.png` + observed actions +
  GT final state, ignoring the degraded trajectory states entirely.
- The MILP encoder, shared by `rosame_milp*` and `pisam_milp_*`.
- The model-prior bridge: `rosame_to_observation_m()` needs no change, because
  `RosameI_Runner` subclasses `PORosame_Runner` and therefore exposes the same
  `rosame.action_schemas` / `binding_table` surface.
- `encoder.repaired_states(i)` — already returns the solved state sequence per
  trace, which is exactly the state-channel pseudo-label source.

**Missing — three pieces:**

1. **A CV → `ObservationT` converter.** `observation_to_trace()` walks a
   `pddl_plus` `Observation`; in imaged mode there is no such object. You need a
   sibling entry point that builds the same structure from a probability matrix.
2. **The two-channel loop.** `MilpPORosame._train_step` adds only `_model_ce()`.
   The imaged loop needs a state-CE term against `repaired_states`, ψ-decayed.
3. **The runner** wiring the two together and registering as a baseline.

---

## 3. Architecture

```
RosameIMilpRunner (benchmark/baselines/rosame_i_milp_runner.py)
  │
  ├─ input resolution ────────── reuse RosameIBaselineRunner._resolve_inputs
  │                              (images, action strings, GT final predicates)
  │
  ├─ MilpRosameI (algorithm_adapters/rosame_milp/milp_loop_i.py)
  │     subclasses RosameI_Runner
  │     _trajectory_loss  = base ROSAME-I loss
  │                       + model CE   (undecayed, from MILP action model)
  │                       + state CE   (psi-decayed, from MILP repaired states)
  │     learn_pooled_with_milp(...)  ← the loop
  │
  └─ MILP round callback
        obs_p ← cv_predictions_to_trace(...)   [NEW, src/milp/converter.py]
        obs_m ← rosame_to_observation_m(...)   [existing]
        solve ← resolve_encoder("cp-sat-observed")(ps_domain, traces, {"state","model"})
        out   ← action_model_sol()  +  repaired_states(i)
```

---

## 4. The CV → MILP state channel

Add to `src/milp/converter.py`:

```python
def cv_predictions_to_trace(
    instance: PSInstance,
    proposition_names: Sequence[str],   # ROSAME's proposition order
    probs,                              # (T+1, n_props) in [0,1]
    actions: Sequence[GroundedActionCall],
    init_fluents: Set[Tuple[str, Tuple[str, ...]]],
    goal_fluents: Optional[Set[Tuple[str, Tuple[str, ...]]]] = None,
) -> Optional[ObservationT]:
    ...
```

Requirements, each mirroring `observation_to_trace` so the encoder cannot tell
the two apart, and `convertor/translator.py::trans_obs_tr` so upstream cannot
either:

- **`obs_p` frame coverage — interior frames only.** Upstream writes CV
  probabilities into `obs_p[t]` for `t` in `2..T`, then **overwrites** the two
  endpoints with hard GT rows: `obs_p[1]` from the initial state and
  `obs_p[T+1]` from the goal, each row `1 - eps` for a true fluent and `eps`
  otherwise. `probs[0]` and `probs[T]` are therefore never consumed by the MILP.
  Mirror this.
- **`obs_p[t]` spans *every* proposition of the instance**, not only those the
  CV head has an opinion about — again per `trans_obs_tr`, which zips
  `self.instance.propositions` against the full prediction row.
- **`prob`** is `clamp(eps, 1 - eps)` of the CV head's raw logit (decision #2).
  Do **not** ternarize it; the encoder's `build_objectives()` already weights
  soft probabilities, and throwing away the confidence is throwing away the
  method.
- **Proposition mapping.** ROSAME's `rosame.propositions` keys are strings; map
  each to a `PSInstance` proposition via `proposition_of(instance, name, args)`.
  Parse name/args the same way `model_bridge._ps_key` does for the schema side.
  A proposition with no `PSInstance` counterpart is dropped (log a count once —
  a large drop count means a grounding mismatch, not a benign skip). Conversely,
  a `PSInstance` proposition the CV head has no column for gets the neutral
  `0.5`, which `_w` weighs as 0 (free), rather than being silently omitted.
- **`init`** — positive fluents of the GT initial state, read from the problem
  PDDL's `:init` (**not** from `probs[0]`). `hard_states()` is closed-world, so
  whatever you pass here forces everything else false at t=1.
- **`goal`** — `gt_final_state_fluents(find_gt_trajectory(problem_pddl_path))`,
  i.e. exactly what `RosameMilpBaseRunner._goal_fluents_for` already computes.
  That is a private method on a *baseline runner*, so it is not callable from
  the `RosameIBaselineRunner` hierarchy; extract it to a module-level function
  and have both runners call it (see §9). Note this is the **second** GT-final
  lookup on the same trajectory: `RosameIBaselineRunner._resolve_final_state`
  has already found it and returned positive predicate *strings* for the γ
  anchor. Two lookups with two search orders can disagree — resolve the path
  once, and derive both the γ-anchor strings and the MILP `goal_fluents` from
  that one result. The resolved file must be hyphen-normalized before either
  derivation; see §7.7 for why, and for the second defect in that same lookup.
- **`actions`** — `t -> Action`, via the existing `_action_of` path. A step whose
  grounded action has no match must raise, as it does today.
- **`include_repeated_args`** — leave **off** (upstream `permutations`
  grounding), matching the other `rosame_*` runners. See `UPSTREAM.md` §4.

Factor the shared tail of `observation_to_trace` (init/goal/actions/hard-state
assembly) into a private helper both entry points call, so the two cannot drift.

---

## 5. The iterative loop

New `benchmark/algorithm_adapters/rosame_milp/milp_loop_i.py`, class
`MilpRosameI(RosameI_Runner)`.

**Reference behaviour** (`ROSAME+MILP` branch, `dl/model.py:274–306` per
`UPSTREAM.md` §2):

- ψ = 0.99 multiplies the **per-trace state/action** pseudo-label weights.
- The **action-model** CE (`loss_pseudo_m`) is **unweighted**; its labels are
  simply overwritten at each solve.
- Actions are observed in our setting, so the action channel does not apply.
  **Two channels, not three.**

**Loss.** Extend `RosameI_Runner._trajectory_loss` — do not fork it. The base
terms (consistency, γ-anchor to GT final, applicability, λ precondition prior)
stay byte-identical; append:

```
loss += model_ce()                                  # undecayed
loss += psi**epochs_since_label * state_ce(trace)   # per-trace, decayed
```

where `state_ce(trace)` compares the CV head's per-frame predictions against the
MILP's repaired state sequence for that trace. Follow upstream's reduction and
normalization exactly; note the deviation already recorded in
`milp_loop.py::_model_ce` — AMLGym's `forward()` ends in Softmax, so apply
cross-entropy directly on probabilities (`-(target * log(p + 1e-9)).sum()`)
rather than through `F.cross_entropy`, which expects logits. The state channel
needs the same treatment for the opposite reason: our CV head emits **raw
logits**, so `F.binary_cross_entropy` (which upstream can use because its head
is `sigmoid`-terminated) would raise on out-of-range inputs. Clamp to
`[eps, 1 - eps]` first, exactly as the MILP boundary does. Per CLAUDE.md,
that reasoning lives here and in the commit message, **not** in a docstring.

**ψ exponent is measured in epochs, not rounds.** `PseudoLabels.update` resets a
relabelled trace's weight to **1**, and `dl/model.py:243` decays it once per
training step in which the trace appears — i.e. once per epoch under a pooled
schedule. Rounds and epochs coincide only at `mip_interval == 1`; do not
conflate them.

**State-CE targets are binary and interior-only.** `extract_sol_label` builds
`torch.zeros(max_t - 2, n_props)` and fills `for t in range(2, max_t)` — the
labels are 1.0/0.0 indicator rows covering the same interior frames the state
channel supplies, never the hard-fixed endpoints.

**Schedule.** Pooled only, mirroring `learn_pooled_with_milp`. Defaults from
`UPSTREAM.md`: `pre_mip_epochs=50`, `mip_interval=1`, `mip_traces=3`,
`mip_time_limit=60`, `agreement_stop=1.0`. Per-domain `epochs` / `gamma` /
`lambda_` come from `_HYPERPARAMS` in `benchmark/baselines/rosame_i_runner.py`.

**Trace↔label alignment.** `mip_traces` takes a **FIFO prefix** of each epoch's
(shuffled) trace order — upstream's `TraceSelector` is a bounded buffer cleared
at the start of every cycle and filled with "the first `capacity` traces
encountered", *not* a criterion-based sample. The state labels therefore only
cover the selected traces. Traces not in the round keep their previous labels
(and keep decaying) — they must **not** silently get zero-valued targets. Track
per-trace `epochs_since_label` for the ψ exponent.

**Budget.** `n_seeds = 1` for this arm (settled), and the loop is
**budget-aware**: derive each solve's time limit from the fold budget still
remaining rather than always spending `mip_time_limit`, and raise
`mip_interval` when the projected number of solves cannot fit. Record any
divergence from the configured defaults in `extra_info` so a shortened run is
never mistaken for a full one. `timeout_check` is consulted between epochs
only — it cannot interrupt a CP-SAT solve, so the per-solve limit is the only
real lever.

**Fallback.** No feasible solution ever → return the plain ROSAME-I model with
`milp_failed=True` in the report, matching `RosameMilpRunner`.

### 5a. The fidelity rule

Where our port and the `ROSAME+MILP` branch differ on anything the pseudo-label
channels touch, **upstream wins**. Concretely, this settles:

| Question | Upstream, therefore ours |
|---|---|
| State-channel loss fn | `F.binary_cross_entropy(pred, label) * weight`, then `/= len(trace_labels)` (`dl/model.py:274–306`) — **not** MSE. Our head is raw-logit, so clamp to `[eps, 1-eps]` before the BCE. |
| Model-channel loss fn | Unweighted CE, `/= len_model`. Already in `milp_loop.py::_model_ce`. |
| ψ bookkeeping | Decayed **per use per epoch** inside `loss()`, reset to `1` on relabel (`convertor/pseudo_label.py`). Not per round. |
| State-label frame coverage | Interior frames only, `t ∈ [2, max_t)` (`extract_sol_label`). Endpoints are hard GT. |
| Base-loss normalization | Upstream normalizes by `B·(T+1)`; our `_trajectory_loss` is sum-reduced. **Do not restate the base loss** — §7.6 requires it byte-identical so `rosame_i` does not move. Scale the *pseudo* terms instead, by the factor that preserves upstream's pseudo/base ratio — see §5b. |

The one place we knowingly diverge is that last row, and it is forced: making the
base loss upstream-scaled would silently change the existing `rosame_i` baseline.

### 5b. The state-channel scale factor (correction to §5a as first written)

§5a originally specified `1 / (n_frames * n_props)`. That factor is wrong by
`n_frames`, and this section records the arithmetic so the number is auditable
without the upstream clone to hand.

Write `u` for the typical per-element BCE magnitude. Upstream:

- base term: sum over propositions, mean over frames → `n_props · u`
- pseudo term: `F.binary_cross_entropy(pred, label)` at its default `reduction="mean"`,
  i.e. mean over frames *and* propositions → `u`
- **upstream ratio pseudo/base = `1 / n_props`**

Ours, with the base loss frozen as sum-reduced over both axes → `n_frames · n_props · u`.
Holding the ratio at `1 / n_props` therefore requires the pseudo term to be
`n_frames · u`: **sum over frames, mean over propositions**, i.e.

```python
F.binary_cross_entropy(interior.clamp(eps, 1 - eps), label, reduction="sum") / n_props
```

The originally-specified `1 / (n_frames * n_props)` applied to a sum-reduced BCE
yields `u`, a ratio of `1 / (n_frames · n_props)` — the state channel would be
`n_frames` times weaker than upstream's, and weaker still on long traces, which
is the failure mode that quietly turns ROSAME-I+MILP back into ROSAME-I.

Implemented in `milp_loop_i.py::_state_ce`. **Unverified against upstream
`dl/model.py:274–306`**: no ROSAME clone is present on the machine this was
written on. The derivation above rests on §5a's own description of upstream, so
if that description is inaccurate this factor inherits the error.

---

## 6. Reporting and registration

- `name = "ROSAME-I_MILP"`, `display_name = "ROSAME-I+MILP"`, a distinct `color`.
- Register `"rosame_i_milp": [RosameIMilpRunner]` in
  `benchmark/baselines/__init__.py::BASELINE_REGISTRY`. Nothing in
  `benchmark/algorithms.py` needs editing — anything not in `_OUR_LEARNER_KEYS`
  falls through to `resolve_baselines`.
- `benchmark/evaluation/cfm/dashboard_config.yaml` → add under `algorithms:`
  with `modes: [image]`, alongside the existing `ROSAME-I` entry.
- `extra_info` should carry at minimum: `n_traces`, `n_gt_goals`, `milp_rounds`
  (per-round `exit_status` / `solve_time_seconds` / `objective_value` /
  `agreement`), `stop_reason`, `final_agreement`, `milp_failed`, `psi`,
  `pre_mip_epochs`, `mip_interval`, `mip_traces`, `encoding_config.as_stats()`,
  and the chosen seed. `_run_baselines` already writes the model to
  `baseline_models/<algo>/model.pddl`.

---

## 7. Constraints and gotchas

1. **Object universes — union-ground the CV head.** ROSAME-I learns a *grounded*
   readout (the head's final `Linear(256, n_props)`), so unlike plain ROSAME —
   whose learned parameters are entirely lifted and which therefore re-grounds
   per problem via `ground_new_trajectory()` — it cannot be re-grounded. Today
   `_ensure_cv_model` raises when a later trajectory grounds differently.

   **Change it to ground once on the union of the fold's object sets.** This is
   upstream's own behaviour, not a departure from it: `main:train.py`'s
   `get_domain_model` grounds from a single `models/domains/<domain>/objects.json`
   per domain (`blocks` = `block1..block5`), i.e. ROSAME-I is a one-universe
   algorithm by construction. `Rosame_Runner.ground_from_dict` already takes a
   `{type_name: [object names]}` dict, so the change is a dict union plus dropping
   the raise, applied to both `rosame_i` and `rosame_i_milp`.

   No masking tensor is needed. Measured universes across the imaged corpora:

   | domain | problems | distinct universes |
   |---|---|---|
   | blocksworld (`blocks_predefined_problems1-10_final-version`) | 10 | **2** — `{a,b,c,d}` ×7, `{a,b,c,d,e}` ×3 |
   | depot | 10 | 1 |
   | gripper | 10 | 1 |
   | hanoi | 5 | 1 |
   | npuzzle | 10 | 1 |

   Only blocksworld varies, and its two universes are **nested**. Under the
   encoder's closed-world reading, a 4-block problem seen through the 5-block
   vocabulary is simply all-false on the `e` propositions, which is correct.
2. **Images must exist on disk.** `_resolve_images` looks next to the problem
   PDDL first, then falls back to
   `src/domains/<bench>/problems/<problem>/` (depot/gripper live only there).
3. **Exactly `T+1` frames for `T` actions**, or the trace is skipped.
4. **Hyphen normalization.** Problem/GT identifiers are normalized
   hyphen→underscore before parsing against the normalized domain. Reuse
   `RosameIBaselineRunner._normalized_tempfile` and friends; do not re-solve it.
5. **Cost.** `n_seeds` × epochs × a CP-SAT solve per epoch after warmup is
   *much* more expensive than `rosame_i`. Concretely, with the plan's own
   defaults on blocksworld: `epochs=100`, `pre_mip_epochs=50`,
   `mip_interval=1` → 50 solves per seed; npuzzle's `epochs=300` → 250. At
   `mip_time_limit=60` and `n_seeds=3` that is *hours*, against a fold budget of
   `learning_timeout_seconds: 300` (`run_fold.py:113` passes
   `conflict_search_timeout or 60`). Hence the settled regime in §5: `n_seeds=1`
   plus budget-aware per-solve limits. Report the divergence explicitly rather
   than silently differing from `rosame_i`.
6. **`_trajectory_loss` cannot be extended by overriding alone.** It computes
   `preds = self.cv_model(images)` internally, optionally applies a *random*
   horizontal flip first (`augment=True`, blocksworld), and returns only a
   scalar. A subclass that wants a state-CE term over the same frames would have
   to run a second forward pass — doubling ResNet cost and, under augmentation,
   scoring a *differently flipped* batch than the base loss saw. Upstream has no
   such split: `dl/network.py:266` computes one `outputs = self.net(...)` per
   batch and every loss term reads that one dict. So split
   `RosameI_Runner._trajectory_loss` into a forward step and a
   loss-from-predictions step, and have the subclass reuse the predictions. This
   is a refactor of an existing file (see §9) and it makes us *more*
   upstream-faithful, not less; the base arithmetic must stay byte-identical and
   `rosame_i`'s results must be unchanged.
7. **The GT final state is wrong today, in two independent ways.** Both are
   fixed here (ratified, §0b), because the γ anchor and the MILP `goal` both
   depend on it and the arm is not interpretable while either is broken.

   - **`gt_trajectories/` is never hyphen-normalized.** Image cells normalize
     `training/trajectories_normalized/` only. Scanning every imaged corpus:
     depot's GT trajectories contain `at-crane at-pile at-truck empty-crane
     in-truck on-pile`, gripper's contain `at-robby`, hanoi's contain
     `clear-disc clear-peg on-disc on-peg smaller-disc smaller-peg`. Parsed
     against the underscore-normalized domain, `proposition_of` finds no
     counterpart and drops them; `hard_states()` is closed-world, so the MILP
     then *asserts them false* at the goal. blocksworld and npuzzle are clean,
     which is why this has not surfaced. Normalize before parsing — the same
     `_normalized_tempfile` path the problem PDDL already goes through.
   - **`_resolve_final_state` cannot see `gt_trajectories/` in image mode.** It
     gates the lookup on `problem_dir.parent.name == "trajectories"`, but image
     runs stage under `trajectories_normalized`. The gate never fires, the
     candidate list falls through to `problem_dir/<problem>.trajectory` — the
     *degraded VLM* trajectory — and the "GT anchor" is silently not GT. Accept
     both directory names.

   Resolve the path **once** and derive both the γ-anchor strings and the MILP
   `goal_fluents` from that one result (§4).

---

## 8. Prerequisites before this can be *run*

1. **A `pisam_milp_single_round` run on imaged cells must exist.** It has never
   been run on any domain. No code is needed — `_load_masked_observations`
   already handles `pre_built_observations=None` as "load from disk (image
   pipeline)", and `run_cdps_phase` is data-source-agnostic. It is a
   `run_config.yaml` change: `source: image`, an image `data_dir`, and
   `algorithms: [pisam_milp_single_round]`. This is the comparison target and it
   is sequenced under the experiment-platform work, not here.
2. ~~Object-universe verification per image data dir.~~ **Done** — measured for
   all five imaged corpora (table in §7.1), and the one mixed case is handled by
   union grounding rather than by avoidance.
3. ~~Confirm `rosame_i` itself runs on every domain you intend to compare on.~~
   **Done, and the earlier alarm was a false one.** `rosame_i` was recorded as
   30/30 null rows on `depot/TO=600__depot_data_from_PV`; re-running the runner
   against a real fold today produces a valid model. That cell's `data_dir` no
   longer exists on disk, so `_resolve_images` found nothing and the runner took
   its "no images" skip in 1.5 ms — a stale-cell artifact, not a learner failure.
   blocksworld's 30/30 nulls *were* the grounding raise, which §7.1 removes.

---

## 9. Files touched

| File | Change |
|---|---|
| `benchmark/baselines/rosame_i_milp_runner.py` | **new** — the runner |
| `benchmark/algorithm_adapters/rosame_milp/milp_loop_i.py` | **new** — `MilpRosameI`, two-channel loop |
| `src/milp/converter.py` | **new fn** `cv_predictions_to_trace` + shared private tail |
| `benchmark/algorithm_adapters/rosame_i_runner.py` | **refactor** — split `_trajectory_loss` into forward + loss-from-predictions (§7.6). Behaviour-preserving; `rosame_i` results must not move. **Plus** union grounding of the CV head (§7.1) — this one *does* move `rosame_i` on blocksworld, from null to a model. |
| `benchmark/baselines/rosame_i_runner.py` | **fix** — both GT final-state defects (§7.7). Also moves `rosame_i` on depot/gripper/hanoi. |
| `benchmark/baselines/rosame_milp_runner.py` | **refactor** — lift `_goal_fluents_for` to a module-level function so both runner hierarchies can call it (§4). Behaviour-preserving. |
| `benchmark/baselines/__init__.py` | registry entry |
| `benchmark/run_config.yaml` | `rosame_i_milp` in `algorithms`, image cell |
| `benchmark/evaluation/cfm/dashboard_config.yaml` | `modes: [image]` entry |

**Not touched:** `run_fold.py`, `benchmark/algorithms.py`, `src/milp/encoder.py`,
anything under `src/plan_denoising/`.

Two notes on the last three rows:

- The **registry name** is `"rosame_i_milp"`; the reported `name` is
  `"ROSAME-I_MILP"` and `display_name` is `"ROSAME-I+MILP"`, matching the
  `ROSAME_MILP*` / `ROSAME-I` conventions already in `BASELINE_REGISTRY` and
  `dashboard_config.yaml`.
- Editing **`run_config.yaml` is a no-op today**: it currently declares
  `source: simulated` with only simulated domains, so adding `rosame_i_milp`
  would insert a baseline that prints its skip message and records a null row in
  every cell. Land the registry entry and leave the run config alone until the
  imaged cell of §8.1 exists; the row belongs to that work, not this one.

---

## 10. Verification

- **Unit** — `cv_predictions_to_trace` on a hand-built 2-step trace: assert
  `obs_p` probabilities survive unrounded, that `obs_p[1]` / `obs_p[T+1]` are
  the hard GT rows rather than CV output, that `init`/`goal` land in
  `hard_states`, and that every action resolves. Mirror `test_rosame_milp.py`'s
  style.
- **Parity — on the shared proposition subset only.** Feed the converter a
  *ternarized* probability matrix derived from a real degraded observation and
  assert the resulting `ObservationT` agrees with `observation_to_trace`'s
  output on the same observation. Note the assertion cannot be set equality of
  `obs_p` rows as originally written: `observation_to_trace` emits rows only for
  predicates *present in the observation's states*, while the CV path emits one
  per instance proposition (§4), and the two coincide only for CWA-completed
  observations with no repeated-argument groundings. Compare the intersection,
  and assert separately that the CV path's extra rows all carry the neutral
  `0.5`. Weakened or not, this is still the strongest guard against the two entry
  points drifting.
- **Refactor guard** — `_trajectory_loss` before and after the §7.6 split must
  return the same scalar for a fixed seed and `augment=False`. Cheap, and the
  only thing standing between the refactor and a silent `rosame_i` regression.
- **`rosame_i` regression check** — the §7.1 and §7.7 changes are *supposed* to
  move `rosame_i`. Bound the movement: on a single-universe domain with clean
  (hyphen-free) GT, union grounding must be a no-op, so re-running one npuzzle or
  blocksworld fold whose problems already share a universe must reproduce the
  previous loss to floating-point tolerance. Everything else that moves must be a
  cell that was previously null or previously anchored on a degraded trajectory.
- **Loop** — with ψ = 0 the state channel must vanish and results must match a
  model-channel-only run; with `agreement_stop=0.0` the loop must exit after one
  round. Both are cheap smoke tests.
- **End to end** — one imaged fold on the smallest domain, asserting a valid
  PDDL model, `milp_failed=False`, and a non-empty `milp_rounds`.

---

## 11. Deliberately out of scope

- The `rosame_milp`-on-imaged-cells control arm (ROSAME over *our* VLM states).
  Cheap to add later — a registry name in an image cell's `algorithms` list, no
  code — but the learner-isolating comparison it would provide already exists in
  simulation mode, where both learners consume the same degraded symbolic traces.
- Any change to the simulated `rosame_milp` GT budget.
- Pretrained-backbone and sigmoid-head ROSAME-I variants (see `future-tasks.md`).
