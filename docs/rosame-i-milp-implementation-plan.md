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
2. **The MILP's state channel comes from the CV head** — `sigmoid` of the
   ResNet-18 logits, per proposition per frame. *Not* from our VLM-inferred
   `.trajectory` states. This is what makes the comparison "their vision stack +
   MILP" vs "our vision stack + MILP", with the MILP encoder held fixed.
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
the two apart:

- **`obs_p[t]`** — a list of `ObservationP(proposition, prob)` for `t` in
  `1..T+1`. `prob` is the CV head's sigmoid output. Do **not** ternarize it;
  the encoder's `build_objectives()` already weights soft probabilities, and
  throwing away the confidence is throwing away the method.
- **Proposition mapping.** ROSAME's `rosame.propositions` keys are strings; map
  each to a `PSInstance` proposition via `proposition_of(instance, name, args)`.
  Parse name/args the same way `model_bridge._ps_key` does for the schema side.
  A proposition with no `PSInstance` counterpart is dropped (log a count once —
  a large drop count means a grounding mismatch, not a benign skip).
- **`init`** — positive fluents of the GT initial state, read from the problem
  PDDL's `:init` (**not** from `probs[0]`). `hard_states()` is closed-world, so
  whatever you pass here forces everything else false at t=1.
- **`goal`** — `gt_final_state_fluents(find_gt_trajectory(problem_pddl_path))`,
  i.e. exactly what `RosameMilpBaseRunner._goal_fluents_for` already computes.
  Reuse that method rather than reimplementing the lookup.
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
loss += psi**rounds_since_label * state_ce(trace)   # per-trace, decayed
```

where `state_ce(trace)` compares the CV head's per-frame predictions against the
MILP's repaired state sequence for that trace. Follow upstream's reduction and
normalization exactly; note the deviation already recorded in
`milp_loop.py::_model_ce` — AMLGym's `forward()` ends in Softmax, so apply
cross-entropy directly on probabilities (`-(target * log(p + 1e-9)).sum()`)
rather than through `F.cross_entropy`, which expects logits. Apply the same
reasoning to the state channel and **document whichever choice you make in a
docstring**.

**Schedule.** Pooled only, mirroring `learn_pooled_with_milp`. Defaults from
`UPSTREAM.md`: `pre_mip_epochs=50`, `mip_interval=1`, `mip_traces=3`,
`mip_time_limit=60`, `agreement_stop=1.0`. Per-domain `epochs` / `gamma` /
`lambda_` come from `_HYPERPARAMS` in `benchmark/baselines/rosame_i_runner.py`.

**Trace↔label alignment.** `mip_traces` samples a *subset* per round, so the
state labels only cover the sampled traces. Traces not in the round keep their
previous labels (and keep decaying) — they must **not** silently get zero-valued
targets. Track per-trace `rounds_since_label` for the ψ exponent.

**Fallback.** No feasible solution ever → return the plain ROSAME-I model with
`milp_failed=True` in the report, matching `RosameMilpRunner`.

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

1. **Shared object universe (the gatekeeper).** The CV head is a fixed-size
   proposition vector fixed at the first grounding; `_ensure_cv_model` raises if
   a later trajectory grounds differently. Adding the MILP does **not** relax
   this — but it does not tighten it either, since the MILP builds a *per-trace*
   `PSInstance` and tolerates heterogeneous groundings. The constraint is
   satisfied in the current data by construction, not by code: generation-mode
   corpora all come from one bundled problem (verified — every problem in
   `npuzzle_generated_problem0__final-version_fixed` declares identical
   `:objects`), and the external domains' authored problem sets use one fixed
   universe (verified for depot). **Re-verify per data dir before running**;
   blocksworld's predefined 1–10 set is the one most likely to vary.
2. **Images must exist on disk.** `_resolve_images` looks next to the problem
   PDDL first, then falls back to
   `src/domains/<bench>/problems/<problem>/` (depot/gripper live only there).
3. **Exactly `T+1` frames for `T` actions**, or the trace is skipped.
4. **Hyphen normalization.** Problem/GT identifiers are normalized
   hyphen→underscore before parsing against the normalized domain. Reuse
   `RosameIBaselineRunner._normalized_tempfile` and friends; do not re-solve it.
5. **Cost.** `n_seeds` × epochs × a CP-SAT solve per epoch after warmup is
   *much* more expensive than `rosame_i`. Respect `timeout_seconds` at both the
   seed level and the round level; consider `n_seeds=1` for the MILP arm and say
   so explicitly in the report rather than silently diverging from `rosame_i`.

---

## 8. Prerequisites before this can be *run*

1. **A `pisam_milp_single_round` run on imaged cells must exist.** It has never
   been run on any domain. No code is needed — `_load_masked_observations`
   already handles `pre_built_observations=None` as "load from disk (image
   pipeline)", and `run_cdps_phase` is data-source-agnostic. It is a
   `run_config.yaml` change: `source: image`, an image `data_dir`, and
   `algorithms: [pisam_milp_single_round]`. This is the comparison target and it
   is sequenced under the experiment-platform work, not here.
2. **Object-universe verification per image data dir** (see §7.1).
3. Confirm `rosame_i` itself runs on every domain you intend to compare on;
   coverage is uneven today.

---

## 9. Files touched

| File | Change |
|---|---|
| `benchmark/baselines/rosame_i_milp_runner.py` | **new** — the runner |
| `benchmark/algorithm_adapters/rosame_milp/milp_loop_i.py` | **new** — `MilpRosameI`, two-channel loop |
| `src/milp/converter.py` | **new fn** `cv_predictions_to_trace` + shared private tail |
| `benchmark/baselines/__init__.py` | registry entry |
| `benchmark/run_config.yaml` | `rosame_i_milp` in `algorithms`, image cell |
| `benchmark/evaluation/cfm/dashboard_config.yaml` | `modes: [image]` entry |

**Not touched:** `run_fold.py`, `benchmark/algorithms.py`, `src/milp/encoder.py`,
anything under `src/plan_denoising/`.

---

## 10. Verification

- **Unit** — `cv_predictions_to_trace` on a hand-built 2-step trace: assert
  `obs_p` probabilities survive unrounded, `init`/`goal` land in `hard_states`,
  and every action resolves. Mirror `test_rosame_milp.py`'s style.
- **Parity** — feed the converter a *ternarized* probability matrix derived from
  a real degraded observation and assert the resulting `ObservationT` is
  equivalent to `observation_to_trace`'s output on the same observation. This is
  the single strongest guard against the two entry points drifting.
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
