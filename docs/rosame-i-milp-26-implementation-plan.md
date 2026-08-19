# Plan: `ROSAME-I_MILP_26` — a faithful ICAPS-26 arm

Target: `xikaioliver/ROSAME` @ `ROSAME+MILP`, commit `95c733f` — architecture, loop and
hyperparameters, added as a new baseline alongside our existing `ROSAME-I_MILP`.

**No code in this document.** Decisions first, then the build order.

---

## 0. The thing to settle before anything else

**ICAPS-26 is unsupervised in the actions. Our benchmark observes them.** That is not a
hyperparameter — it is a different problem.

Evidence:
* The net predicts actions: `a_logit = self.action(cat([z_ext[:, :-1], z_ext[:, 1:]]))`,
  `a = action_activation(a_logit)` (`dl/model.py:123-125`), softmax over `adim()`
  (`dl/mixins/action.py:29`).
* Every loss term is *weighted by the predicted action distribution* —
  `(a[:, :-1, :] @ mse_prefix)` (`dl/model.py:171`), same in `loss_app` (`:190-194`).
* `DL_to_MIP` and `MIP_to_DL` both include `'action'` (`train_common.py:41-42`), and the MILP
  returns action pseudo-labels (`translator.extract_sol_label`).
* `run_fixer` calls `model_permutation(...)` before making pseudo-labels
  (`convertor/convertor.py:104`) — schema identity is only determined **up to a permutation**,
  which is exactly what you need when nobody told you which action was which.

So there are two coherent targets, and they answer different questions:

| | **A. Faithful-26** | **B. Adapted-26** |
|---|---|---|
| actions | predicted (unsupervised) | pinned to the observed action (one-hot) |
| `model_permutation` | required | unnecessary — schemas are named |
| MILP channels | state + action + model | state + model |
| comparable to CDPS / PI-SAM? | **No** — strictly harder problem | **Yes** — same information |
| answers | "how does the published ICAPS-26 system do on our data?" | "how does the ICAPS-26 *architecture* compare, given our observations?" |

**DECIDED: B is the default, A stays reachable behind a flag.** B is the arm that goes in the
comparison table — it matches the information our PI-SAM arms get. A is for a "we also ran the
published system as-is" paragraph.

**Implementation shape (keeps the difference to one line).** Compute `a_logit` exactly as
upstream does, and override only

```
a = action_activation(a_logit)   →   a = one_hot(observed_action)
```

Every `@` contraction in `loss_pred`, `loss_app` and `loss_prior` then works unchanged — it just
selects the observed row. Consequences, all mechanical: `loss_pseudo_a` drops out,
`model_permutation` becomes dead code (schemas are named), and `DL_to_MIP` / `MIP_to_DL` lose
`'action'`. Switching to A is "don't override `a`".

---

## 1. What "identical to upstream" can and cannot mean

Achievable exactly: architecture, loss terms and their weights, the MILP loop structure and
cadence, ψ decay, the read-out.

**Not achievable, and must be declared as deviations:**

| Upstream | Ours | Why |
|---|---|---|
| `epoch: 5000` | chosen by **step count**, not copied | see §1.1 |
| `batch_size: 128` | config value, `min(batch_size, N)` | **no small-data assumption** — see §1.1 |
| `lr: 1e-4` | matchable ✅ | — |
| `cp_type: mip-gurobi` | `cp-sat` | no Gurobi licence; the vendored factory already registers both |
| corpus-wide image normalisation | computed once over the whole `data_dir` | see §4 |
| `beta_reconst: 0` | same ✅ | reconstruction is off upstream by default anyway |

The honest framing for the thesis: *"the ICAPS-26 architecture, trained under our data regime."*

### 1.1 Batching: build for any N, not for today's N

The corpus will grow. Nothing in this plan may assume a small fold. Use a real `DataLoader`
with `batch_size` straight from config and effective batch `min(batch_size, N)`; steps per epoch
is then `ceil(N / batch)` and scales for free. Today's 3–8 traces simply yield one step per
epoch; at N = 500 you get four. That is upstream's own behaviour, not a special case.

Two consequences that follow from this rather than from small data:

* **Choose epochs by steps.** Upstream's 5000 epochs at batch 128 over a large corpus is
  ~5000·(N/128) optimizer steps. Copying "5000" at N = 8 gives 5000 steps — a different budget
  wearing the same number. This is the same trap that produced the ICAPS-24 epoch-semantics
  problem; decide the target in steps and derive epochs from N.
* **`TraceSelector` is FIFO with `capacity = mip_traces`**, so at large N it fills from the
  first batch(es) of each cycle and the shuffle seed starts to matter. Pin it.

---

## 2. Strategy: vendor, don't reimplement

We already vendor `planning_structs/` and `constraint_opt/` **verbatim from this exact commit**
(`src/milp/vendor/`, `UPSTREAM.md`). The MILP solver layer is done. What's missing is the DL
side and the loop glue.

Vendor these, verbatim where possible, under `src/milp/vendor/` (or a sibling `vendor_dl/`):

| Path | Verbatim? | Note |
|---|---|---|
| `dl/model.py` | yes | `ROSAMEGoal` + `Net` + losses |
| `dl/mixins/{encoder_decoder,action,action_model,output,pair,plot}.py` | yes | ResNet encoder, ConvTranspose decoder, action head |
| `dl/util/ROSAME/rosame.py` | **yes — and this matters** | their `Domain_Model` **differs from AMLGym's**: `params_types` sorted vs unsorted, different grounded-proposition string format, and a duplicate-object rejection in `ground()`. Do **not** substitute our `PORosame_Runner.rosame`. |
| `dl/util/{layers,util,dataset}.py` | yes | `dapply`, `TraceDataset` |
| `convertor/{convertor,selector,pseudo_label,translator}.py` | yes | the MILP loop |
| `util/model_perm.py` | yes | `model_permutation` (needed for A; inert for B) |
| `dl/network.py` | **no — replace** | see §3 |
| `dl/main/*`, `dl/util/tuning.py` | **no** | hyperparameter sweep + TensorBoard harness; not our runner's job |

Reimplementing `dl/model.py`'s loss by hand is the one thing I'd refuse to do — the
action-weighted `@` contractions and the γ placement (`loss_app` γ-weights the *first* step,
`loss_pred` γ-weights the *last*) are easy to get subtly wrong and impossible to notice.

---

## 3. Replace the harness, keep the model

`dl/network.py`'s `train()` is a full research harness: TensorBoard `file_writer`, `alive_bar`,
checkpointing to `best_model.pth`, a train/val split, `evaluate(val_data)` every 9 epochs,
`dump_actions()`. None of that fits a `BaselineRunner`.

What we need to re-express, and **only** this:

```
for epoch in range(epochs):
    for batch in loader:
        outputs = net(img_traces, action_traces, inits, goals)
        loss = model.loss(outputs, targets, indices, convertor.pseudo_labels)
        step
        if epoch >= pre_mip_epoch and epoch % mip_interval == 0:
            trace_selector.update(indices, outputs['z'], outputs['a'], inits, goals)
    if epoch >= pre_mip_epoch and epoch % mip_interval == 0:
        convertor.run_fixer(trace_selector, mip_time_limit)
```

That is `dl/network.py:265-305` with the logging removed. Preserve exactly:

* the **two** gating sites (`:272` in-batch selector update, `:303` post-epoch solve) — they
  are one condition evaluated twice, and both must move together;
* `TraceSelector` is **FIFO with `capacity = mip_traces`** and must be `clear()`ed per cycle;
* ψ decay lives in the loss, not the loop: `trace_labels[idx] = (weight * pseudo_weight_decay, ...)`
  (`dl/model.py:290`) — it decays **only** state/action labels, never `loss_pseudo_m`
  (already documented in our `UPSTREAM.md` §2).

Free bonus: `pre_mip_epoch >= epochs` disables the MILP entirely, which gives us the
**ICAPS-26 DL-only** arm from the same code with one parameter. That's the other half of the
2×2 we discussed.

---

## 4. The data adapter — the real work

Upstream's contract (`dl/main/rosame_full.py:10-25`, `dl/util/dataset.py:348-362`):

```
img_traces    (N, T, C, H, W)   float, /255, then normalize_traces(traces[:, 1:])
state_traces  (N, T, n_props)   clamped to [0,1]
action_traces (N, T, adim)
__getitem__   -> (img_trace, state_trace, action_trace, idx)
net(...)      <- (img_traces, action_traces, inits, goals)
```

Points that will bite:

1. **The first frame is dropped** — `img_traces[:num_examples, 1:]`. The initial state enters
   separately as `inits`, not as an image. Our `_resolve_images` returns all T+1 frames; the
   adapter must slice.
2. **`inits` and `goals` are given, hard.** Maps cleanly onto our GT init (problem `:init`) and
   GT final state — the same two anchors our current arm already uses. Good news: no new data
   requirement.
3. **`state_traces`** is *supervision* for their `state_accuracy` metric, not a training input
   in the DL-only path. In our imaged setting we do have degraded VLM states — feeding them
   would be a deviation (upstream trains from pixels). Recommend: use them for logging only.
4. **Image normalisation — DECIDED: compute once over the whole `data_dir`.**
   `normalize_traces` computes per-pixel mean/std over the dataset and stashes it in a global
   `parameters` dict. Computing it *per fold* would be worse than it sounds: over ~25–90 images
   the statistic is noisy, and the same problem would be normalised differently in fold 0 than
   in fold 3, so folds stop being comparable and the arm becomes sensitive to fold composition.
   Computing it once across all problems in the `data_dir` is fold-independent and closer to
   upstream's corpus-level statistic. It is a statistic-level touch of held-out pixels — standard
   practice, and it goes in the deviation register as a footnote.
5. **Resize.** Upstream's synth path uses `transforms.Resize(64)`; `ResNetMixin.autocrop_dimensions`
   crops to a multiple of 32. Our `_IMAGE_TF` already targets 64×64 — verify they agree rather
   than assuming.

---

## 5. Per-domain assets — generate all five from OUR domains

`Convertor.__init__` (`convertor/convertor.py:44-52`) loads, per domain:

* `planning_structs/specs/<domain>/domain.json` — the domain spec
* `pddl/<domain>/domain.pddl` — the ground-truth model, for `model_permutation`'s `mip_gt_dist`
  diagnostic only (`run_fixer` guards `gt_am is None`, so a missing one degrades logging, not
  correctness — and under option B the permutation is dead code anyway)

Shipped: `blocksworld`, `gripper`, `hanoi`, `8-puzzle`, `logistics`. **Correction to an earlier
draft of this plan: those four overlapping names are NOT usable.** They encode *their* domain
variants, not ours:

| | their spec | ours |
|---|---|---|
| hanoi | one `object` type; `clear`, `on`, `smaller`; one action `move` | typed `peg`/`disc`; `clear-disc`, `clear-peg`, `on-disc`, `on-peg`, `smaller-disc`, `smaller-peg`; four `move_*` actions |
| blocksworld | `arm-empty`, `on-table`, `pickup`, `putdown` | `handempty`, `ontable`, `pick_up`, `put_down` |
| gripper | `at-robby`, `at`, `free`, `carry` / `move`, `pick`, `drop` | identical — the one that happens to match |

**DECIDED: write a generator that emits the spec JSON from a parsed `pddl_plus_parser` Domain,
and run it over all five of our `src/domains/*.pddl`.** The format is trivial — hanoi's is 331
bytes of `name` / `types` / `predicates` / `action_schemas` — so the generator is ~30 lines and
it retires the entire "which domain variant is this" bug class. Depot then stops being special:
it is the fifth output of the same generator, not a hand-written exception.

Also extend the hardcoded alias `domain_name = "blocksworld" if domain_name == "blocks"` to our
bench keys.

### 5.1 Does our ICAPS-24 arm have this problem? No — structurally

AMLGym's `prepare_rosame` (`rosame_runner.py:228-235`) builds the `Domain_Model` **directly from
the parsed PDDL**:

```python
types      = self._prepare_types(self.domain.types)
predicates = self._prepare_predicates(self.domain.predicates)
actions    = self._prepare_action_schema(self.domain.actions)
self.objects = self.get_objects(self.problem.objects)
```

`self.domain` is `DomainParser(domain_file).parse_domain()` on *our* file. No JSON spec is
involved, so there was never anything to adjust — hanoi's six typed predicates, blocksworld's
`handempty`/`pick_up`, depot's `at-crane`/`on-pile` are all derived automatically. Object
universes were handled deliberately too: `ground_union` unions every problem's objects across
the fold, grounds once, and asserts stability (`docs/rosame-i-implementation-plan.md` §7.1).

**One thing to verify rather than assume.** `check_predicate` (`rosame_runner.py:254`) falls
back to trying *permutations* of the arguments when a predicate string isn't found verbatim.
For same-typed args both orderings exist as distinct propositions so the exact match wins; for
mixed types only one permutation is type-valid, so it also resolves. It is probably sound — but
it is a fuzzy match on the path that builds the GT final-state vector, and a wrong match there
would silently corrupt the γ anchor. Add a test that round-trips every grounded predicate of
each domain through `check_predicate` and asserts identity. Cheap, and it retires the question
for both the 24 and 26 arms.

---

## 6. Parameters to pin (from `train_common.py`)

```
aae_width 1000   aae_depth 3   feature_dim 512   hidden_dim 256
lambda 0.2   gamma 10   beta_pred 1   beta_app 1   beta_reconst 0
optimizer Adam   lr 1e-4
epoch 5000*  batch_size 128*
pre_mip_epoch 50   mip_interval 1   mip_traces 3
pseudo_weight_decay 0.99   mip_time_limit 60
DL_to_MIP [state, action, model]   MIP_to_DL [state, action, model]*
cp_type mip-gurobi*
```

`*` = the five we cannot or should not match; see §1 and §0. Everything unstarred is pinned
verbatim. Note λ and γ **match our existing arms already** — no change there.

**Reconstruction — DECIDED: vendor the decoder, keep `beta_reconst: 0`.** That is upstream's
released configuration exactly. Practical consequence worth knowing: with the weight at zero,
`loss_reconst` contributes no gradient, so `feature_decoder` *and* `feature_composer` are inert
— they cost memory and init time, nothing else. This also retires an earlier worry in this plan
about the AAE's parameter count versus our data: the trainable path is encoder → symbol_net →
sigmoid plus the action head and schema MLPs, barely more than ICAPS-24. Keeping the modules
buys a one-parameter ablation (`beta_reconst: 1` → "does reconstruction help at our scale?") for
free. Ignore `GaussianOutput` in `dl/mixins/output.py`: `StateAE` installs `VanillaRenderer` and
`loss_reconst` is plain MSE, so it is unused.

**MILP cadence.** This is the *same* mechanism our existing arms already have — `pre_mip_epochs
50`, `mip_interval 1`, solves = `epochs − 50` (hence hanoi's observed 21 rounds at 70 epochs) —
and `_solve_time_limit` (`milp_loop_i.py:145`) already divides the remaining budget over the
remaining solves. What changes at 26's scale is not safety but *fidelity*: schedule ~4950 solves
and the guard silently doubles `mip_interval` until each solve can afford `_MIN_SOLVE_SECONDS`,
quietly turning `mip_interval: 1` into 64. So: pick the epoch count so the natural solve count
fits the budget, and **assert `mip_interval_used == mip_interval`**, failing the row loudly if
the guard had to intervene. That converts a silent deviation into a visible one.

Critically: **do not reuse `_HYPERPARAMS`** (the ICAPS-24 per-domain 70/100/300 table). ICAPS-26
is domain-independent at 5000 epochs. Mixing the two tables would be exactly the kind of silent
assumption we're trying to eliminate.

---

## 7. Integration

* New runner class in `benchmark/algorithm_adapters/rosame_milp/` (the loop) + a
  `BaselineRunner` in `benchmark/baselines/`, registered as `rosame_i_milp_26`.
* Row name `ROSAME-I_MILP_26`; the DL-only variant `ROSAME-I_26` (via `pre_mip_epoch ≥ epochs`).
* Work subdir per row label, same rule as `milp_work_subdir` — so it cannot collide with the
  existing arm's models.
* It is a *baseline*, so it reaches cells through `backfill_baseline`, and inherits the
  `--learn-timeout` default of 300 that we already have to override manually.
* Skip cleanly on simulation-mode cells (no images), same as `rosame_i`.

---

## 8. Deviation register (ship this in the thesis)

Every item here is a place we knowingly differ. Keeping the list short and explicit is the
whole point of the exercise.

1. Actions observed, not predicted (option B) — *the* deviation; state it first.
2. Training budget chosen by step count, not 5000 epochs × batch 128.
3. CP-SAT instead of Gurobi.
4. Image normalisation computed over the whole `data_dir` rather than a held-out-clean split.
5. `state_traces` used for logging only.
6. All five domain specs generated from our `src/domains/*.pddl`, none reused from upstream.

## 8.1 What this arm will *not* fix

Set expectations before the run, so a null result reads as confirmation rather than
disappointment. The gripper failure is an **identifiability limit, not an optimisation
failure**: `free(g)` is true at both hard-anchored endpoints, so `M_true` (pick deletes `free`,
drop adds it) and `M_static` (nothing touches `free`) are *observationally equivalent* given the
init anchor, the goal anchor and the observed actions. They differ only in the soft interior,
where the CV head has little signal. No encoder change creates information that is not in the
pixels.

So: predict in advance that the 26 arm does **not** fix gripper's `false_plans_ratio = 1.000`.
If it does, that is evidence the AAE encoder genuinely extracts `free` where ICAPS-24's head did
not — which is a more interesting finding than the one we were looking for. Either outcome is
publishable; an unstated expectation is not. Full write-up in
`docs/algorithm_comparison_analysis.md` §3.2.

---

## 9. How we'll know the port is right

In order, each cheap and each falsifiable:

1. **Shape/parity test** — one trace through their `Net.forward` and through our adapter; assert
   `z`, `a_logit`, `z_suc_aae`, `p_applicable` shapes and that `z ∈ [0,1]`.
2. **Loss-parity test** — construct a tiny synthetic trace, compute `loss_dict` under the
   vendored `dl/model.py` and under our harness; assert equality to float tolerance. This is the
   test that actually protects the port.
3. **MILP-parity test** — one `run_fixer` call, assert the pseudo-labels it produces match a
   direct invocation of the vendored translator.
4. **Degenerate-model guard** — the same assertion I'd want on ROSAME-I: reject a learned model
   with zero add-and-delete effects across all schemas, loudly. We would have caught the
   empty-effects collapse on day one with this.
5. **Sanity run** — one blocksworld fold, DL-only mode, `pre_mip_epoch ≥ epochs`. If effects are
   non-empty here and were empty for ICAPS-24, that alone is a publishable result.

---

## 10. Order of work

| Phase | Deliverable | Risk |
|---|---|---|
| 1 | Vendor `dl/` + `convertor/` + `util/model_perm.py`; import-only smoke test | low |
| 2 | Data adapter (fold → their contract) + shape-parity test | **medium — most of the work** |
| 3 | Harness replacement (§3) + loss-parity test | medium |
| 4 | DL-only arm (`ROSAME-I_26`), one blocksworld fold | low |
| 5 | Turn the MILP on (`ROSAME-I_MILP_26`), one fold, verify `mip_gt_dist` logs | medium |
| 6 | Depot spec JSON | low, fiddly |
| 7 | Full grid across 5 domains via backfill | compute only |

Phase 4 is a real milestone: it gives the old-vs-new DL comparison *before* any MILP work, and
it's where the empty-effects question gets answered.

---

## 11. Decisions taken (2026-08-17)

1. **Option B is the default; A behind a flag** (§0). One-line difference: override `a` with a
   one-hot of the observed action.
2. **Batching is a config value supporting any N** (§1.1). No small-data assumption anywhere.
   Epoch count derived from a target *step* count, not copied from 5000.
3. **Reconstruction: vendor the decoder, `beta_reconst: 0`** (§6) — upstream's own setting, and
   it keeps a free ablation.
4. **Image normalisation: once over the whole `data_dir`** (§4.4).
5. **MILP cadence: same mechanism as our current arms**; the new requirement is asserting
   `mip_interval_used == mip_interval` so the budget guard can't silently rewrite it (§6).
6. **All five domain specs generated from our own PDDL** (§5). Depot is not a special case.

### Still open

* **Wall-clock per cell.** Everything else is settled; this one sets the epoch count via §1.1.
  Needs a number from you before Phase 5.
