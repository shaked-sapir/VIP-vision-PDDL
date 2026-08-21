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
selects the observed row. Switching to A is "don't override `a`".

**The consequences are not all mechanical.** Three of them need a decision, each verified against
`95c733f`:

| | what upstream actually does | what B requires |
|---|---|---|
| `loss_pseudo_a` | **survives.** `model.py:299` computes it unconditionally: `preds_a` = `a_logit`, targets = `self.pseudo_label(outputs, goals)` — the *action-model-based* argmax at `:223`, not anything the MILP produced. The `if 'action' in MIP_to_DL` block only *overwrites* rows for MILP-labelled traces. Dropping `action` from `MIP_to_DL` therefore does **not** disable it | **replace the target, don't delete the term.** Under B we know the true action, so cross-entropy against the *observed* one is both better-founded and closer in spirit than distilling against a possibly-wrong inference. Deleting it instead would silently change the encoder's regularisation, since the gradient runs `a_logit → action → z_ext → z → encoder`. Registered as a deviation (§8) |
| `model_permutation` | **not dead**, and **not a name matcher.** `run_fixer` calls it unconditionally on the *pseudo-label* path against `obs_m`; its outputs `name_dl_cp, args_dl_cp` go straight into `extract_sol_label`. Only the *second* call, against `gt_am`, is the diagnostic | **bypass it — pass explicit identity mappings.** See §0.1: it maximises numerical agreement over a permutation search and never compares names, so under B it can and will return a non-identity mapping |
| `DL_to_MIP` / `MIP_to_DL` | `objectives` (= `DL_to_MIP`) is handed to `problem_builder(...)` and selects which agreement terms enter the CP objective. `trans_obs_tr` builds `obs_a` from `a` **regardless** | **only `MIP_to_DL` drops `action`** (those labels we already have). `DL_to_MIP` **keeps** it — under B `a` is one-hot at `1-eps`, so that channel is precisely how the observed actions reach the solver. Dropping it leaves `act[(i,t,a)]` free, constrained only by applicability and frame axioms: an arm strictly *weaker* than either A or B |

**And make the action channel a hard constraint, not an objective term.** Under B the actions are
data, not belief; an objective weight lets the solver buy model simplicity by inventing a
different action sequence. Ship it with an infeasibility diagnostic — hard-fixing endpoints is
exactly what produced the goal-fluent infeasibility we spent a week on, and the same failure mode
applies here.

### 0.1 Why `model_permutation` must be bypassed rather than asserted

`util/model_perm.py:71` is a **search that maximises numerical agreement**, not a check that names
line up. It enumerates every schema-name permutation passing `sig_match` and every argument
permutation passing `type_match`, scores each with `perm_agree`, and returns the arg-max. It
**never compares names for equality**. So "the schemas are named, therefore it returns the
identity" does not follow, and an assert-identity test would fail rather than protect:

* `sig_match` compares **sorted** type lists (`model_perm.py:13-18`). Our hanoi's `move_disc_peg`
  and `move_peg_disc` both sort to `[disc, disc, peg]`, so **swapping them is a legal candidate**.
* `type_match` permutes same-typed argument positions (`:21-25`). `move_disc_disc` has three
  `disc` parameters, so all 3! orderings are candidates.
* `perm_agree` scores against `obs_m`, the **DL's current** model. Before the network has learned
  anything, agreement is near-chance across candidates and the arg-max is effectively arbitrary —
  and this runs from `pre_mip_epoch`, i.e. exactly when the model is still noise.

A wrong mapping is silent: `extract_sol_label` uses it via
`instance.get_permuted_action(action, name_dl_cp, args_dl_cp)`, and `extract_sol_model` via
`get_permuted_action_schema(...)`, so every pseudo-label is relabelled and nothing downstream
notices.

**DECIDED: under option B, do not call the search. Pass identity mappings directly**, which the
same data structures already express:

```
name_dl_cp = {n: n for n in schema_names}
args_dl_cp = {name: tuple(range(1, arity + 1)) for name in schema_names}   # x_cp == x
```

Keep the *second*, GT-facing call for the `mip_gt_dist` diagnostic only. Under option A the search
is genuinely required and comes back — one more reason A is a separate phase (§7), not a flag.

---

## 1. What "identical to upstream" can and cannot mean

Achievable exactly: architecture, loss terms and their weights, the MILP loop structure and
cadence, ψ decay, the read-out.

**Setting by setting** — ✅ matched verbatim, ⚠️ declared deviation (the ⚠️ rows are exactly the
deviation register in §8):

| Upstream | Ours | | Why |
|---|---|---|---|
| `lr: 1e-4` | `1e-4` | ✅ | — |
| `beta_reconst: 0` | `0` | ✅ | reconstruction is off upstream by default anyway |
| `mip_traces: 3` + FIFO `TraceSelector` | verbatim | ✅ | §1.1 |
| `batch_size: 128` | verbatim; effective batch `min(batch_size, N)` | ✅ | **no small-data assumption** — §1.1 |
| actions predicted (unsupervised) | actions **observed** (option B) | ⚠️ | §0 — the deviation that matters most |
| `epoch: 5000` | `5000` is the code default; the grid runs a **calibrated** value inside the 600 s cell budget, plus one 5000-epoch control cell per domain | ⚠️ | §1.1, §1.2 |
| no seeding; one run per config | `n_seeds` (our harness) | ⚠️ | §1.3 — multiplies the budget directly |
| `device: cuda else cpu` | pin `cuda` else `cpu`; **never MPS** | ⚠️ | §1.3 |
| per-domain image augmentation keyed on `parameters["domain"]` | **not ported** | ⚠️ | §3 — every one hard-asserts a layout our renders do not have |
| `num_workers: 64, persistent_workers` | sized from the fold | ⚠️ | §1.1, performance only |
| `cp_type: mip-gurobi` | `cp-sat` | ⚠️ | no Gurobi licence; the vendored factory already registers both |
| corpus-wide image normalisation | computed once over the whole `data_dir` | ⚠️ | §4.4 |
| image resize: **none** (native) | per-domain configurable, default `Resize(64)` (int, aspect-preserving); `resize: null` reproduces upstream exactly | ⚠️ | §4.6 — a variable to experiment on, not a constant |

The honest framing for the thesis: *"the ICAPS-26 architecture, trained under our data regime."*

### 1.1 Batching: build for any N, not for today's N

The corpus will grow. Nothing in this plan may assume a small fold. Use a real `DataLoader`
with `batch_size` straight from config; steps per epoch is `ceil(N / batch)` and scales for free.
Today's 3–8 traces yield one step per epoch; at N = 500 you get four.

`min(batch_size, N)` is **not an adaptation of ours** — `dl/network.py:233` already does
`batch_size = min(batch_size, len(train_dataset))`. Copy it and claim nothing.

**Do not copy the DataLoader's worker settings.** Upstream uses `num_workers=64,
prefetch_factor=8, persistent_workers=True` (`network.py:236-239`), sized for a large corpus on a
GPU box. Spawning 64 persistent workers to serve a fold of 3–8 traces costs more than the work
itself, and `persistent_workers` holds them across all 5000 epochs. Set workers from the fold
size (`0` at our N) and record the value — this is a performance deviation, not an algorithmic
one, so it goes in the deviation register as a footnote.

**DECIDED: `epoch: 5000` stays as the verbatim default and is overridable per run.** The corpus
size is not known in advance, so a step-derived default would be guessing on the operator's
behalf. Upstream's number is the honest default; the operator lowers it when the data or the
budget calls for it.

**DECIDED: `TraceSelector` is pinned to upstream behaviour** — FIFO, `capacity = mip_traces` (3),
`clear()`ed once per cycle (`dl/network.py:256`). No re-selection strategy of our own. At large N
it fills from the first batch(es) of each cycle, so the shuffle seed matters: pin it.

One caveat the epoch default carries, which must be surfaced rather than absorbed. Be aware that
"epoch" means different things on the two sides: upstream's epoch is a full pass at batch 128
over a large corpus; at N = 8 it is a single optimizer step. Copying 5000 copies the *number*,
not the training budget. That is fine as a default — it just means the number should be read as
"upstream's setting", not as "a calibrated budget".

### 1.2 Pre-flight budget check (required by the 5000 default)

`epoch 5000` + `pre_mip_epoch 50` + `mip_interval 1` schedules **4950 CP-SAT solves per cell**.
At `mip_time_limit: 60` that is up to ~82 hours per cell, times 30 cells times 5 domains. And
because §6 requires `mip_interval_used == mip_interval`, the faithful default would make *every*
cell fail that assertion rather than quietly deviating.

That is the correct failure, but not a useful one. So at runner start, compute

```
projected_solves  = (epochs - pre_mip_epoch) / mip_interval
projected_seconds = projected_solves * per_solve_estimate + epochs * per_epoch_dl_estimate
```

and if it exceeds the cell's learn timeout, **refuse to start** with a message naming values that
would fit (`epochs ≤ X`, or `mip_interval ≥ Y`). Upstream's number stays the documented default;
the operator is told exactly what to override, before burning a grid instead of after.

`per_solve_estimate` is a **measured** figure, **not `mip_time_limit`** — see the table below for
why that distinction decides whether the check is usable at all.

**The cell's learn timeout is `learning_timeout_seconds: 600`** — the same
budget every other arm in the grid already gets. Chosen deliberately over a larger arm-specific
budget: the grid's entire point is a like-for-like comparison, and an arm that is handed 8× the
wall clock of its competitors is not being compared to them. If the 26 arm needs more time than
CDPS and PI-SAM to reach a comparable model, that *is* a finding, and it should appear as a
finding rather than be papered over by a bespoke budget.

What 600 s buys, using the **measured** cost rather than the nominal `mip_time_limit` (630 CP-SAT
samples, median solve **0.318 s**; the 60 s cap never binds):

| | nominal (`mip_time_limit` 60 s) | measured (0.318 s/solve) |
|---|---|---|
| 5000 epochs → 4950 solves | ~82 h | ~26 min MILP + DL ≈ **~70 min/cell** |
| fits in 600 s | epochs ≤ 60 | **epochs ≈ 700–750** (before `n_seeds`) |

This is why `per_solve_estimate` above is calibrated rather than read off `mip_time_limit`:
projecting from the cap would refuse a configuration that in fact finishes in a tenth of the
budget (`epochs ≤ 60` versus the ~700–750 that actually fit). Concretely — seed the estimate at
0.318 s, measure the running median over the first 20 solves, and re-project once against the
remaining wall clock.

**Grid `epochs` is a conservative round number the check is allowed to lower, not a derived
constant.** The 5000-epoch cost above is itself an estimate, so pinning a precise epoch count to
it manufactures false precision — solving the equation exactly gives ~730, and that number would
still be wrong the moment adapter, grounding, decode and eval time are counted. Set the config
value at **`600`**, let the pre-flight lower it, and let each row record the epochs it actually
ran. If the check never binds, raise the config value; do not raise it by arithmetic.

Plus **one unbudgeted control cell per domain at `epochs: 5000`** (~70 min each, ~6 h total),
run outside the timeout. That control is what keeps the equal-budget choice honest — without it,
"the 26 arm underperforms" and "the 26 arm was undertrained" are indistinguishable.

**Do not ration `mip_time_limit`.** Upstream passes the constant: `convertor.run_fixer(
trace_selector, self.parameters["mip_time_limit"])` (`network.py:303`) — there is no per-solve
scheduler upstream at all. Our `_solve_time_limit` (`milp_loop_i.py:145`), which divides the
remaining budget over the remaining solves and floors each at `_MIN_SOLVE_SECONDS = 5`, is our own
invention. Reusing it here would make decision 4 and decision 9 jointly unsatisfiable: 550 solves
× a 5 s floor is 2750 s against a 600 s cell, so the guard would widen `mip_interval` and trip the
assertion on **every** cell. Pass the constant, and let this section's pre-flight own the budget
instead. That is both the faithful choice and the one that removes the contradiction.

### 1.3 Run-level settings the plan must pin explicitly

None of these is in `train_common.py`'s parameter dict, which is exactly why they get decided by
accident:

* **Seeds.** Upstream has no seed key and runs each config once; our arms run `n_seeds` and keep
  the lowest final loss. That multiplies every number in §1.2 by `n_seeds`. **DECIDED: `n_seeds`
  applies as it does to our other arms, and the §1.2 projection is multiplied by it before the
  refuse-to-start check** — otherwise the check passes and the grid still overruns by 3×.
* **Device.** Upstream is `'cuda' if torch.cuda.is_available() else 'cpu'` (`train_common.py:34`).
  Our default is cuda > mps > cpu, and MPS has already produced one silent-crash class in this
  repo. **DECIDED: cuda else cpu, MPS excluded, recorded in `run_params`.**
* **Augmentation transforms.** Upstream selects a per-domain `TraceDataset` transform inside
  `_run_training` — the method we replace outright (§2.1). It therefore cannot fire unless we
  copy it, and we do not: our datasets are rendered scenes at our own resolutions, and every one
  of upstream's transforms hard-asserts a layout ours do not have. §3 says what not to port and
  why. **No grid-render code path exists on our side and none is being added.**

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
| `convertor/{selector,pseudo_label,translator}.py` | yes | FIFO selector, label store, DL↔CP translation |
| `convertor/convertor.py` | **no — needs asset placement** | two unguarded loads in `__init__`; see §2.1 |
| `util/model_perm.py` | yes | `model_permutation` (needed for A; inert for B) |
| `dl/network.py` | **yes — vendor, then subclass** | it is the base class; see §2.1 |
| `dl/main/normalization.py` | **yes** | `normalize_traces` lives here and we need it (§4.4) |
| `dl/util/tuning.py` | **no — stub the `parameters` global** | it is a grid-search harness; see §2.1 |
| `dl/main/{common,rosame_full}.py` | **no** | argparse subcommands + `.npy` loader; see §2.1 |

### 2.1 The vendoring boundaries, precisely

**`dl/network.py` must be vendored — it is the base class.** `ROSAMEGoal` inherits it
(`ROSAMEGoal → StateAE → AE → Network`), and `dl/model.py` leans on it throughout:
`self.parameters[...]` in *every* loss term (λ, γ, β's, `MIP_to_DL`, `pseudo_weight_decay`),
`add_metric`, `local()`, `save()`/`load()`, and the `_build_around` / `_build_primary` /
`build_aux` scaffolding that constructs the net at all. So: vendor it verbatim and **subclass,
overriding only `_run_training` / `train`** (lines 217–334). That is where the TensorBoard
`file_writer`, `alive_bar`, `best_model.pth` checkpointing, the `evaluate(val_data)` every 9
epochs, `dump_actions()` and `resume()` live — none of which fit a `BaselineRunner` that already
owns timeouts, work dirs and result rows. Everything outside that one method stays upstream.

**`dl/util/tuning.py` is a hyperparameter *search* harness**, not the algorithm:
`train_common.py` states it plainly — *"If the value is a list, it is interpreted as a
hyperparameter choice… a separate experiment will be run and recorded."* We do not want a sweep
running inside a fold. **But** `dl/main/normalization.py` imports `from ..util.tuning import
parameters` — the global that caches the image mean/std — so that one coupling has to be stubbed
with a plain dict rather than dropped.

**`dl/main/{common,rosame_full}.py` are the CLI layer**: argparse subcommands and a loader that
reads `traces.npy` / `states.npy` / `actions.npy` from `dl/data/`. Our adapter (§4) replaces the
loader, but should **reproduce `rosame_full.load_data`'s preprocessing sequence exactly** —
`/255` → drop frame 0 → `normalize_traces` — rather than inventing its own. Note that
`load_data` also stashes `parameters["picsize"]`, a second unkeyed global alongside the mean/std
pair in §4.4; both need the same treatment.

**`convertor/convertor.py` cannot be dropped in as-is** — not because of its logic, which we do
want verbatim, but because `Convertor.__init__` performs **two unguarded filesystem loads** at
paths relative to the vendor parent:

```
spec_path = <vendor>/planning_structs/specs/<domain_name>/domain.json   -> load_domain_from_json
gt_am     = <vendor>/pddl/<domain_name>/domain.pddl                      -> parse_pddl_domain
```

Neither is behind a guard, so both must exist before the first `Convertor(...)` is constructed —
and it *is* constructed on every training run (`network.py:245`), including MILP-free ones. The
fix is asset placement, not code surgery: emit `specs/<domain>/domain.json` per §5 and drop a
`pddl/<domain>/domain.pddl` symlink or copy pointing at `src/domains/<domain>.pddl`. Also extend
the hardcoded `domain_name = "blocksworld" if domain_name == "blocks"` alias to our bench keys.

Reimplementing `dl/model.py`'s loss by hand is the one thing I'd refuse to do — the
action-weighted `@` contractions and the γ placement (`loss_app` γ-weights the *first* step,
`loss_pred` γ-weights the *last*) are easy to get subtly wrong and impossible to notice.

---

## 3. Override one method, keep everything else

Per §2.1: vendor `dl/network.py`, subclass it, and override **only** `_run_training` / `train`.

What that override needs to re-express, and **only** this:

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
* `trace_selector.clear()` at `:256`, once per cycle, before collection;
* ψ decay lives in the loss, not the loop: `trace_labels[idx] = (weight * pseudo_weight_decay, ...)`
  (`dl/model.py:290`) — it decays **only** state/action labels, never `loss_pseudo_m`
  (already documented in our `UPSTREAM.md` §2).

**Read `pseudo_weight_decay` carefully before reproducing it — it does not do what the name
suggests.** `weight_mask_a` (`model.py:272`) is initialised to **ones**, and only MILP-labelled
rows are overwritten with the decaying `weight`. So `loss_pseudo_a` is a *full-strength* term on
every trace from epoch 0, targeting the network's own `pseudo_label()` output, and the decay
progressively down-weights **MILP labels relative to the network's self-labels** rather than
fading the pseudo-label pressure overall. `loss_pseudo_s` is the opposite: zero without MILP
labels, then divided by `len(trace_labels)`. Reproduce both asymmetries deliberately, or you
will have reproduced the name and not the algorithm. Under option B this is also the term §0
retargets at the observed action.

**And the decay's effective strength is a function of N.** `PseudoLabels.update`
(`convertor/pseudo_label.py:24`) writes `self.traces[idx] = (1, state_label, action_label)` — the
weight is **reset to 1** every time a trace is re-labelled. So `pseudo_weight_decay` only bites in
the gap between one selection of a trace and the next. At upstream's corpus size a trace may wait
many cycles and decay meaningfully; at our N = 3–8 with `mip_traces = 3` most traces are
re-selected almost every cycle, leaving the decay close to inert. The parameter matches upstream;
the behaviour it produces does not. Report it that way rather than listing ψ decay as matched.

**Do not port the transform-selection block** at the head of `_run_training`
(`network.py:218-231`). It picks a `TraceDataset` augmentation from `parameters["domain"]` —
`BlocksworldPilePermute` / `ColumnPermute` for `blocks`, `RoomBallPermute` for `gripper`,
`ItemPermute` for `logistics`, plain otherwise. Write `TraceDataset(train_data)` and move on.

Two reasons, and only the second is subtle:

1. **They cannot run on our data.** Each hard-asserts a layout: `RoomBallPermute` a 4×6 grid of
   16 px cells (64×96 exactly), `ColumnPermute` a 6×5 grid (96×80), `BlocksworldPilePermute`
   64×64 with a specific renderer's table and robot heights baked in as fractions. Our renders are
   none of these at any resize setting, so they raise rather than corrupt.
2. **`gripper` is the one domain key that matches one of our bench names**, and there is exactly
   one resize value — `[64, 96]` — at which `RoomBallPermute` would pass its guard and then
   permute rectangles of a scene that has no cell structure, leaving the symbolic labels intact.
   Nothing downstream would catch that. Since the block lives in the method we replace, the risk
   only exists if someone ports it "for fidelity" later; this paragraph is the note that stops
   them.

For the record, so nobody re-derives it: these are **label-preserving object permutations** — a
real data multiplier upstream gets and we do not. Reproducing the idea for our renders would mean
re-deriving each permutation against our own scene geometry. That is a research task, not a port,
and it is out of scope here. It is *not* a gap against the ICAPS-24 arm we compare to (see §5.1).

Free bonus: `pre_mip_epoch >= epochs` disables the MILP entirely, which gives us the
**ICAPS-26 DL-only** arm from the same code with one parameter. That's the other half of the
2×2 we discussed.

---

## 4. The data adapter — the real work

### 4.1 The exact shape contract, and why our data is one frame too long

Read off `rosame_full.load_data:11-24`, `network.py:262-263`, `model.py:123` and
`model.py:322` (`state_accuracy` compares `targets[0][:, 1:-1]` against `z`), the **raw arrays**
upstream loads are:

| array | raw length | after `img_traces[:, 1:]` |
|---|---|---|
| `traces.npy` (images) | L | **T = L − 1** |
| `states.npy` | L + 1 | T + 2 |
| `actions.npy` | L | T + 1 |

So upstream's raw images and actions are the **same** length, and states are one longer. Ours are
not: `rosame_i_runner.py:190` asserts **N + 1 images for N actions**. We have one image too many.

The reconciliation is forced. Set L = N — that is, **drop our last frame before applying
upstream's own frame-0 slice**:

```
init   = GT s_0                         (enters as `inits`, not as an image)
images = frames 1 .. N-1                (T = N-1)
goal   = GT s_N                         (enters as `goals`, not as an image)
actions = a_1 .. a_N                    (T+1 = N)  ✅
states  = s_0 .. s_N                    (T+2 = N+1) ✅
```

Three consequences, all of which must be stated rather than discovered:

1. **Both endpoint images are unused** — frame 0 by upstream's slice, frame N by ours. Only their
   *symbolic* GT states enter. The 26 arm therefore sees **N − 1** images where our 24 arm sees
   **N + 1**. That is an information difference between the arms, not a formatting detail, and it
   lands squarely on the Phase-4 comparison (§10).
2. **§4.5's preprocessing table carries both drops**, not just frame 0.
3. **A 2-image trace gives T = 0** — no images at all. Same degenerate class that already
   required the "skip the state channel when a trace has no interior frame" fix on the 24 arm.
   Guard it explicitly and skip the trace loudly, do not let a zero-length tensor propagate.

### 4.2 Which grounding defines the proposition space

`Convertor.__init__` builds `Instance(self.domain, self.objects)` from the **upstream**
`planning_structs.Instance`, whose `_build_propositions` enumerates
`permutations(self.objects, p.arity)` — *distinct objects only*, so `on(a, a)` does not exist.
Our arms use `RepeatedArgsInstance` (`src/milp/converter.py:79`) precisely to restore those.

Vendoring `convertor/` verbatim therefore gives the 26 arm a **different `n_props`** from both
`pisam_milp_*` and our 24 arm. Self-consistent inside the arm, and fine for its own
`solving_ratio` — but every cross-arm precision and recall number would then be computed over a
different vocabulary, which is a threat to the headline table rather than a footnote.

**DECIDED: use the upstream `Instance` (repeated args excluded) inside the 26 arm, and register
it** — matching upstream is the point of this arm, and silently widening its proposition space to
match ours would be the deviation. But: report the 26 arm's `n_props` next to the others, and add
a test that `trans_full_state`'s zip of the state vector against `instance.propositions` uses the
*same object and the same index order* as the DL's symbol vector. That zip is unguarded, and a
mismatch there mislabels every proposition silently.

That settles which grounding *function* is used. It does not settle **what object set is
grounded**, which is a separate axis and is still open — §4.2a.

### 4.2a OPEN AND BLOCKING: one grounding for the run, or one per problem?

**Nothing in Phase 2 or Phase 4 may start until this is pinned.** It is not a Phase-5 detail; it
fixes the shape of the DL head, so it is upstream of both `ROSAME-I_26` and `ROSAME-I_MILP_26`.

Upstream grounds **once**, and shares that one `Instance` across every trace in the bundle:

```python
# convertor/convertor.py:48-49, :69
self.objects  = [(o, t.name) for t, obj_list in rosame_domain.objects.items() for o in obj_list]
self.instance = Instance(self.domain, self.objects)
...
return self.problem_builder(self.domain, Traces(self.instance, obs_m, obs_t), self.objectives)
```

The DL symbol head is sized from that same object set, which is what makes the unguarded
`zip(self.instance.propositions, probs)` (`translator.py:66`) safe there.

Our 24 arm grounds **per problem** — `build_ps_instance(ps_domain, partial_domain, problem)`
(`rosame_i_milp_runner.py:155`) with `Traces(instance=None, ...)` (`:193-197`), each `obs_t` built
against its own instance. The union-width CV columns a problem lacks are dropped at that boundary
(`converter.py:473-486`); the `N of M CV propositions have no counterpart` warning is that drop
being reported.

Upstream never has to choose, because its corpora are object-homogeneous. **Ours are not.**
Blocksworld problems carry 4 or 5 blocks, so on a 4-block trace 11 of the union's 36 propositions
name the absent block `e`.

| | one grounding (upstream) | per-problem groundings (our 24 arm) |
|---|---|---|
| instances | one, over the corpus union | one per problem |
| absent-object propositions | **live CP variables**, pinned false by the hard init + frame axioms | do not exist |
| DL ↔ CP width | equal by construction | CV head is union-width, CP narrower; the converter bridges |
| encoding size | +44% on a 4-block blocksworld trace | minimal |
| fidelity | verbatim | a deviation — register it in §8 |

Expected behavioural difference: **none**. The phantom propositions are constrained only by the
closed-world hard init and the frame axioms, and no observed action binds `e`, so they form a
variable block decoupled from the lifted schema variables — binding runs through the observed
action's arguments. The cost should be a constant objective offset, not a distorted model. **That
is a prediction, not a measurement**, and it is the same class of claim this plan has already been
wrong about twice.

What makes it blocking rather than deferrable is everything downstream that the choice sizes:

* the DL symbol head's output width, and therefore the shape of every checkpoint;
* `extract_sol_label`'s `torch.zeros(..., len(self.instance.propositions))` (`translator.py:129`)
  and the `loss_pseudo_s` BCE it feeds;
* whether `cv_predictions_to_trace`'s drop-and-fill path is on the 26 path at all;
* the vocabulary every cross-arm precision and recall number is computed over.

Discovering it in Phase 5 means rebuilding the adapter and re-running everything measured before
it.

**Independent of which way it goes: the union must be taken over the whole `data_dir`, not the
fold.** Measured across all 30 cells of each of the five image experiments:

| domain | per-problem object counts | fold-union `n_props` |
|---|---|---|
| **blocksworld** | **{4, 5}** | 36 in all 30 cells |
| depot | {10} | 49 |
| gripper | {10} | 28 |
| hanoi | {7} | 55 |
| npuzzle | {17} | 153 |

Stable today — but by *composition*, not by construction: every blocksworld fold happens to
contain at least one 5-block problem. A fold drawing only 4-block problems would ground at
`n_props = 25`, giving that fold its own head width and its own vocabulary, and the grid would
then average precision and recall across two of them. That is §4.4's argument — *"folds stop being
comparable and the arm becomes sensitive to fold composition"* — applied to the vocabulary instead
of the pixel statistic. Same fix, same cache key. It also makes a `backfill_baseline` re-run of a
single cell vocabulary-identical to the row it replaces, which fold-level grounding does not
guarantee.

**Gate, required whichever option is chosen.** Solve one 4-block blocksworld trace under both
groundings and assert the recovered action model is identical. If it is not, the phantom-variable
analysis above is wrong, and the choice stops being a fidelity question and becomes a measurement
one.

### 4.3 Other points that will bite

1. **`inits` and `goals` are given, hard.** Maps cleanly onto our GT init (problem `:init`) and
   GT final state — the same two anchors our current arm already uses. No new data requirement.
2. **`state_traces` is not supervision-free, and it is not logging-only.** Its **endpoints are
   the hard anchors**: `network.py:262-263` slices `inits = state_traces[:, 0, :]` and
   `goals = state_traces[:, -1, :]`, and `model.py:256` re-reads `goals = targets[0][:, -1, :]`
   inside the loss. Only the **interior** rows (`[:, 1:-1]`) are pure logging, consumed by
   `state_accuracy`. **DECIDED: endpoints are GT (as above); interior rows are zeros with a mask,
   or 0.5 if the harness cannot mask — never our VLM states.** Feeding our degraded symbolic
   channel into the competitor's input is the mirror image of the resolution asymmetry in
   `docs/algorithm_comparison_analysis.md` §5, and in our favour. Consequence: `state_acc` is
   measured against filler and **must not be reported as accuracy**; log it as a training
   diagnostic or not at all.
3. **Image normalisation** must be corpus-level, not fold-level — see §4.4.

### 4.4 DECIDED: image normalisation is computed once over the whole `data_dir`

`normalize_traces` computes per-pixel mean/std over the dataset and stashes it in a global
`parameters` dict. Computing it *per fold* would be worse than it sounds: over ~25–90 images the
statistic is noisy, and the same problem would be normalised differently in fold 0 than in fold 3,
so folds stop being comparable and the arm becomes sensitive to fold composition. Computing it
once across all problems in the `data_dir` is fold-independent and closer to upstream's
corpus-level statistic. It is a statistic-level touch of held-out pixels — standard practice, and
it goes in the deviation register (§8) as a footnote.

**The upstream cache is unkeyed and must not be reused as-is.** `dl/main/normalization.py:8`:

```python
def normalize(x, save=True):
    if ("mean" in parameters) and save:
        mean = np.array(parameters["mean"][0]); std = np.array(parameters["std"][0])
```

One global slot named `"mean"`, no key of any kind, and it is never invalidated. Two properties
make that dangerous rather than merely sloppy:

* `mean = np.mean(x, axis=0)` over `[B*T, C, H, W]` produces a **per-pixel** array, so the cached
  statistic **carries the image shape**. Under `Resize(64)` hanoi is 3×64×180 and blocksworld is
  3×64×64 — a single process that touches both reuses a wrong-shaped array and either raises or
  broadcasts silently.
* `parameters["picsize"]` (set in `load_data`) has exactly the same problem.

**DECIDED: key the cache on `(domain, resize-form)` and persist it to disk** under the
`data_dir`, so a fold, a re-run and a `backfill_baseline` of the same cell all normalise
identically. Recompute only when the key is absent. An in-memory-only cache would make a
backfilled row silently incomparable to the row it replaces.

### 4.5 Resize and standardisation — the exact deltas

| | upstream 26 | ours today (24 arm) | the 26 arm should |
|---|---|---|---|
| scale | `/255` (`rosame_full.py:12`) | `ToTensor()` → [0,1] | same — equivalent |
| **resize** | **none.** Native `picsize` from the npy; only `ZeroPad2d((0,dW,0,dH))` from `autocrop_dimensions` so H,W are multiples of 32 | `Resize((64,64))` — **forced square**, aspect distorted (`rosame_i_runner.py:34`) | **per-domain configurable, default `Resize(64)`** (int, aspect-preserving, ICAPS-24 form) — see §4.6; the 24 arm changes to match (§4.6.1) |
| **standardise** | per-pixel mean/std over all frames of all traces (`normalize_traces` → `normalize`), cached in the `parameters` global | **none** | **add it**, computed once over the whole `data_dir` (§4.4) |
| frame 0 | **dropped** (`img_traces[:, 1:]`) | kept | drop |
| frame N (last) | n/a — upstream's raw images are already one shorter than ours | kept | **also drop** — see §4.1; both endpoints enter symbolically, so the 26 arm sees N−1 images to the 24 arm's N+1 |

### 4.6 DECIDED: resize is a per-domain configurable, default `Resize(64)` (ICAPS-24 form)

Resolution is not a detail to settle once — it is a variable worth *experimenting* on, because
`docs/algorithm_comparison_analysis.md` §5 shows its effect is domain-specific (geometry survives
at 64 everywhere; only depot's text-encoded identity dies, 40 px → 4.3 px). So it becomes a knob,
not a constant.

**Two layers, deliberately belt-and-braces:**

1. **Code fallback = `64` as an `int`** — i.e. `transforms.Resize(64)`, shorter edge, aspect
   **preserved**: the ICAPS-24-faithful form. Anything that does not ask for a resolution gets
   that, and nothing old crashes.
2. **The value is nevertheless written out explicitly for every domain**, so an operator can see
   what resolution a run used without inferring it from a default.

**Status: built, in the `_HYPERPARAMS` style.** `_RESIZE` in
`benchmark/baselines/rosame_i_runner.py` lists all five domains at `64` explicitly beside
`_HYPERPARAMS`, keyed on the same `_infer_domain_name` bench key; `RosameI_Runner`/`MilpRosameI`
take a `resize` argument; `backfill_baseline` grew `--resize N|H,W|native` as a per-run override.
That satisfies layer 1, layer 2 and the three requirements below without new plumbing through
`BaselineRunner`, which is what this section asked for.

**Not built: a `config.yaml` surface.** The sketch below is the shape it would take if the value
ever needs to vary per *experiment* rather than per *domain* or per *run*. It needs `resize`
threaded from config through `BaselineRunner`, which the paragraph above deliberately avoids, so
it stays a sketch until something actually needs it. Today the two live surfaces are the `_RESIZE`
table (the per-domain default) and `--resize` (the per-run override):

That second point is not redundancy for its own sake. Three times in this project a default that
lived only in code has produced a wrong or unexplainable result — `backfill_baseline`'s
`--learn-timeout 300` silently halving the budget, `run_params`' `normalized` flag, and
`mip_time_limit: 60` which never actually binds (measured median solve: 0.318 s). Resolution is
now an experimental variable; it belongs where the experiment is configured.

```python
# benchmark/baselines/rosame_i_runner.py — built, beside _HYPERPARAMS
_RESIZE_DEFAULT: int = 64
_RESIZE: Dict[str, object] = {
    "blocksworld": 64,   # int   -> Resize(N), shorter edge, aspect preserved (ICAPS-24 form)
    "hanoi": 64,         # [H,W] -> Resize((H,W)), forced (torchvision order is (h, w))
    "npuzzle": 64,       # None  -> no resize at all, native (ICAPS-26 form)
    "gripper": 64,
    "depot": 64,         # candidate for 224 — see analysis §5.1
}
```

```bash
# per-run override, suffixes the row name automatically
python -m benchmark.backfill_baseline ... --resize 64,64     # the old forced-square form
python -m benchmark.backfill_baseline ... --resize native    # no resize
```

```yaml
# NOT BUILT — the per-experiment surface, if it is ever needed
domains:
  hanoi:
    image_preprocessing: {resize: 64}
```

Resolution rules:

* `int` → aspect-preserving, the ICAPS-24 form;
* `[H, W]` → forced, the form we use today. **Note the order**: `transforms.Resize((a, b))` is
  (height, width), not (width, height) — writing `[85, 64]` for "85 wide, 64 tall" silently gives
  the transpose;
* `null` → native, the ICAPS-26 form (and the only one that exercises `ZeroPad2d` /
  `autocrop_dimensions` for real);
* absent → **`64` (int, aspect-preserving)**, the code fallback.

**Applies to every pixel arm** — `rosame_i`, `rosame_i_milp`, and the new 26 arm — read from the
same place, so an experiment at a given resolution is comparable across all of them. Keyed on the
bench name the runner already infers via `_infer_domain_name`, exactly as `_HYPERPARAMS` is, so no
new plumbing through `BaselineRunner` is required.

Three requirements that follow, each of which we have already been bitten by once:

1. **Record it.** ~~The effective per-domain preprocessing goes into `run_params.json`~~, and the
   resolution into each row's `algorithm_specific`. A row must know what produced it.

   **Amended: `algorithm_specific` only; the `run_params.json` half is withdrawn.** It was the
   wrong home for this. `run_params.json` is written by `experiment_runner` and describes a *run*,
   but resize is a property of a single *arm* within it, and the backfill path — which is how
   these rows are actually produced — never writes `run_params.json` at all. Requiring it there
   would have been satisfiable only on one of the two paths, which is worse than not requiring it.
   `algorithm_specific.resize` is per-row, is written on both paths, and proved sufficient in
   practice: it is what let the §9.6 census separate pre- and post-change rows post hoc, with
   `<ABSENT>` as an unambiguous marker for the old ones.
2. **Label it. Built** (`BaselineRunner.row_name`), **with the rule tightened.** The suffix keys
   on the *effective* resize — `_resize_tag(effective) != _resize_tag(_RESIZE_DEFAULT)` — not on
   whether a `--resize` override was passed. The first implementation exempted the per-domain
   `_RESIZE` table on the grounds that it *is* that domain's default, but the table is precisely
   where a divergence is expected to live (§5.1's depot-at-224), so an off-default entry produced
   a bare `ROSAME-I` row at a resolution no other `ROSAME-I` row was trained at. Two resolutions
   averaged into one row name would be two algorithms wearing one label — the exact failure the
   `__gt=none` suffix rule exists to prevent.

   `row_name` takes the **domain path**, not a bench string: `_infer_domain_name` derives the key
   from the parsed PDDL while callers hold a different string entirely (the results-dir name, or
   `--domain`), so accepting the caller's would let a row be labelled at one resolution and
   trained at another.
3. **Budget for it.** `null`/large values multiply the per-epoch cost roughly with pixel count
   (depot native is 88× the pixels of 64). The §1.2 pre-flight check must therefore include the
   image cost, not just the CP-SAT solve count, or a `resize: null` depot run at 5000 epochs will
   silently promise something it cannot finish.

#### 4.6.1 ACTION ITEM: fix `_IMAGE_TF` in the existing 24 arm too

ICAPS-24 (`main/train.py:211,226,237`) uses `transforms.Resize(64)` — int → shorter edge, aspect
**preserved**. Our `_IMAGE_TF` (`benchmark/algorithm_adapters/rosame_i_runner.py:34`) uses
`transforms.Resize((64, 64))` — tuple → forced, aspect **distorted**. These are different
operations on the same nominal number, and the difference is not cosmetic:

| domain | native | aspect | `Resize(64)` (upstream) | `Resize((64,64))` (ours today) | distortion |
|---|---|---|---|---|---|
| blocksworld | 480×480 | 1.00 | 64×64 | 64×64 | none |
| npuzzle | 187×187 | 1.00 | 64×64 | 64×64 | none |
| gripper | 800×600 | 1.33 | 85×64 | 64×64 | **1.33× horizontal squeeze** |
| depot | 800×600 | 1.33 | 85×64 | 64×64 | **1.33× horizontal squeeze** |
| hanoi | 630×224 | 2.81 | 180×64 | 64×64 | **2.81× horizontal squeeze** |

**Decision: change `_IMAGE_TF` to the int form, in the 24 arm as well as the new one**, so the
default is upstream-faithful everywhere and the two arms stay comparable.

**Why this may matter beyond faithfulness — TESTED, AND IT DOES NOT.** The hypothesis below was
the reason Phase 0 came first. It was run as a 4-domain, 30-paired-cell A/B (§9.6, analysis §5.3)
and **refuted**: hanoi, the 2.81× case it was built on, was 4/4 empty in 30/30 cells under *both*
transforms, with not one cell differing. The subsection is kept as written, because the reasoning
was sound and the prediction was sharp enough to be killed cleanly — which is the point of
recording it. Read what follows as the hypothesis, not as a finding.

The change to `_IMAGE_TF` stands regardless, on the faithfulness grounds above.

Shaked's hypothesis, recorded verbatim because it is
testable and, if right, is a partial answer to a question `docs/algorithm_comparison_analysis.md`
§3.1 leaves open:

> *"if we resized it differently in our impl than in the upstream's — that may lead to an
> explanation why in our experiments rosame-i got so low metrics compared to what they reported in
> their paper."*

The mechanism is plausible. Hanoi loses 2.81× of its horizontal extent, and hanoi's whole state
signal is *horizontal*: which peg a disc sits on is a left/right position, and disc identity is
width. Squeezing 630 px of horizontal layout into 64 px compresses exactly the axis that carries
the fluents, while the vertical axis — which carries only stacking order — is left alone. Hanoi is
also the domain where ROSAME-I is most completely collapsed (4/4 actions with empty effects,
effect recall 0.000). Gripper and depot take the same treatment at 1.33×; blocksworld and npuzzle
are already square and are therefore **unaffected**, which is itself the discriminating prediction
— blocksworld is also the domain with the *least* collapse (2/4).

**Two things went wrong in that last sentence, and they were the load-bearing ones.** First, the
figure: the 30-cell census run alongside the A/B measured blocksworld at **3/4 empty in 17 cells
and 4/4 in 13 — mean 3.43/4, and never 2/4 in any cell**, so the claimed correlation (least
distorted ⇒ least collapsed) was much weaker than stated before anything was tested. Second, and
worse, blocksworld is *square*: it takes the identical operation under both transforms and so
**could not have moved**. It was a null control, not the discriminating comparison this paragraph
presents it as. The A/B therefore had one real arm, hanoi, and hanoi was flat.

Two caveats, so this is not oversold:

* It is **not** a competing explanation to the optimisation collapse (analysis §3.1); it is a
  possible *aggravator*. A degenerate optimum reachable at any resolution stays reachable at
  85×64 — and the collapse also occurs in blocksworld and npuzzle, which take no distortion.
  **Measured: not an aggravator either** (§9.6). The hedge was not conservative enough.
* The prediction is directional only. Confirming it needs the re-run in §9 with both forms.
  **Done; it disconfirmed.**

**Cost of the change: all existing ROSAME-I and ROSAME-I_MILP rows become non-comparable** with
rows produced after it — moot for ROSAME-I (its 150 rows are already void, analysis §3.1) but
*not* moot for ROSAME-I_MILP, whose rows are currently usable. Land the `_IMAGE_TF` change and the
ROSAME-I re-run together, and re-run ROSAME-I_MILP in the same pass.

**Revised by the A/B result.** "Non-comparable" turns out to be a statement about *labels*, not
about *numbers*: with the transform measured inert, the existing ROSAME-I_MILP rows remain
substantively comparable. What is actually wrong with them is that they carry no recorded `resize`
under a bare `ROSAME-I_MILP` name, which under the new default reads as the aspect-preserving arm.
Re-run them with `--force` for label integrity — but it is bookkeeping, and it should yield to
§9's item 8 (the schedule fix) if compute is contended.

---

## 5. Per-domain assets — generate all five from OUR domains

`Convertor.__init__` (`convertor/convertor.py:44-52`) loads, per domain:

* `planning_structs/specs/<domain>/domain.json` — the domain spec
* `pddl/<domain>/domain.pddl` — the ground-truth model

**Both are mandatory, and the `gt_am is None` guard in `run_fixer` is unreachable.**
`parse_pddl_domain(...)` runs in `Convertor.__init__` with no guard, so a missing
`domain.pddl` raises before any guard is consulted. Nor is the permutation optional under option
B: `run_fixer` calls `model_permutation` unconditionally on the pseudo-label path against
`obs_m`, and only the *second* call — against `gt_am` — is the `mip_gt_dist` diagnostic. Ship
both assets for all five domains (§2.1).

Shipped: `blocksworld`, `gripper`, `hanoi`, `8-puzzle`, `logistics`. **None of them is reusable
except `gripper`.** The overlapping names encode *their* domain variants, not ours — and
`logistics` has no counterpart here at all, while our depot has none there:

| | their spec | ours |
|---|---|---|
| hanoi | one `object` type; `clear`, `on`, `smaller`; one action `move` | typed `peg`/`disc`; `clear-disc`, `clear-peg`, `on-disc`, `on-peg`, `smaller-disc`, `smaller-peg`; four `move_*` actions |
| blocksworld | `arm-empty`, `on-table`, `pickup`, `putdown` | `handempty`, `ontable`, `pick_up`, `put_down` |
| gripper | `at-robby`, `at`, `free`, `carry` / `move`, `pick`, `drop` | identical — the one that happens to match |
| npuzzle | shipped as `8-puzzle`; variant not verified | moot — generated either way |
| depot | **not shipped** | generated |

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

**Nor is there an augmentation gap against ICAPS-24.** `main/train.py:165-237` splits into two
families, and only one of them is ours:

| family | domains | network | transform |
|---|---|---|---|
| `grid_*` | blocks, gripper, logistics | `CVGrid` / `GridConv` over 28×28 MNIST-composed cells | `RearrangeColumn` / `RearrangeBalls` / `RearrangeItems` |
| **`synth_*`** | **blocks, hanoi, 8-puzzle** | **resnet18 + 512/256/n_props — what we port** | **`Resize(64)`, plus `RandomHorizontalFlip(0.5)` on blocks only** |

Our `_AUGMENT_DOMAINS = {"blocksworld"}` horizontal flip therefore matches `synth_blocks`
**exactly**, and hanoi and npuzzle correctly get none. For every domain where a published synth
counterpart exists, our augmentation is upstream's augmentation. The `Rearrange*` multipliers
belong to a different architecture on different inputs and are not a comparison point.

Note what the table also says: there is **no `synth_gripper` and no `synth_depot`**. Upstream's
gripper is a cell grid in both branches. Our gripper and depot rows are therefore a *new*
measurement of this architecture on rendered scenes, not a reproduction of a published one — which
is a statement about what our numbers can be compared to, not a suggestion that we should build
grid renders. We should not; all five of our domains are real images by design.

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

`*` = the four we do not match, each for a reason already given: `epoch` and `batch_size` in §1.1
and §1.2; **`MIP_to_DL` alone** loses `action` under option B, while **`DL_to_MIP` keeps it** —
that channel is how the observed actions reach the solver (§0); and `cp_type` is CP-SAT for want
of a Gurobi licence. Everything unstarred is pinned verbatim. Note λ and γ **match our existing
arms already** — no change there, and `lr: [1e-4]` / `DL_to_MIP: [[...]]` are single-element
*tuning-sweep* lists in `train_common.py`, so our config must unwrap them rather than copy the
brackets.

**Parameters the dict does not contain**, injected by the CLI layer we are skipping, so we must
supply them ourselves: `action_dim` (read by `adim()`, `dl/mixins/action.py:38`, and it sizes the
action head's output layer), `domain` and `data_loc` (which upstream uses to pick the augmentation
we do not port, §3), and `picsize` (set inside `load_data`). Anything `self.parameters[...]`
reaches for that is absent will `KeyError` at build time rather than fall back — which is the good
failure, but only if we enumerate them up front.

**Encoder shape notes**, verified so the adapter is not written on guesses: `_build_encoder`
(`dl/mixins/encoder_decoder.py`) returns `[nn.ZeroPad2d((0, dW, 0, dH)), resnet18]`, sets
`fc = nn.Linear(encoder_feature_shape[0], fdim())` where `encoder_feature_shape[0]` is layer4's
**channel** count (512 for resnet18, independent of input size), and derives the pad from
`autocrop_dimensions(input_shape)`. Two consequences: the head is genuinely resolution-agnostic,
so non-square inputs from `Resize(64)` need no architectural change; and `input_shape` is fixed
once from `train_data[0].shape[1:]` (`network.py:train`), so every trace in a run must share
`(C, H, W)` — another reason the padding in §6.1 has to be uniform per run.

**Reconstruction — DECIDED: vendor the decoder, keep `beta_reconst: 0`.** That is upstream's
released configuration exactly. Practical consequence worth knowing: with the weight at zero,
`loss_reconst` contributes no gradient, so `feature_decoder` *and* `feature_composer` are inert
— they cost memory and init time, nothing else. So the AAE's parameter count is **not** a
data-scale concern: the trainable path is encoder → symbol_net → sigmoid plus the action head and
schema MLPs, barely more than ICAPS-24. Keeping the modules
buys a one-parameter ablation (`beta_reconst: 1` → "does reconstruction help at our scale?") for
free. Ignore `GaussianOutput` in `dl/mixins/output.py`: `StateAE` installs `VanillaRenderer` and
`loss_reconst` is plain MSE, so it is unused.

### 6.1 DECIDED: the 26 arm uses `src/milp/encoder.py`, not the vendored solvers

`src/milp/vendor/constraint_opt/cp_sat.py:29` — and identically `mip_gurobi.py:18` — is

```python
self.max_t = traces.obs_t[0].step + 1
```

the **first** trace's length, applied to all `mip_traces` traces in the bundle. Upstream's corpora
are length-homogeneous so this never bites there; our folds are not, so it would either `KeyError`
or, worse, mis-encode a shorter trace against a longer horizon. Switching solvers is not an
escape — both vendored backends carry the same line.

Our `src/milp/encoder.py` already fixed this (per-trace `_steps(i)`, `obs_p.get(t, [])`), so:
**use it, and register the fork in §8.** The alternative — patching the vendored file — would
break the "verbatim from `95c733f`" guarantee `UPSTREAM.md` makes, and that guarantee is worth
more than the fidelity of this one line. Note this is also what makes the 0.318 s/solve figure in
§1.2 transferable: it was measured on *our* encoder, so it applies exactly when the 26 arm uses
the same one.

**The uniform-horizon assumption is not confined to the encoder — it runs through the extractor
too.** `translator.extract_sol_label` sizes its outputs from a single scalar:

```python
state_label  = torch.zeros(problem.max_t - 2, len(self.instance.propositions))   # must equal T
action_label = torch.zeros(problem.max_t - 1, dtype=torch.long)                  # must equal T+1
```

and `loss_pseudo_s` then does `binary_cross_entropy(outputs["z"][batch_id], state_label)` against
a `z` row of shape `[T_i, S]`. With ragged traces those disagree for every trace but the one that
set `max_t`. So swapping the encoder alone is not enough: **either our encoder exposes a padded,
uniform `max_t` and the padding is carried consistently into the pseudo-labels, or
`extract_sol_label` is adapted to a per-trace horizon.** Decide it once, here, rather than
discovering it as a shape error in Phase 5.

**DECIDED: pad + mask, applied end to end** — dataset (`TraceDataset.__getitem__` indexes
fixed-shape arrays and default-collate cannot batch differing `T`), encoder, and pseudo-labels.
Bucket-by-length would be cheaper but changes batch composition, and therefore changes which
traces the FIFO `TraceSelector` sees first, which decision 6 explicitly pins to upstream. Padding
keeps that pinning meaningful, and it is the option that keeps one `max_t` honest across all three
layers.

**Two silent-failure paths in the extractor, both worth an assertion:**

* **Action fallback.** `extract_sol_label` scans for the first action variable true at step `t`
  and, if none is, writes `chosen = 0` — *the first action in the instance* — with no signal.
  Under B the actions are hard-constrained, so this should be unreachable; assert that it is,
  because if it ever fires the pseudo-label is a fabricated action fed at full weight into
  `loss_pseudo_a`.
* **`extract_sol_model`'s precedence is `add > del > pre > none`.** An atom that is both a
  precondition and an add-effect is labelled `add`, dropping the precondition. Harmless while
  `forbid_redundant_adds` is on, since the CP solution cannot express that case — but the **depot
  ablation with `forbid_redundant_adds=False`** (`docs/algorithm_comparison_analysis.md` §3.3)
  makes it expressible, so the label goes lossy precisely in the experiment designed to test that
  constraint. Note it there before running it.

### 6.2 MILP cadence

The mechanism is the one our existing arms already have — `pre_mip_epochs 50`, `mip_interval 1`,
solves = `epochs − 50` (hence hanoi's observed 21 rounds at 70 epochs).

**But do not reuse `_solve_time_limit`.** Upstream passes the constant —
`convertor.run_fixer(trace_selector, self.parameters["mip_time_limit"])` (`network.py:303`) — and
has no per-solve scheduler at all. Ours (`milp_loop_i.py:145`) rations the remaining budget across
the remaining solves with a `_MIN_SOLVE_SECONDS = 5` floor, which at this arm's solve counts would
force `mip_interval` wider on every cell and trip the assertion below universally (§1.2). Pass the
constant `mip_time_limit`; the §1.2 pre-flight owns the budget.

With that, **assert `mip_interval_used == mip_interval`** and fail the row loudly if anything had
to intervene. That converts a silent deviation into a visible one — and with the rationing gone,
the assertion can actually hold.

Critically: **do not reuse `_HYPERPARAMS`** (the ICAPS-24 per-domain 70/100/300 table). ICAPS-26
is domain-**independent**: one epoch count for every domain — 5000 as the code default, 750 as the
grid value (§1.2). Mixing in the 24 arm's per-domain table would be exactly the kind of silent
assumption we're trying to eliminate.

---

## 7. Integration

* New runner class in `benchmark/algorithm_adapters/rosame_milp/` (the loop) + a
  `BaselineRunner` in `benchmark/baselines/`.
* **Two registry keys, not one:** `rosame_i_milp_26` → row `ROSAME-I_MILP_26`, and
  `rosame_i_26` → row `ROSAME-I_26` (the DL-only variant, `pre_mip_epoch ≥ epochs`). One key
  emitting two row names would break the same rule §4.6 requirement 2 exists to enforce — a row
  name must identify the algorithm that produced it.
* **Option A is out of scope for this build.** §0 keeps it reachable as a flag, but it needs
  `model_permutation` live against a GT model, both MILP channels carrying `action`, and
  `loss_pseudo_a` on its original self-distilled target — a different validation surface, not a
  toggle. §10 builds B only; A is a follow-up with its own phase.
* Work subdir per row label, same rule as `milp_work_subdir` — so it cannot collide with the
  existing arm's models.
* It is a *baseline*, so it reaches cells through `backfill_baseline`, whose `--learn-timeout`
  default is **300** — half the 600 s this arm is budgeted for (§1.2). Every invocation must pass
  `--learn-timeout 600` explicitly, and the runner should record the value it actually received in
  `run_params` so a silently-halved cell is visible in the row rather than only in the shell
  history.
* Skip cleanly on simulation-mode cells (no images), same as `rosame_i`.

---

## 8. Deviation register (ship this in the thesis)

Every item here is a place we knowingly differ. Keeping the list short and explicit is the
whole point of the exercise.

**Algorithmic**

1. Actions observed, not predicted (option B) — *the* deviation; state it first. It carries three
   sub-deviations (§0): `loss_pseudo_a` retargeted from the self-distilled `pseudo_label()` argmax
   to the observed action; `MIP_to_DL` drops `action` while `DL_to_MIP` keeps it; the action
   channel is a hard constraint rather than an objective term.
2. **Training budget.** `epoch: 5000` is the code default; the grid runs a calibrated value inside
   a 600 s cell budget, with one 5000-epoch control cell per domain outside it (§1.2). Report the
   epoch count actually used per run, never the default. "Epoch" is also not the same unit on both
   sides (§1.1) — upstream's is a full pass at batch 128 over a large corpus.
3. CP-SAT instead of Gurobi, **and `src/milp/encoder.py` instead of the vendored `cp_sat.py`**
   (§6.1) — the vendored encoder takes the first trace's length for the whole bundle, which our
   ragged folds violate.
4. `mip_time_limit` passed as a constant, with no per-solve rationing (§6.2).
4a. `model_permutation` bypassed on the pseudo-label path; identity mappings passed instead
    (§0.1). Retained for the `mip_gt_dist` diagnostic only.
4b. Pseudo-labels padded and masked to a uniform horizon, since `extract_sol_label` sizes its
    outputs from a single `max_t` (§6.1).
5. Per-domain image augmentation (`BlocksworldPilePermute`, `RoomBallPermute`, `ItemPermute`)
   **not ported** — each hard-asserts a render layout ours does not have, so they are inapplicable
   rather than switched off (§3). Upstream therefore trains its `blocks` / `gripper` / `logistics`
   runs with a label-preserving data multiplier we do not have. State it when reporting the 26
   arm against upstream-26 numbers; it is not a factor against our own 24 arm (§5.1).
6. `n_seeds` runs per cell where upstream runs once (§1.3).

**Data**

7. Image normalisation computed over the whole `data_dir` rather than a held-out-clean split,
   and cached per `(domain, resize-form)` on disk rather than in one unkeyed global (§4.4).
8. Resize defaults to `Resize(64)` (int, aspect-preserving) rather than upstream-26's native size
   (§4.6) — configurable per domain, and `resize: null` reproduces upstream exactly. Report the
   value actually used per run.
9. **Both endpoint frames are dropped**, not just frame 0 (§4.1). The 26 arm sees N−1 images
   where our 24 arm sees N+1, with both anchors entering symbolically.
10. `state_traces` **endpoints are hard anchors** (GT init and GT goal, as upstream); only the
    **interior** rows are filler, and they are zeros-with-mask rather than our VLM states.
    Consequence: `state_acc` is measured against filler and is not reported as accuracy (§4.3).
11. Proposition space follows upstream's `Instance` (no repeated args), so the 26 arm's `n_props`
    differs from our other arms' — report it alongside them (§4.2).
12. All five domain specs and GT `domain.pddl` assets generated from our `src/domains/*.pddl`,
    none reused from upstream (§5).

**Operational (footnotes, no algorithmic content)**

13. DataLoader workers sized from the fold rather than upstream's `num_workers=64` (§1.1).
14. Device pinned to cuda-else-cpu; MPS excluded (§1.3).

Everything *not* on this list is pinned verbatim — see the ✅ rows in §1.

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

**Tests 1–3 are Phase 1 work, not Phase 3.** Three of the five errors an external review found in
the first draft of this plan — the T+1/T+2 alignment, `loss_pseudo_a` surviving option B, and
`state_traces` endpoints being anchors rather than logging — are all things a shape or loss
parity test catches mechanically on day one. Writing them against the vendored code *before* the
adapter exists is cheaper than discovering the same facts by reading.

In order, each cheap and each falsifiable:

1. **Shape/parity test** — one trace through their `Net.forward` and through our adapter; assert
   `z` is `[B,T,S]`, `a_logit` is `[B,T+1,adim]`, `z_suc_aae` and `p_applicable` are
   `[B,T+1,adim,S]`, `state_traces` is `[B,T+2,S]`, and `z ∈ [0,1]`. Assert the T = N−1
   relationship of §4.1 explicitly, and that a 2-image trace is rejected rather than yielding
   T = 0.
2. **Loss-parity test** — construct a tiny synthetic trace, compute `loss_dict` under the
   vendored `dl/model.py` and under our harness; assert equality to float tolerance. This is the
   test that actually protects the port. Cover both `loss_pseudo_a` regimes: no MILP labels
   (`weight_mask_a` all ones) and some MILP labels (decayed weights on those rows only).
3. **MILP-parity test** — one `run_fixer` call, assert the pseudo-labels it produces match a
   direct invocation of the vendored translator. Also assert: the identity mappings of §0.1 are
   what reach `extract_sol_label` / `extract_sol_model` (**not** that `model_permutation` returns
   the identity — it would not, §0.1); `trans_full_state`'s zip is index-aligned with the DL
   symbol vector (§4.2); `state_label`/`action_label` shapes match `z`/`a_logit` for a **ragged**
   bundle, not just a uniform one (§6.1); and the `chosen = 0` action fallback never fires (§6.1).
4. **Degenerate-model guard** — the same assertion I'd want on ROSAME-I: reject a learned model
   with zero add-and-delete effects across all schemas, loudly. We would have caught the
   empty-effects collapse on day one with this.
5. **Sanity run** — one blocksworld fold, DL-only mode, `pre_mip_epoch ≥ epochs`. If effects are
   non-empty here and were empty for ICAPS-24, that alone is a publishable result.
6. ~~**Resize A/B on the 24 arm**~~ — **DONE. The hypothesis is refuted**; see
   `algorithm_comparison_analysis.md` §5.3 for the full result. Run on four domains, 30 paired
   cells each. Hanoi — the 2.81× case, the only real test — was **4/4 empty in 30/30 cells under
   both transforms, zero cells different**. Gripper moved marginally the wrong way; depot's 5-vs-2
   split is a sign test at p ≈ 0.45; blocksworld could not move at all, being square, so its half
   of the stated prediction was vacuous and the A/B was one-armed by construction.

   **The disjunction this item wrote in advance now resolves to its second branch**: the
   distortion is a faithfulness fix only, and the collapse belongs to data volume, schedule and
   epoch semantics (analysis §3.1, items 1–3) — **all three, equally**. Item 8 below is the live
   experiment that follows; item 7 remains the separate 26-arm budget question it always was.
7. **Budget control** — one cell per domain at `epochs: 5000` outside the timeout (§1.2), so
   "the arm underperforms at 750 epochs" and "the arm is undertrained at 750 epochs" stay
   distinguishable. **This is a 26-arm item and stays one.** `5000` earns its place here only
   because it is the value §1.2's pre-flight calibrates *away from*; it is the 26 code default
   (§1.1). It is not a number the 24 arm has any use for — that arm runs the ICAPS-24 paper's own
   per-domain counts, 70 / 100 / 300 (`benchmark/baselines/rosame_i_runner.py:37-41`), and 5000
   against those is off-paper on both papers at once.

8. **Schedule fix on the 24 arm — DONE IN CODE, and it was a deletion, not an ablation.** This
   item was scoped as "flip `train_per_trajectory` to `False` and compare". The premise did not
   survive checking where the flag came from: it has **no counterpart in upstream ICAPS-24**.
   `main`/`train.py` builds `DataLoader(trainset, args.batch_size, shuffle=True)` and takes one
   reshuffled pass over the pooled corpus per epoch — there is no per-trajectory loop to select
   between. The flag was a local invention, introduced on the image arm by
   `docs/rosame-i-implementation-plan.md` decision #1 (now marked REVERSED) and documented there
   at the time as the "safe default" that mirrors the *symbolic* baseline's schedule.

   So `learn_per_trajectory` and the dispatcher argument were **removed** from
   `benchmark/algorithm_adapters/rosame_i_runner.py`. There is one loop, `learn_pooled`, and no
   knob. Suite: 412 passed, unchanged by the removal.

   **The symbolic arm keeps its per-trajectory loop.** There, `PORosame_Runner.learn_per_trajectory`
   reproduces AMLGym's vendored `learn_rosame` and is pinned to it by
   `test_po_rosame_runner.test_local_loop_matches_vendored` — i.e. on that arm per-trajectory *is*
   the upstream schedule. The two arms have different upstreams; deleting globally would have
   broken fidelity on the symbolic one. The `--train-per-trajectory` CLI flags therefore stay, and
   their help text now says they do not reach ROSAME-I.

   **What this costs.** Every ROSAME-I and ROSAME-I_MILP row on disk (150 of them) was trained
   under the removed schedule, so they are stale on substance, not just on labels. The `--force`
   re-run of both pixel arms (analysis §7 item 1) stops being optional bookkeeping and becomes a
   prerequisite for reading anything off the 24 arm, and for Phase 4's 24-vs-26 comparison. Run it
   at the paper's own per-domain epochs, 70 / 100 / 300.

   **What this does not settle.** Analysis §3.1's other two suspects — data volume and epoch
   semantics — are untouched. Removing the deviation tells us what the faithful arm scores; it
   does not attribute the collapse. New rows are distinguishable by `"schedule": "pooled"` in
   `algorithm_specific`; old ones carry `"train_per_trajectory": true`.

   **The §4.6 labelling rule still binds `epochs`.** It does not reach `row_name` — only the
   resize suffix does — so an epoch sweep would still average two budgets under one `ROSAME-I`
   label. Label first, run second. And the cheapest prior check is unchanged: the adapter returns
   only `_total_loss` at the end, never a curve, so one cell with per-epoch loss logged settles
   whether "undertrained" is on the table at all.

---

## 10. Order of work

| Phase | Deliverable | Risk |
|---|---|---|
| **0** | ~~`_IMAGE_TF` → `Resize(64)` in the 24 arm (§4.6.1) + the resize A/B (§9.6)~~ — **CLOSED.** Transform shipped and made per-domain configurable with row-name suffixing; A/B run on 4 domains × 30 paired cells; **hypothesis refuted** (§9.6, analysis §5.3). The ROSAME-I baseline the 26 arm is measured against is now upstream-faithful *and* known unmoved by the change | low — done |
| 1 | Vendor `dl/` + `convertor/` + `util/model_perm.py`; **generate all five specs and `pddl/<domain>/domain.pddl` assets (§5, §2.1)**; import-only smoke test; **write parity tests §9.1–9.3 against the vendored code, before the adapter exists** | low |
| **1½** | **PIN §4.2a — one grounding or per-problem groundings. A decision, not a deliverable, and it gates Phases 2–6.** Our problems differ in object count where upstream's do not, so upstream's single `Instance` is a choice for us rather than a default. It sizes the DL head, so it cannot be deferred to the MILP phase | **blocking** |
| 2 | Data adapter (fold → their contract, §4.1–4.4) — must pass §9.1 | **medium — most of the work** |
| 3 | Harness replacement (§3) — must pass §9.2 | medium |
| 4 | DL-only arm (`ROSAME-I_26`), one blocksworld fold | low |
| 5 | Turn the MILP on (`ROSAME-I_MILP_26`) with `src/milp/encoder.py` (§6.1), one fold, verify `mip_gt_dist` logs — must pass §9.3 | medium |
| 6 | Full grid across 5 domains via backfill | compute only |
| 7 | *(follow-up, out of scope here)* option A — predicted actions (§0, §7) | high |

Phase 4 is a real milestone: it gives the old-vs-new DL comparison *before* any MILP work, and
it's where the empty-effects question gets answered — against a 24 baseline that Phase 0 has
already made upstream-faithful.

**But Phase 4 is not an architecture comparison unless you say what else moved.** A 24→26 delta
has at least eight candidate causes, and only the first is the one people will assume:

| | ICAPS-24 arm | ICAPS-26 arm |
|---|---|---|
| state head | raw logits, no sigmoid | `z = sigmoid(...)` ∈ [0,1] (`model.py:109`) |
| loss reduction | `reduction='sum'` | normalised by `B(T+1)` (`model.py:260-261`) |
| augmentation | horizontal flip on blocksworld (`_AUGMENT_DOMAINS`) | disabled (§1.3) |
| images per trace | N + 1 | **N − 1** (§4.1) |
| proposition space | `RepeatedArgsInstance` | upstream `Instance`, no repeated args (§4.2) |
| grounding scope | one `Instance` **per problem**; surplus union columns dropped at the converter | **undecided — §4.2a.** One shared `Instance` if upstream is followed, which makes absent-object propositions live CP variables |
| epoch budget | per-domain 70/100/300 | one calibrated value (§1.2) |
| batch size | 1 — pooled and reshuffled per epoch, but one trace per optimizer step (§9 item 8) | 128, per `train_common.py` (`vendor/UPSTREAM.md`) |

The last row is a narrowed gap, not a closed one. §9 item 8 removed the *ordering* deviation on
the 24 arm — it now reshuffles the pooled corpus each epoch, as upstream does — but left the batch
dimension at 1, which is the residual noted in `docs/rosame-i-implementation-plan.md` decision #1.
Hold it fixed or name it; do not let a batch-size effect be reported as an architecture effect.

Report the delta with that table attached, or hold the movable ones fixed in a dedicated ablation.
Presenting it as "old architecture vs new architecture" would be the same error as the resize
confound in `docs/algorithm_comparison_analysis.md` §5 — a real effect attributed to the wrong
cause.

---

## 11. Decisions taken

Every decision below was checked against the clone at `95c733f` (branch `ROSAME+MILP`).

**Scope**

1. **Option B is the default** (§0): override `a` with a one-hot of the observed action. Its three
   real consequences — `loss_pseudo_a` retargeted rather than deleted; `DL_to_MIP` **keeps**
   `action` while `MIP_to_DL` drops it; and `model_permutation` **bypassed** in the pseudo-label
   path in favour of explicit identity mappings (§0.1 — it is an agreement-maximising search that
   never compares names, so it can return a non-identity relabelling and silently corrupt every
   pseudo-label) — are decided in §0, not inherited. **Option A is a follow-up phase, not a flag**
   (§7): it needs the permutation search live, and a different validation surface.
2. **Two registry keys**, `rosame_i_milp_26` and `rosame_i_26` (§7). A row name must identify the
   algorithm that produced it.

**Budget**

3. **Wall clock: `learning_timeout_seconds: 600` per cell** — the same budget as every other arm,
   so the comparison stays like-for-like. Grid `epochs` is a **conservative config value the
   pre-flight is allowed to lower**, not a derived constant, plus **one unbudgeted 5000-epoch
   control cell per domain** so "underperforms" and "undertrained" stay distinguishable (§1.2).
4. **Pre-flight budget check** (§1.2), projecting from a **calibrated per-solve estimate, not
   `mip_time_limit`** — the 60 s cap never binds — and **multiplied by `n_seeds`** before the
   refuse-to-start comparison (§1.3).
5. **`mip_time_limit` is passed as a constant; `_solve_time_limit` is not reused** (§6.2). It is
   our invention, not upstream's, and its 5 s floor would trip decision 12's assertion on every
   cell. Removing it is both the faithful choice and the one that resolves the contradiction.

**Port**

6. **`epoch: 5000` verbatim as the code default; `batch_size` and the FIFO `TraceSelector` pinned
   to upstream** (§1.1). `min(batch_size, N)` is upstream's own line, not our adaptation.
7. **Vendor `dl/network.py` and subclass it**, overriding only `_run_training`; vendor
   `main/normalization.py`; stub `util/tuning.parameters`; skip the CLI layer (§2.1).
   **`convertor/convertor.py` needs asset placement, not vendoring alone** — two unguarded loads
   in `__init__` mean the spec JSON *and* a GT `domain.pddl` must exist for all five domains.
8. **Reconstruction: vendor the decoder, `beta_reconst: 0`** (§6) — upstream's own setting, and
   it keeps a free ablation.
9. **The 26 arm uses `src/milp/encoder.py`, not the vendored `cp_sat.py`** (§6.1): the vendored
   encoder takes the first trace's length for the whole bundle and our folds are ragged. Ragged
   batching is handled by **pad + mask**, not bucketing, so the pinned FIFO selector stays
   meaningful.
10. **All five domain specs and GT `domain.pddl` assets generated from our own PDDL** (§5). Depot
    is not a special case.
11. **Run-level settings pinned explicitly** (§1.3): `n_seeds` as for our other arms; device
    cuda-else-cpu with **MPS excluded**; DataLoader workers sized from the fold. Upstream's
    per-domain image augmentations are **not ported** — they live in the method we replace and
    assert layouts our renders do not have (§3).
12. **MILP cadence: assert `mip_interval_used == mip_interval`** so nothing can silently rewrite
    it (§6.2).

**Data**

13. **Alignment: drop the last frame as well as frame 0**, giving T = N−1 (§4.1). Upstream's raw
    images and actions are the same length with states one longer; ours has one image too many.
    Both endpoint images therefore go unused and enter symbolically — an information difference
    from the 24 arm that must be stated in the Phase-4 comparison.
14. **`state_traces` endpoints are hard anchors, not logging** (§4.3): GT init and GT goal, as
    upstream. Interior rows are **zeros with a mask, never our VLM states**, and `state_acc` is
    consequently not reported as accuracy.
15. **Proposition space follows upstream's `Instance`** — no repeated args (§4.2) — so the 26
    arm's `n_props` differs from our other arms'. Report it alongside them, and test that
    `trans_full_state`'s zip is index-aligned with the DL symbol vector.
16. **Image normalisation: once over the whole `data_dir`, cached per `(domain, resize-form)` on
    disk** (§4.4). Upstream's cache is a single unkeyed global holding a shape-carrying per-pixel
    array; reusing it across two domains is a silent corruption.
17. **Resize: per-domain configurable, default `Resize(64)`** — int, aspect-**preserving**, the
    ICAPS-24 form (§4.6), recorded in each row's `algorithm_specific` (the `run_params` half is
    withdrawn, §4.6 req. 1) and suffixed into the row name whenever the **effective** value is
    off-default (not merely when an override was passed, §4.6 req. 2). **The existing 24 arm's
    `_IMAGE_TF` changes to match** (§4.6.1): today's forced `Resize((64, 64))` squeezes hanoi
    2.81× and gripper/depot 1.33× along the horizontal axis — on hanoi, the axis the fluents live
    on. Held as a candidate *aggravator* of the empty-effects collapse, not a competing
    explanation. Consequence: ROSAME-I_MILP's rows must be re-run alongside ROSAME-I's.

    **SHIPPED AND TESTED. The aggravator claim is refuted** (§9.6, analysis §5.3): 4 domains ×
    30 paired cells, and hanoi — the case the claim was built on — did not move in a single cell.
    The default stands on faithfulness alone, and its measured inertness is what makes it a clean
    de-confounder for the Phase-4 comparison rather than a change that moves the baseline. The
    re-run consequence survives, downgraded from comparability to label hygiene (§4.6.1).

**Process**

18. **Parity tests §9.1–9.3 are Phase 1 work**, written against the vendored code before the
    adapter exists (§9, §10). Three of the five errors an external review found in the first draft
    of this plan would have been caught mechanically by them.

### Still open

**19. BLOCKING — grounding scope: one `Instance` for the run, or one per problem? (§4.2a)**

Upstream builds a single `Instance` from the ROSAME domain's object universe and shares it across
every trace (`convertor/convertor.py:48-49`); our 24 arm builds one per problem and drops the
surplus columns at the converter (`rosame_i_milp_runner.py:155`, `converter.py:473-486`). Upstream
never had to choose because its corpora are object-homogeneous. Ours are not — blocksworld
problems carry 4 or 5 blocks — so following upstream here is a **decision to make**, not a default
to inherit, and the two options give different `n_props`, different DL head widths, and different
CP variable sets on any problem smaller than the union.

**Implementation of `ROSAME-I_26` and `ROSAME-I_MILP_26` cannot start until this is pinned**
(Phase 1½, §10). It sizes the symbol head, so it is not a MILP-phase concern and deferring it means
rebuilding the adapter and discarding anything measured before it. Two sub-questions travel with
it, both in §4.2a: the union must be taken over the whole `data_dir` rather than the fold either
way, and the equivalence gate (same model under both groundings on a 4-block blocksworld trace)
must be run before the choice is treated as cosmetic.

*The remaining unknowns are empirical and are answered by the §9 validation runs, not by a
decision.*
