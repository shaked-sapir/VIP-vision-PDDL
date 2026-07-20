# Implementation Plan: ROSAME-I Baseline (Imaged Mode)

> **Audience**: an implementing agent (Claude Opus) with access to this repo
> (`VIP-vision-PDDL`) and the user's local ROSAME clone at
> `~/Documents/BGU/thesis/ROSAME` (branch `main` holds the ICAPS-24 ROSAME-I
> code; you only need `train.py` as the loss/training reference — do NOT use
> the `ROSAME+MILP` branch for this task).
>
> **Goal**: add ROSAME-I (ICAPS-24: CV state predictor + ROSAME trained
> jointly, actions observed) as a baseline algorithm that runs on our
> **imaged-mode** experiment data, so it can be compared against CDPS's VLM
> pipeline under the identical fold protocol. No MILP. Simulation mode is out
> of scope (the existing `rosame` baseline covers it).

---

## 0. Settled design decisions (do not revisit)

1. **Two training loops, selectable via a flag; default per-trajectory.**
   A `--train-per-trajectory / --no-train-per-trajectory` CLI flag (default
   **True** = per-trajectory) chooses the loop; the runner carries a
   `train_per_trajectory: bool = True` constructor arg (source of truth).
   The two loops differ ONLY in the training *schedule*; both require a shared
   object universe across trajectories, because ROSAME-I's CV head is a
   fixed-size proposition vector created at the first grounding (identical-
   grounding assert, §4). This is unlike plain ROSAME/CDPS, which need no CV
   head and tolerate varying object sets.
   - **Per-trajectory (default):** the incremental pattern of
     `PORosame_Runner` / `RosameBaselineRunner.learn` — for each trajectory:
     `add_problem` → ground → train that trajectory's frames for the full
     epoch budget, then move on; CV-model and schema parameters persist across
     trajectories (continual-learning style).
   - **Pooled:** ground once (first problem), precompute per-trajectory
     tensors against that single grounding, then for each epoch iterate over
     ALL traces (shuffled), one optimizer step per trace — closest to the
     ICAPS-24 `train.py` DataLoader.
   A trajectory whose actions don't map to the shared grounding (or whose
   image count ≠ T+1) is skipped with a loud warning in both modes. Both loops
   share the same per-trace loss primitive (§4). Per-trajectory is the safe
   default (mirrors the existing baseline's schedule); pooled is the more
   paper-faithful option.
2. **Anchor the FINAL state only**, exactly as in ICAPS-24 `train.py`
   (`gamma * MSE(domain_preds[:, -1], label[:, -1])`). Do NOT anchor the
   initial state. The final-state label comes from the GT trajectory JSON
   (see §3.3). s0-is-GT is a CDPS assumption, not a ROSAME-I one.
3. **Random-init ResNet-18 backbone**, exactly as in the paper
   (`torchvision.models.resnet18()` with no pretrained weights). The
   pretrained-weights variant is a recorded future-task only.
4. **Hyperparameters**: use the paper's per-domain defaults (see §5).
5. **Seed handling (the small-data variance mitigation)**: the runner trains
   `n_seeds` (default 3) independent models and returns the one with the
   **lowest final training loss** (a selection rule that never touches test
   data). All seeds' models and final losses are persisted (§4.4).

---

## 1. Repo context you need

- `benchmark/algorithm_adapters/po_rosame_runner.py` — `PORosame_Runner`
  extends AMLGym's `Rosame_Runner`
  (`AMLGym/amlgym/algorithms/rosame/experiment_runner/rosame_runner.py`,
  imported via the try/except fallback at the top of `po_rosame_runner.py`).
  Reuse from this stack: domain parsing, `prepare_rosame` /
  `add_problem` / `ground_new_trajectory` (per-problem grounding),
  `check_action` / `check_predicate` (map grounded action/prop strings to
  ROSAME indices), `rosame_to_pddl` (threshold → PDDL text).
- `benchmark/baselines/` — pluggable baseline runners.
  `base_runner.py::BaselineRunner` is the interface
  (`name`, `display_name`, `color`, `learn(domain_path,
  prepared_trajectories, work_dir, timeout_seconds) -> (model_str|None,
  extra_info_dict)`). `rosame_runner.py::RosameBaselineRunner` is the
  reference implementation; `__init__.py::BASELINE_REGISTRY` registers keys.
- `benchmark/experiment_running_helpers/run_fold.py::_run_baselines` — calls
  each runner on the fold's `prepared_trajectories`
  (tuples `(traj_path, masking_path, problem_pddl_path, gt_indices)`),
  saves the returned model to `<cell>/baseline_models/<name>/model.pddl`,
  evaluates via `result_builders.evaluate_and_build_result`, merges the row
  into `fold_result.json`.
- `benchmark/backfill_baseline.py` — retrofits baseline rows into existing
  experiment cells (reconstructs `prepared_trajectories` from each cell's
  frozen `original_observations/` + `--data-dir`). The new runner must work
  through this path too (that is how it will actually be run on the existing
  imaged experiments).
- Dashboard (`benchmark/evaluation/cfm/build_dashboard.py`) discovers
  algorithms dynamically from `fold_result.json` rows — a new baseline
  appears automatically; **zero dashboard work**.
- ROSAME-I reference: `~/Documents/BGU/thesis/ROSAME/train.py` (branch
  `main`) — the `run()` function is the loss spec; the `synth_*` branches
  are the relevant CV configuration (ResNet-18 + MLP head, `Resize(64)`).

## 2. Deliverables

1. `benchmark/algorithm_adapters/rosame_i_runner.py` — `RosameI_Runner`.
2. `benchmark/baselines/rosame_i_runner.py` — `RosameIBaselineRunner`,
   registered in `BASELINE_REGISTRY` under key **`rosame_i`**.
3. Image-resolution helper (see §3.2) — put it in the baseline module or a
   small shared util; do not duplicate per-domain path logic that already
   exists in config/handlers if reusable.
4. Config plumbing for per-domain hyperparameters (§5).
5. Two new rows in `future-tasks.md` (§7) — only if not already added.
6. Smoke tests (§6).

## 3. Data plumbing

### 3.1 Inputs per fold

`prepared_trajectories` gives, per trajectory: the degraded `.trajectory`
file (NOT used for training ROSAME-I — it trains from images), the problem
PDDL path, and implicitly the problem name (`problem_pddl_path.stem`).
ROSAME-I needs, per trajectory:

- the ordered list of **state images** (T+1 images for T actions),
- the ordered list of **observed grounded actions** (parse the `.trajectory`
  file's `(operator: ...)` lines, or reuse `TrajectoryParser` as
  `RosameBaselineRunner` does — actions only, ignore states),
- the **final GT state** for the γ-anchor (§3.3).

### 3.2 Image locations (verified)

Images sit next to the problem PDDL, so resolve them from
`problem_pddl_path.parent` (which is what both `run_fold` and `backfill`
already hand us):

- PDDLGym-rendered domains (blocksworld, hanoi, npuzzle):
  `<data_dir>/training/trajectories/<problem>/state_000000.png` —
  **6-digit**, zero-based (confirmed on disk; NOT 4-digit).
- External-image domains (gripper, depot): images ship in the source tree
  problem dir (`src/domains/<domain>/problems/<problem>/state_*.png`); the
  `problem_pddl_path.parent` for those cells points there too.

Glob `state_*.png` in that dir and **sort numerically, not
lexicographically** (`src/utils/containers.py::sort_objects_numerically`).
**Assert count == T+1** per trajectory; skip with a loud warning if not
(misalignment would silently corrupt training).

### 3.3 Final-state anchor

The GT final state comes from the problem's GT trajectory, preferring
`<data_dir>/gt_trajectories/<problem>/<problem>.trajectory`, falling back to
the training-dir `<problem>.trajectory`. Both are standard PDDL trajectories
(`(:init)` / `(operator:)` / `(:state)`), so **parse with
`TrajectoryParser(partial_domain, problem).parse_trajectory(path)`** (exactly
as `RosameBaselineRunner` does for actions) and take
`components[-1].next_state`. Build a binary vector over `rosame.propositions`
via `check_predicate` (positive fluents → 1, closed world → 0). **Guard
against `check_predicate` returning `None`** (collect matched propositions into
a set, ignore `None`). This is the only GT state ROSAME-I receives.
(The `_trajectory.json` is not needed — the `.trajectory` is authoritative and
its states are complete.)

### 3.4 Action indices

For each `(operator: (...))`, map to ROSAME's action index via
`self.check_action(action_str)` on the **current grounding** (after
`add_problem`/`ground_new_trajectory` for that problem) — same as
`PORosame_Runner.prepare_rosame_data` does.

## 4. `RosameI_Runner` (algorithm_adapters)

Subclass **`PORosame_Runner`** (NOT the raw AMLGym `Rosame_Runner`) so it
inherits the G1 `add_problem` fix — build-once / re-ground-after — for free
(see §8.5). Its `prepare_rosame_data` (0.5-masking encoder) is unused by
ROSAME-I and harmless. Same import-fallback pattern as `po_rosame_runner.py`.
Key members:

```python
class RosameI_Runner(PORosame_Runner):
    def __init__(self, domain_file, device=None, seed=8800,
                 lr_schema=1e-3, lr_cv=1e-3, batch_size=128):
        # device: cuda > mps > cpu autodetect (torch.backends.mps.is_available()).
        #   The vendored Domain_Model is pinned to CPU, so build()'s
        #   pre/add/dele come back on CPU; move them to `device` in the loss
        #   (see §4 loss port). Cross-device autograd carries gradients back to
        #   the CPU schema params; a mixed-device Adam param-group is fine.
        #   Allow device="cpu" to force everything onto CPU.
        # cv_model created lazily on first ground (needs |propositions|):
        #   torchvision.models.resnet18()  (random init — decision #3)
        #   .fc = Sequential(Linear(512,512), ReLU, Linear(512,256), ReLU,
        #                    Linear(256, n_props))
        # NO activation on the CV outputs — RAW logits, exactly as train.py's
        # synth/resnet path (decision G3: minimal deviation for a fair
        # comparison; only CVGrid, unused here, applies a sigmoid). preds are
        # used directly in the loss below.
```

**Important — CV head dimension vs per-problem grounding**: the head size is
`len(self.rosame.propositions)` of the current grounding. With the
per-trajectory loop (decision #1), if a later problem's grounding has a
different proposition count, the head no longer matches. Handle explicitly:
create the CV model on the first grounding; on each subsequent grounding
**assert** the proposition list is identical (names and order); if not,
abort the cell with a clear error ("ROSAME-I requires a shared object
universe across trajectories; got differing groundings"). All current imaged
datasets satisfy this; the assert is the guard for future data.

`learn_rosame_i(self, images, action_indices, final_state_vec, epochs,
gamma, lambda_)` — one trajectory. Faithful port of `train.py::run()`
adapted to a single trace (batch dimension 1 or frames-as-batch):

```
preds   = cv_model(images)                        # (T+1, n_props) — RAW logits, no activation
pre, add, dele = rosame.build(action_indices)    # (T, n_props) each; on CPU → .to(device)
domain_preds = preds[:-1] * (1 - dele) + (1 - preds[:-1]) * add
loss  = MSE(domain_preds[:-1], preds[1:-1])      # consistency on t=1..T-1
loss += gamma * MSE(domain_preds[-1], final_state_vec)   # γ-anchor (decision #2)
loss += MSE((1 - preds[:-1]) * pre, 0)           # applicability
loss += lambda_ * MSE(pre, 1)                    # precondition prior
```

This is **bit-for-bit equivalent to `train.py::run()`** for a single trace:
their trace is `trace_len` (= our T) pre-state images + a separate GT `label`
for the post-final state. Our (T+1)-th image (`preds[-1]`, the observed final
state) is therefore **intentionally unused as a target** — the GT anchor
replaces it (decision #2). Keep loading T+1 and slicing (the `count == T+1`
assert is a cheap integrity check). Mind the exact `train.py` indexing
(`preds[:, 1:]` / `domain_preds[:, :-1]`) and replicate it for the single
trace as above.

- **Device**: `build()` returns CPU tensors (vendored `Domain_Model` is
  CPU-pinned); `.to(device)` `pre/add/dele` before the algebra. `final_state_vec`
  and `images` on `device`. Everything else follows.
- **`check_action` may return `None`** (duplicate-arg filter / unmappable
  action). ROSAME-I needs a contiguous image↔action chain, so if ANY action in
  a trajectory maps to `None`, **skip the whole trajectory with a loud warning**
   — never drop a single transition mid-sequence.

Optimizer: Adam over `[{schema params, lr_schema}, {cv params, lr_cv}]`,
created ONCE (persists across trajectories). Images: load PNG → RGB tensor,
`Resize(64)`; transforms per domain follow the paper's synth settings
(`RandomHorizontalFlip(0.5)` for blocksworld only, plain resize otherwise).

### 4.3 Two training loops (decision #1)

Both loops share the same per-trace loss primitive (`_trajectory_loss`) and
the same single grounding (built on the first problem; identical-grounding
assert on every subsequent one). They differ only in schedule:

- `learn_per_trajectory(prepared, epochs, ...)` — for each prepared trace in
  order: run the full `epochs` budget of optimizer steps on that trace, then
  move to the next. CV + schema params persist (continual-learning style).
- `learn_pooled(prepared, epochs, ...)` — precompute every trace's tensors
  once, then for each epoch iterate over ALL traces in a shuffled order, one
  optimizer step per trace. Closest to `train.py`'s DataLoader.

A dispatcher `learn_full(prepared, train_per_trajectory, ...)` selects the
loop. Both return the final total training loss (used for seed selection).

### 4.4 Multi-seed wrapper (decision #5)

`learn_full` runs the selected loop `n_seeds` times (fresh CV + fresh schema
params + torch/python seeds set per run), records each run's final total
training loss, picks the model with the lowest, and returns
`(pddl_str, extra_info)` where `extra_info` includes
`{"seeds": {seed: final_loss}, "chosen_seed": s, "train_per_trajectory":
flag}`. Save every seed's PDDL to
`work_dir/baseline_models/ROSAME-I/seed_<s>/model.pddl` (the harness saves
the chosen one to `baseline_models/ROSAME-I/model.pddl` on top).

## 5. Hyperparameters (paper defaults, config-exposed)

Defaults dict in the baseline module, keyed by our domain names:

| domain | epochs | lambda_ | gamma |
|---|---|---|---|
| blocksworld | 100 | 0.2 | 10 |
| hanoi | 70 | 0.2 | 10 |
| npuzzle | 300 | 0.4 | 10 |
| gripper | 100 | 0.2 | 10 |
| depot | 100 | 0.2 | 10 |

(gripper/depot have no synth counterpart in the paper; use the generic
100/0.2/10 and note it.) `n_seeds` default 3. Allow overrides via the
runner's constructor; do not add new required config keys.

## 6. `RosameIBaselineRunner` (baselines)

- `name` → `"ROSAME-I"`, key `rosame_i`, `color` → `"#d55181"` (distinct
  from ROSAME's `#e8710a`).
- `__init__(self, train_per_trajectory: bool = True, n_seeds: int = 3,
  device=None)` — the loop flag is a constructor arg. Threading:
  `get_baselines` / `resolve_baselines` / `resolve_algorithms` gain a keyword
  `train_per_trajectory=True`, passed to a runner class **only if its
  `__init__` accepts it** (via `inspect.signature`, so `RosameBaselineRunner`
  is untouched). CLIs (`backfill_baseline.py`, `experiment_runner.py`) expose
  `--train-per-trajectory / --no-train-per-trajectory`
  (`argparse.BooleanOptionalAction`, default True) and forward the value.
  `run_fold.py` is unchanged (it receives already-instantiated runners).
- `learn(...)`: resolve images + actions + final states per trajectory
  (§3); **if any trajectory has no resolvable images, and none resolve at
  all — this is a simulated-mode cell: print a clear `[ROSAME-I] skipping:
  no images (simulation-mode cell?)` and return `(None, {})`** (the harness
  handles the null row).
- Respect `timeout_seconds` as a soft wall-clock budget across the whole
  multi-seed loop: check elapsed time between epochs; on expiry, finish the
  current seed and skip remaining seeds (record in extra_info).
- Register in `BASELINE_REGISTRY`; `benchmark/algorithms.py` needs no change
  (it derives choices from the registry).

## 7. future-tasks.md additions (table rows, follow existing format)

1. *ROSAME-I: ImageNet-pretrained backbone variant* — we deliberately ship
   random-init (paper-faithful); pretrained weights would likely help at our
   tiny data scale (3–8 trajectories); implement as a labeled variant and
   compare.
2. *ROSAME-I: sigmoid-activation CV head variant* — we ship raw-logit CV
   outputs (paper-faithful synth/resnet path) for a fair comparison; the
   paper's `CVGrid` path applies a sigmoid. A sigmoid variant could stabilise
   the [0,1] proposition semantics; implement as a labeled variant and compare.

## 8. Verification (acceptance criteria)

1. `python -c "from benchmark.baselines import get_baselines;
   get_baselines(['rosame_i'])"` succeeds (torch/torchvision required —
   run inside `venv11`).
2. Smoke test on ONE imaged cell (hanoi is single-problem, smallest):
   `python -m benchmark.backfill_baseline --experiment-dir
   benchmark/running_results/hanoi/TO=600__hanoi_generated_problem1__final-version
   --data-dir benchmark/data/hanoi/hanoi_generated_problem1__final-version_fixed
   --baselines rosame_i --cells fold0_numtrajs3 --dry-run` then without
   `--dry-run`. Verify: `fold_result.json` gains a `ROSAME-I` row with
   finite metrics; `baseline_models/ROSAME-I/model.pddl` + per-seed models
   exist; row merge is idempotent on re-run with `--force`.
3. The produced PDDL parses (`DomainParser`) and contains `:action` blocks
   for all observed actions.
4. Simulation-mode guard: running the same command against a sim cell
   (e.g. hanoi `simulation-checkup-after-gtfixed__mask=0.0__noise=0.0`)
   yields the skip message and a null-metrics row, no crash.
5. Dashboard: after backfilling ≥1 imaged experiment, rebuild
   (`python -m benchmark.evaluation.cfm.build_dashboard`) and confirm
   `ROSAME-I` appears as a checkbox/line in the image tab's vs-baselines
   mode without dashboard code changes.
6. Record one cell's wall-clock in the summary (expect ~30–90 s/cell on
   M1 Pro with MPS; minutes on CPU).

## 8.5 RESOLVED — G1: `add_problem` wiped schema parameters

The implementing agent correctly identified that the vendored
`Rosame_Runner.add_problem` rebuilds the `Domain_Model` (fresh schema
parameters) on every call, so per-trajectory learning kept only the last
trajectory. **This is now fixed** in
`benchmark/algorithm_adapters/po_rosame_runner.py`: `PORosame_Runner`
overrides `add_problem` to build the domain model once (first problem) and
only `ground_new_trajectory()` afterwards — restoring the original ROSAME
semantics (regrounding does not affect the learned lifted model, and is
valid across *different* object sets per their README).

Consequences for this task:
- `RosameI_Runner` should subclass or replicate the SAME override (build
  once, re-ground per problem). Call sequence per trajectory:
  `add_problem(problem)` (no-op rebuild after the first) →
  `ground_new_trajectory()` (harmless if doubled) → train on that
  trajectory. Schema AND CV parameters persist across trajectories.
- For the multi-seed loop (§4.4): construct a fresh `RosameI_Runner` per
  seed (do NOT try to reset weights in place).
- The identical-grounding **assert remains required for ROSAME-I only**
  (the CV head's output dimension is fixed at first grounding); plain
  ROSAME needs no such assert.
- The vendored AMLGym files must stay untouched; the fix lives in our
  subclass. (AMLGym's own `algorithms/ROSAME.py` adapter has the same
  defect upstream — out of scope here.)

## 8.6 RESOLVED — G2–G5 (design questions raised during review)

- **G2 (device)**: autodetect `cuda > mps > cpu`; CV model + images on the
  device, `.to(device)` the CPU-pinned `build()` outputs before the loss.
  Cross-device autograd carries gradients to the CPU schema params. `device=
  "cpu"` forces everything to CPU. (§4.)
- **G3 (CV activation)**: **RAW logits, no sigmoid** — replicate `train.py`'s
  synth/resnet path exactly. Chosen deliberately for a *fair* comparison
  (minimal deviation from the original ROSAME-I code); do not add any
  activation on `cv_model` outputs. (§4.)
- **G4 (final-state source)**: parse the GT `.trajectory` with
  `TrajectoryParser`, take `components[-1].next_state`; prefer
  `gt_trajectories/<problem>/`, fall back to the training-dir copy. Guard
  `check_predicate` `None`. (§3.3.)
- **G5 (frame count)**: the loss port is equivalent to `train.py`; the (T+1)-th
  image is intentionally unused as a target (GT anchor replaces it). Keep the
  `count == T+1` assert. (§4.)
- Also: `check_action` `None` ⇒ skip the whole trajectory (never drop a lone
  transition). (§4.)

## 9. Known pitfalls

- **MPS**: autodetect but allow `device="cpu"` override; some torch ops
  fall back silently — fine, but keep tensors float32.
- **Image sorting**: `state_2.png` vs `state_10.png` — numeric sort only.
- **Frame alignment**: T+1 images per T actions; assert, don't assume.
- **The degraded `.trajectory` states must NOT be used as training input**
  — ROSAME-I trains from raw images; only actions (+ GT final state) come
  from files. Using VLM-inferred states would make it our pipeline, not
  theirs.
- **Determinism**: set `torch.manual_seed`, `random.seed`, and
  (if CUDA) `torch.cuda.manual_seed_all` per seed run.
- **Do not modify** `PORosame_Runner`, `RosameBaselineRunner`, the result
  schema, or the dashboard. This task is purely additive.
- Branch: work on the branch the user specifies at runtime (default:
  current checked-out branch; do not create worktrees).
