# Plan: the imaged experiments, five domains, small and large corpora

Status: draft v3, 2026-10-24, after the user's answers to v1 (large image
corpora will be built; GPU nodes for the pixel arms; GT labels kept from the
generating process; frame 0 to be VLM-classified without leaking). Nothing has
been run or coded yet. Every number marked "measured" comes from result files
under `benchmark/running_results/<domain>/TO=600__*`, from the symbolic corpora
under `benchmark/data/*/large_corpus*`, or from the symbolic sweep's docs.

The goal is the imaged counterpart of the symbolic protocol: the same fold
protocol, the same arm set where an arm exists for images, the same row-label
conventions, run on the BGU cluster with one GPU type for the pixel arms.

---

## 0. What exists today, and the facts that shape the plan

### 0.1 Image corpora: small only

| domain | corpus (dashboard `image.small.experiment_dir` reads it) | problems | frames | frame size | images from |
|---|---|---|---|---|---|
| blocksworld | `blocks_predefined_problems1-10_final-version` | 10 | 134 | 480×480 | PDDLGym render |
| hanoi | `hanoi_generated_problem1__final-version_fixed` | 10 | 94 | 630×224 | PDDLGym render |
| npuzzle | `npuzzle_prob0_6-10steps_generate` | 10 | 166 | 187×187 | PDDLGym render |
| depot | `multi_problem_04-07-2026T01:32:08__model=gpt-5.2__steps=100` | 10 | 99 | 800×600 | `planning-visualizer` (sibling repo), planner traces |
| gripper | `multi_problem_04-07-2026T14:50:04__model=gpt-5.2__steps=100` | 10 | 130 | 800×600 | `planning-visualizer`, planner traces |

There is no large image corpus. Section 6 is the plan to build one.

### 0.2 The inference pipeline as it runs today (one problem at a time)

`benchmark/data_generator.py` builds a **new handler per problem**, and
`LLMVisualComponentsMixin.init_visual_components(frame 0 of that problem)`:

1. **Object detection makes no LLM call.** `LLMObjectDetector.__init__` has
   `inference_mode=False` everywhere it is constructed, and `_detect_once`
   then returns the GT object list of the problem's own `_trajectory.json`
   (`llm_object_detector.py:95`). The object universe is ground truth; the
   `gpt-5.2` "object detection" model in `config.yaml` is never invoked.
2. **Fluent classification is one-shot with the problem's own frame 0.**
   `fewshot_examples = [(frame_0.png, GT predicates of frame 0)]`
   (`llm_blocks_fluent_classifier.py:48`, same in depot/gripper/hanoi/npuzzle).
   The system prompt lists the (GT) objects; the user turn carries the example
   image, the example's GT facts as an assistant turn, then the target image.
3. **Frames 1..T are classified** with that example, one call per frame,
   up to 3 retries on an unparseable answer, `UNCERTAIN` for any grounded
   predicate the model did not mention. Concurrency is per window
   (`ConcurrentFrameClassifier`, default 12 workers; `--inference-workers`).
4. **Frame 0 is never classified.** State 0 of the written trajectory is the
   example's GT facts (`image_trajectory_handler.py:133,152-155`). So every
   image cell is an "init state known" cell, and the symbolic paper's "no GT"
   column has no image analogue on disk.
5. **GT is preserved** for every problem: `gt_trajectories/<problem>/` holds the
   true `.trajectory` and `_trajectory.json` (PDDLGym domains: exported before
   the noisy rebuild, `_run_generation_inference`; external domains: the
   visualizer's `trajectory_writer.py` output copied through).
6. **Prompt caching**: the backend sends `prompt_cache_key(task, system_prompt)`
   and records `prompt_tokens_total` / `prompt_tokens_cached` on the backend
   object, but nothing writes those counters anywhere; no run on disk has a
   token or latency record.

### 0.3 Mask and noise are measured, not set

An image cell has one realised masking rate (VLM `UNCERTAIN` answers) and one
realised flip rate (VLM wrong answers) per domain, computable today from
`gt_trajectories/` with `benchmark/algorithm_adapters/lamanna/realised_rates.py`.
Nothing records it for image cells yet, and the dashboard's image section has
no flipped-per-state number (`build_dashboard.py` derives it from a simulated
`noise_ratio`). Section 3.3.

### 0.4 The pixel arms on disk ran on CPU and were budget-truncated

Measured `learning_time_seconds` on the small image dirs (30 cells each):

| arm | blocksworld | depot | gripper | hanoi | npuzzle |
|---|---|---|---|---|---|
| ROSAME-I_24 (median / max) | 241 / 302 s | 121 / 180 | 174 / 301 | 132 / 255 | 300 / 301 |
| ROSAME-I_MILP_24 | 110 / 177 | 77 / 102 | 102 / 160 | 60 / 103 | 321 / 519 |
| ROSAME-I_26 (`device=cpu`) | 808 / 1358 | 1064 / 1532 | 1637 / 4960 | 2023 / 3301 | 1093 / 1500 |
| ROSAME-I_MILP_26 (`device=cpu`) | 839 / 1446 | 1070 / 1728 | 1566 / 4869 | 2015 / 3738 | 1525 / 2011 |
| PISAM_MILP_LOOP | 3 / 19 | 9 / 51 | 4 / 29 | 11 / 49 | 17 / 100 |

ROSAME-I_24 sits on the 300 s wall in every blocksworld and npuzzle cell, so
those rows stopped at whatever epoch the clock reached, not at the per-domain
epoch count (`_HYPERPARAMS` in `benchmark/baselines/rosame_i_runner.py`:
100 / 70 / 300 / 100 / 100). The 26 arms ran past the cell timeout
(`budget_mode=preflight` on CPU, up to 4960 s). All pixel rows need a GPU rerun
before they are comparable with anything. GPU nodes are agreed for both corpora.

---

## 1. Arms

The paper's grid is arm × {MILP, no MILP} × {init GT, no GT}. On images the
"init GT" column is what the data is today; the "no GT" column needs the
frame-0 classification of section 3.1. Two baseline families exist for images:
**pixel** arms that learn from the frames, and **two-stage** arms that learn
from the VLM-inferred symbolic trajectories (the same input our arms read).

| paper arm | MILP | registry key | row label (small / large) | input | notes |
|---|---|---|---|---|---|
| NPI-SAM | yes | `pisam_milp_loop` | `PISAM_MILP_LOOP__schema=pre_and_add__solve=60` / `...__m=4__solve=60` | VLM trajectories | the paper's mapped arm, settings in 2.1 |
| PI-SAM | no | `pisam` | `PISAM` | VLM trajectories | no knobs |
| ROSAME-I (pixel) | no | `rosame_i_24` | `ROSAME-I_24` | frames + actions + GT final state | 2.2 |
| ROSAME-I+MILP (pixel) | yes | `rosame_i_milp_24` | `ROSAME-I_MILP_24` (+ `__goal=none` if added) | frames + actions + GT init and final | 2.3 |
| ROSAME (two-stage) | no | `rosame_24` | `ROSAME_24` | VLM trajectories | already ran in the depot image dir; symbolic settings of 2.5 |
| ROSAME+MILP (two-stage) | yes | `rosame_milp_24`, `rosame_milp_24_tag` | `ROSAME_MILP_24__goal=none`, `..._TAG__goal=none` | VLM trajectories | the paper's mapped symbolic arm (init fixed, final free) |
| ICAPS-26 network (pixel) | no / yes | `rosame_i_26`, `rosame_i_milp_26` | `ROSAME-I_26`, `ROSAME-I_MILP_26` | frames | **decision needed**, 2.4 |

With the no-GT column (3.1): `PISAM__s0=noisy`,
`PISAM_MILP_LOOP__gt=none__s0=noisy__schema=pre_and_add__solve=60[__m=4]`,
`ROSAME_24__s0=noisy`, `ROSAME_MILP_24[_TAG]__gt=none__s0=noisy`, and for the
pixel MILP arms `--milp-gt-anchoring none` (not plumbed yet, 5.7).

Not run on images:

- `nolam`, `offlam`: regime-gated off for every image cell (`supports` in
  `nolam_runner.py:128` / `offlam_runner.py:40`: image cells both mask and flip).
  Keep the gate `strict`.
- `cdps`, `cdps_anchored`, `pisam_milp_single_round`: not in the paper
  (`paper-uses-mapped-arms-only`). Leave out unless an ablation asks for them.

Imaged rows must carry the symbolic suffix conventions so the merge config and
the dashboard treat them alike: `__solve=60`, `__schema=pre_and_add`, `__m=4`,
`__goal=none`, `__gt=none`, `__s0=noisy`, `__res=<n>`, `__ep=<n>`. The
`pisam_milp_*` labels already do (`benchmark/algorithms.py`); the imaged
ROSAME runners emit `__res=` and `__ep=`/`__mode=` but **not** `__goal=none`
or `__solve=` (5.7).

---

## 2. Settings per arm

Two levels: (a) what the symbolic sweep already pinned and the imaged arm must
match, (b) knobs the symbolic ablation never touched.

### 2.1 NPI-SAM (`pisam_milp_loop`) and PI-SAM (`pisam`)

Source of truth: the `pisam_milp:` block (small: `run_config.yaml`; large:
`run_config_large.yaml`) plus `benchmark/milp_configs/npisam_schema_nonempty_{small,large}.yaml`.

| knob | small | large | level |
|---|---|---|---|
| `gt_anchoring` | `init_only` (anchored) | `init_only` | a |
| `schema_nonempty` | `pre_and_add` | `pre_and_add` | a (the 2026-10-08 switch) |
| `time_limit_seconds` (one solve) | 60 | 60 | a |
| `subset_size` | `half` | 4 (`__m=4`) | a |
| `sampler` / `learner_input` / `pool_policy` / `w_prior` | random / subset_only / frozen / tiebreak | same | a |
| `stop.no_improvement_rounds` | blank | `auto` | a |
| `stop.budget_seconds` | inherits learning timeout | same | a |
| `eq16` | off | off | a |
| learning timeout | 300 s | 3600 s | a (2.6) |
| anything image-specific | none | none | — |

Nothing here reads images; the arm sees the VLM trajectories exactly as it saw
the simulated ones.

### 2.2 ROSAME-I, ICAPS-24 pixel arm (`rosame_i_24`)

What it is given (`benchmark/baselines/rosame_i_runner.py`,
`benchmark/algorithm_adapters/rosame_i_runner.py`): the frames, the observed
action sequence, and the **GT final state** as the only state supervision
(`gamma`-weighted anchor on the last frame's prediction). It takes **no init
state** (`rosame_i_runner.py:319-322`: "the network takes only a goal anchor").
This is the asymmetry the user expects: ROSAME-I needs the final state for the
CV backprop; symbolic ROSAME did not. The anchors come from `gt_trajectories/`
via `image_fold_inputs.resolve_fold_inputs`, never from the degraded trajectory.

| knob | value today | level | large-corpus note |
|---|---|---|---|
| epochs | per-domain table 100 / 70 / 300 / 100 / 100 (paper) | a | the runner has **no `epochs` override**: `_instantiate` drops the `epochs` kwarg, so the table is a hard ceiling. A convergence-ruled large run needs the override (5.5) |
| `lambda_`, `gamma` | 0.2 (npuzzle 0.4), 10 | a | keep |
| schedule | pooled, one optimizer step per trace, shuffled per epoch | a | the symbolic large run used `batch_size: 128`; the pixel adapter has no batch knob. Decide whether to add one or document the asymmetry |
| `n_seeds` | 3, `base_seed` 8800 | a | 3 on small; on large probably 1 (the symbolic arms use one seed); must be the same for 24 and 26 |
| `resize` | 64 (shorter edge, aspect kept) all domains | a | `__res=224` exists as a depot legibility ablation; not a paper arm |
| augmentation | horizontal flip p=0.5 | a | keep |
| CV backbone | ResNet-18 from scratch, raw logits | a | keep |
| `rosame_convergence` | off (small) | b | the imaged rule tested in `test_imaged_convergence.py` is window 40 / 0.002 / patience 3 / min 50. Large: on, with an epoch ceiling (5.5) |
| `device` | autodetect cuda > mps > cpu | b | **pin `cuda` and fail fast** (5.3) |
| memory | every trace's frames are pinned on the device at prepare time (`_prepare_trace`) | b | ~1.7 GB on the GPU at L=2000 × ~16 frames × 3×64×85 at 64 px; ~20 GB at 224 px. Needs a streaming path or a measured ceiling before a large run (5.4) |
| timeout | checked between epochs; one seed always finishes | a | on GPU the per-domain epoch count should finish inside 300 s; verify in the pilot |

### 2.3 ROSAME-I+MILP, ICAPS-24 pixel arm (`rosame_i_milp_24`)

Everything in 2.2 plus the MILP loop (`milp_loop_i.py`). What the MILP fixes:
`init_fluents` and, with `goal_mode: gt`, `goal_fluents`, both from
`gt_trajectories/` (`rosame_i_milp_runner.py:138-164`; the on-disk rows show
`n_gt_goals` = fold size, so the final state did reach it). This is the "both
init and final fixed" setting the user expects for ROSAME-I+MILP.

| knob | value today | level |
|---|---|---|
| `goal_mode` | `gt` (init + final fixed) | a; the symbolic paper arm is `__goal=none` (final free). Decide whether the pixel arm gets the same variant. The final state stays in the **loss** either way (it is ROSAME-I's only state supervision); `goal_mode none` removes it only from the MILP |
| `n_seeds` | 1 | a (the 24 pixel arm uses 3; align them) |
| `pre_mip_epochs` | 50 | a |
| `mip_interval` | 1 (doubled automatically when the remaining budget cannot afford 5 s per solve) | a |
| `mip_traces` | 3 (upstream) | b: the symbolic large run used 4 to match NPI-SAM's subset; same choice here |
| `mip_time_limit` | 60, budgeted down to the fair share of the remaining time | a; label stays bare at 60, `__solve=<n>` otherwise |
| `psi` | 0.99 (pseudo-label decay) | a |
| encoding | upstream dialect: `schema_nonempty=pre_and_add`, `forbid_redundant_adds`, `delete_implies_precondition`, prior weighting `rosame` | a |
| epochs | per-domain table, no override | same gap as 2.2 |

### 2.4 ICAPS-26 network, DL-only and +MILP (`rosame_i_26`, `rosame_i_milp_26`)

Pinned run parameters (`src/milp/rosame26_training.py`, `rosame26_runner.py`,
`rosame26_milp_runner.py`): `lr` 1e-4, `batch_size` 128 capped at the fold size,
`epochs` 600 default lowered by the §1.2 pre-flight to fit the cell timeout,
`pre_mip_epoch` 50, `mip_interval` 1, `mip_traces` 3, `mip_time_limit` 60 as a
constant, `n_seeds` 3, resize 64, device cuda-else-cpu. Both init and goal are
anchored in the MILP (§4.3 of the 26 plan).

Two documented defects (`docs/rosame-i-26-failure-suggestions.md`):

1. The MILP is **UNSAT on every gripper and hanoi solve and 89% of depot's**
   (the `add >= 1` half of `schema_nonempty`), so `ROSAME-I_MILP_26` is a
   DL-only run under a MILP label on those domains.
2. The models have almost no effects (effect recall 2 to 3% on three domains);
   gate 7 showed it is undertraining, and the code default of 5000 epochs does
   not fit any budget on CPU.

A GPU changes (2) materially (the pre-flight will grant far more epochs), so the
CPU rows on disk say nothing about this arm under the paper budget.
**Decision for Roni:** (i) keep both 26 arms as presented baselines, (ii) keep
only `ROSAME-I_26` DL-only plus a footnote, or (iii) drop them to an appendix.
If kept, run with `budget_mode: converge` (the only mode that emits the
best-loss model) and record solve usability per domain; do not report
`ROSAME-I_MILP_26` as a MILP result where usability is 0%.

### 2.5 Two-stage symbolic arms on the VLM trajectories

`rosame_24`, `rosame_milp_24`, `rosame_milp_24_tag` with exactly the symbolic
settings per corpus: small `batch_size 0`, `normalize_base_loss false`,
convergence off, 300 s; large `batch_size 128`, `normalize_base_loss true`,
convergence rule on (window 10 / 0.002 / patience 3 / min 50), `epochs 2000`
ceiling, `mip_traces 4`, 3600 s; `rosame_seed 8800`; `goal_mode none` for the
paper's mapped MILP arm and the default `gt` rows for completeness. The MILP's
init and final states come from `gt_trajectories/` by trajectory name; verify
`n_gt_goals` on image cells in the pilot.

### 2.6 Budgets, folds and training sizes

| | small (10 problems) | large (section 6) |
|---|---|---|
| `cv_scheme` | `montecarlo` (as the small symbolic grid) | `kfold` (as v4) |
| `n_folds` | 5 | 5 |
| `num_trajectories` | 3..8 | the v4 list, cut at the pool size (6.1) |
| `learning_timeout_seconds` | **300** (the image dirs on disk used 600 for CDPS and 300 for the backfilled baselines; use 300 for every arm so small-image rows are comparable with small-symbolic rows) | 3600 |
| `planning_timeout_seconds` | 60 | 60 |
| `frame_axiom_mode` | `after_gt_only` | same |

The budget is wall-clock, so the hardware class differs between the pixel arms
(GPU node) and everything else (cpu256). That asymmetry is inherent to comparing
a CV learner with a symbolic one and is stated once in the paper; what we must
avoid is the *same* arm running on two hardware classes. Every log records
`cpu_model` and should also record `nvidia-smi -L`; `device` is already in
`algorithm_specific`.

---

## 3. Data questions settled with the user, and how to do them

### 3.1 Frame 0: classify it without leaking its labels

The leak, stated precisely: today the one-shot example *is* frame 0 of the
same problem with its GT facts. Classifying frame 0 with that prompt would hand
the model the answer. Any fix must give frame 0 an example that is not itself,
and should give frames 1..T the same kind of example, or frame 0 would carry a
different noise process from the rest of the trajectory.

**Proposed protocol: one held-out example problem per domain, shared by the
whole corpus.**

- Generate one extra problem per domain (same object universe; the large
  corpora are `uniform_object_universe: true`, and the visualizer generators
  take fixed object counts). It is the example and is **excluded from the
  corpus** (never a training or test problem).
- The classifier's `fewshot_examples` becomes `[(example/frame_k.png, GT facts
  of that frame)]`, built once per domain, not per problem. `k` is any frame;
  the first is fine.
- Classify **all** frames 0..T of every problem with it, one call each.
- Write two artifacts per problem, mirroring the symbolic layout:
  `<problem>.trajectory` with the VLM state 0 (the no-GT data), and the
  init-GT copy with state 0 **replaced** by the GT state from
  `gt_trajectories/` (what the frozen observations are today). Which is the
  "frozen" one and which the staged `observations_noisy_s0/` copy follows the
  symbolic convention: frozen = init GT, staged = VLM state 0.
- Object detection stays GT (`inference_mode=False`), explicitly. Turning the
  LLM detector on would add one call per problem (cheap) but would let a
  missed or hallucinated object change the grounding, which neither PI-SAM's
  observation model nor ROSAME-I's shared grounding (identical-grounding
  assert) can absorb. Record `object_universe: gt` in the corpus metadata.

Consequences:

- The prompt prefix (system prompt + example image + example facts) is now
  **identical across the whole corpus**, so OpenAI's prompt cache applies to
  every call after the first, not only within a problem. This is also what
  makes the cost table in 6.2 hold.
- The protocol differs from the small corpora on disk (own frame 0 as the
  example). Regenerate the five small corpora under the new protocol too
  (about 650 frames, section 6.2's per-frame cost makes that a few dollars),
  so small and large image rows share one noise process.
- The few-shot example's object names must map onto the corpus's (the
  `imaged_obj_to_gym_obj_name` tables assume one fixed universe per domain;
  true for the large corpora).

Alternatives considered: leave-one-out (use another problem's frame 0 for
frame 0 only) keeps per-problem prefixes and gives frame 0 a different example
than frames 1..T; using frame T of the same problem for frame 0 leaks the
problem's object layout. Both rejected.

### 3.2 Object detection: there is nothing to measure

Because detection returns GT today (0.2), the "object detection with gpt-5.2"
cost is zero and the plan keeps it so (3.1). If Roni wants an imaged-detection
error class in the paper, that is a separate arm of the data generator, not a
default.

### 3.3 True labels and realised rates: keep (1), and report structure too

The user prefers computing the rates from the true labels of the generating
process (option 1). That is already the state of the data: `gt_trajectories/`
has the true `.trajectory` for every problem, in both the PDDLGym and the
visualizer path, and `realised_rates.py` computes hidden and flipped fractions
against it. So (1) costs one call per fold instance and a dashboard change.

On "the rates won't tell us much": they tell one thing that nothing else can,
which is where an image cell sits on the simulated grid. A sentence like "the
VLM corruption on blocksworld is equivalent to mask ≈ x / noise ≈ y, and
NPI-SAM at that simulated cell behaves the same" is the bridge between the two
experiments, and it is what justifies the simulated grid as a model of the real
pipeline. What the rates do **not** capture is structure, and that is worth
reporting beside them:

- per-predicate confusion (which predicates the VLM gets wrong;
  `benchmark/evaluation/trajectory_fluent_confusion.py` exists),
- persistence: the fraction of a frame's flips that recur in the next frame.
  The simulated corruption draws every hidden or flipped atom independently,
  per atom and per state; VLM errors are not independent (the same predicate
  tends to be wrong in consecutive frames, and errors cluster on the same
  objects), and frame-axiom propagation and the MILP react differently to the
  two,
- the rate at frame 0 after VLM classification against frames 1..T (3.1 makes
  frame 0 ordinary; this checks it).

Record all of it per fold instance in `fold_info.json` (`realised_mask_rate`,
`realised_flip_rate`, `n_atoms`, plus the two structure numbers) and let the
dashboard's image section read it instead of the simulated `noise_ratio`.
Option (2), not showing flips, is rejected: the flip rate is the one number
that places the image experiment on the simulated axes.

### 3.4 Supervision parity: what each arm is told, and how to equalise it

The evaluation itself is identical for every arm (held-out GT test problems,
the same AMLGym metrics on the emitted PDDL). The asymmetry is in the
**training input**: ROSAME-I cannot run without a labelled frame per
trajectory, because the CV head has to be told what the symbols mean, while
our arms were already run with one clean state (`gt=init`) and with none
(`gt=none`). Counting clean states per training trajectory:

| clean states per trajectory | our arms | pixel arms | two-stage symbolic arms |
|---|---|---|---|
| 0 | `PISAM__s0=noisy`, `PISAM_MILP_LOOP__gt=none__s0=noisy…` | **new** `ROSAME-I_24__anchor=vlm` (3.4.1) | `ROSAME_24__s0=noisy`, `ROSAME_MILP_24__gt=none__s0=noisy` |
| 1 (init) | `PISAM`, `PISAM_MILP_LOOP__schema=…__solve=60` | **new** `ROSAME-I_24__anchor=init` (3.4.2) | `ROSAME_24`, `ROSAME_MILP_24__goal=none` |
| 1 (final) | — | `ROSAME-I_24` (upstream: goal anchor) | — |
| 2 (init + final) | **new** `PISAM_MILP_LOOP__gt=all…` (3.4.3) | `ROSAME-I_MILP_24` (upstream), `ROSAME-I_26`, `ROSAME-I_MILP_26` | `ROSAME_MILP_24` (`goal_mode gt`) |

Proposal: present the comparison **row by row of this table**, and say in the
text which row is which. Three additions make every row populated on both
sides:

1. **`ROSAME-I_24__anchor=vlm` (0 clean states).** The anchor stays at
   frame 0 (the position of the Init row) but its target is the VLM's own
   classification of frame 0 instead of the GT state; atoms the VLM left
   `UNCERTAIN` drop out of the anchor loss. This is the same swap the symbolic
   arms undergo between the Init and No-GT rows (state 0 still exists, its
   source changes from GT to VLM), so the pixel arm has exactly the
   information our `gt=none` arms have. An anchor is unavoidable: without one
   nothing ties the CV head's output slots to named predicates and the
   trivial constant-state solution minimises the loss. Code:
   `_prepare_trace` reads the staged trajectory's state 0 and its masking
   line instead of `gt_init_predicates`; a masked-MSE in
   `_loss_from_predictions`.
2. **`ROSAME-I_24__anchor=init` (1 clean state, same position as ours).** The
   `gamma` anchor moves from `domain_preds[-1]` to `domain_preds[0]`
   (`rosame_i_runner.py:320`); `ResolvedTrace.gt_init_predicates` is already
   resolved for the 26 arms. This is the exact information NPI-SAM anchored
   gets. Upstream anchored the goal because the ICAPS-24 problems came with a
   known goal; anchoring the known initial state is the same assumption at the
   other end, and it is what our arms assume.
3. **`PISAM_MILP_LOOP` with init and final fixed (2 clean states).** The data
   prep exists (`cdps_anchored`'s `anchor_endpoints=True` →
   `inject_gt_states_by_percentage(..., anchor_final=True)`) and the encoder
   has `GtAnchoring.ALL_GT_STATES`; the two are not wired together for the
   `pisam_milp_loop` phase (`run_fold.py:729` only does it for CDPS). Wire it
   and label the row by the anchoring (`__gt=all`, check the suffix
   `_shared_milp_suffix_parts` emits). This is the row to put next to
   `ROSAME-I_MILP_24` upstream and the 26 arms.

What this does **not** equalise: the pixel arms also receive the GT **object
universe** (3.2) and, through the VLM pipeline's one-shot example, so do ours;
that is domain-level supervision shared by both and is stated once. A stricter
symmetric variant, giving ROSAME-I's CV head only the held-out example
problem's labelled frames (the same single labelled problem the VLM sees) and
no per-trajectory anchor, is possible but likely leaves the CV head unable to
generalise from one trajectory; propose it as an appendix row at most, not as
a headline comparison, since it would read as handicapping the baseline.

Recommended headline rows for the paper's image section: row "1 (init)" for
NPI-SAM vs ROSAME-I (init anchor), with the upstream goal-anchored ROSAME-I
beside it to show the position does not change the story (or that it does);
row "2" for the MILP pair; row "0" to show what each side does with no GT at
all. Decision for Roni in section 7.

### 3.5 Alignment checks per arm (pilot checklist)

1. ROSAME-I_24 gets the GT final state and no init state (code: yes). Pilot:
   `algorithm_specific` has no init anchor field; loss includes the `gamma` term.
2. ROSAME-I_MILP_24 fixes init and final in the MILP (`goal_mode gt`). Pilot:
   `n_gt_goals == num_trajectories` on every cell of every domain.
3. ROSAME-I_MILP_26 fixes init and goal in the MILP (§4.3). Pilot:
   `milp_rounds[*].exit_status` per domain.
4. NPI-SAM anchors state 0 only (`gt_anchoring init_only`) on the frozen
   (init-GT) data; the `__gt=none__s0=noisy` row runs on the staged copy of 3.1.
5. PI-SAM raw: `PISAM` on frozen, `PISAM__s0=noisy` on staged.
6. Two-stage ROSAME+MILP receives init (and with `goal_mode gt` the final)
   from `gt_trajectories/`. Pilot: `n_gt_goals`.
7. Realised rates and structure numbers (3.3) are in every `fold_info.json`.
8. The new parity variants (3.4) record their anchor in `algorithm_specific`
   (`anchor: gt_final | gt_init | vlm_final`; `gt_anchoring: all_gt_states`).

---

## 4. Cluster: GPU, resources, job layout

### 4.1 Which GPU, and how

The pixel arms are ResNet-18 on 64 px frames; VRAM need is well under 8 GB at
64 px, so any card is enough. What matters is **one card type for every pixel
task**, for the same reason the symbolic sweep pinned `cpu256`: a wall-clock
budget is hardware-dependent. Pick by availability:

```bash
sinfo -o "%P %N %G %f %T" | grep -i gpu
```

and choose the type with the most idle nodes on `main` (per the BGU skill:
`gtx_1080` / `rtx_2080` at 8 GB, `rtx_3090` / `rtx_4090` / `titan_rtx` at 24 GB,
`rtx_6000` at 48 GB, `tesla_p100` at 16 GB). Then in the pixel template:

```bash
#SBATCH --partition main
#SBATCH --gpus=1
#SBATCH --constraint <chosen_type>      # e.g. rtx_2080
#SBATCH --cpus-per-task 6
#SBATCH --mem 24G                       # the per-GPU default; cap is 60G
```

`--qos` partitions only with a Golden Ticket; otherwise `main`. One GPU per
job. The job verifies the device before learning:

```bash
nvidia-smi -L
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || exit 1
```

and the runners are told `device=cuda` explicitly (5.3), so a job that lands
without a usable CUDA build fails instead of training on CPU.

Environment: `vip_venv11` on the cluster holds `torch==2.9.1`; check it is a
CUDA build (`torch.version.cuda`) on a GPU node. If not, a separate env with the
CUDA wheel per the BGU skill §7, used only by the pixel template.

### 4.2 Two job families, two out roots

| family | arms | node | template (new, `scripts/cluster_image/`) | out root |
|---|---|---|---|---|
| pixel | `rosame_i_24`, `rosame_i_milp_24`, (`rosame_i_26`, `rosame_i_milp_26`) | GPU, one type | `pixel_fold.sbatch` → `backfill_baseline.py --out-root ... --device cuda` | `benchmark/running_results_image_pixel/` |
| symbolic | `pisam_milp_loop`, `pisam`, `rosame_24`, `rosame_milp_24[_tag]` | `cpu256`, `--gpus=0` | `symbolic_fold.sbatch` → `benchmark_runner` (our arms) then `backfill_baseline` (two-stage arms) | `benchmark/running_results/` (our arms) + `benchmark/running_results_image_symbolic/` (baselines) |

One job per output cell at a time (`one-job-per-output-cell`): the two families
never share an out root, and within a family one task owns a cell. The staged
no-GT copies get their own roots, as on the symbolic side.

Granularity: small corpora, one task per (domain, all folds), as
`anchored_small_cell_manifest.csv` does. Large corpora, one task per
(domain, fold), the whole L sweep inside the task, as `fold_manifest.csv` does.

### 4.3 Resources per task (to confirm in the pilot, 5.9)

| | pixel task (small) | pixel task (large, per fold) | symbolic task (small) | symbolic task (large, per fold) |
|---|---|---|---|---|
| GPU | 1 | 1 | 0 | 0 |
| CPUs | 6 | 6 | 3 | 6 |
| RAM | 24G | 24G, remeasure after 5.4 | 12G | 24G (v2/v4 peaks 7 to 15 GB per 5-fold cell) |
| `--time` | 1-00:00:00 | 3-00:00:00 | 1-00:00:00 | 3-00:00:00 |
| arms × cells | 2 to 4 arms × 30 cells × ≤300 s ≈ 2 to 10 h | unknown until the pilot | minutes | as v4 (fold ≈ 2.5 h at L≤2000) |

Caps (`bgu-cluster-per-user-caps`): ~265 concurrent 6-CPU jobs, ~2000 queued
tasks; GPU concurrency is bounded by the chosen type's node count. Submit per
domain as independent arrays, no chaining.

### 4.4 Disk

Frames are small: the renders are flat-colour RGBA PNGs. Measured mean size
over every frame on disk, and the projection (frames per problem = mean
actions + 1, from the symbolic corpora):

| domain | mean KB / frame | frames / problem | GB at 500 | GB at 1000 | GB at 2500 |
|---|---|---|---|---|---|
| blocksworld (480×480) | 2.6 | 12.6 | 0.02 | 0.03 | 0.08 |
| hanoi (630×224) | 1.6 | 16.0 | 0.01 | 0.03 | 0.07 |
| npuzzle (187×187) | 7.8 | 15.1 | 0.06 | 0.12 | 0.30 |
| depot (800×600) | 13.6 | 17.0 | 0.12 | 0.24 | 0.59 |
| gripper (800×600) | 17.1 | 16.6 | 0.15 | 0.29 | 0.73 |
| **five domains** | | | **0.35 GB, 39k files** | **0.71 GB, 77k files** | **1.77 GB, 193k files** |

Add the symbolic side of each corpus (the 2500-window blocksworld corpus is
102 MB and 12.5k files, depot 220 MB) and the VLM output (one `.trajectory`,
`.masking_info`, `_trajectory.json` and `.pddl` per problem, a few KB each,
four more files per problem). Bytes are not the constraint anywhere; at 2500
problems the five image corpora are under 2 GB. **Files are**: home is a
file-count quota (`pack_fold.sh` exists because of it), and 193k PNGs plus
~50k symbolic files is more than the whole symbolic sweep left on disk after
packing. So: keep each corpus as one tarball on the cluster and unpack it to
the job's local SSD (`--tmp=10G`, `/scratch`) at task start, or under group
storage rather than home if that is file-count-free. On the laptop and for the
pixel arms' training nothing special is needed. Results stay packed
(`pack_fold.sh`); leave `baseline_models/<arm>/seed_*/model.pddl` (the merge
copies them).

---

## 5. To settle, check or code before anything runs

1. **Decide with Roni**: the 26 arms (2.4); the pixel `goal=none` variant (2.3);
   whether the no-GT column is presented for images (3.1 makes it possible).
2. **Build the corpora** (section 6), after the pre-check (6.3).
3. **Device plumbing.** Add `device` to `backfill_baseline.py` (`--device`) and
   to `benchmark_runner._RUNNER_KWARG_KEYS`; make the imaged runners raise when
   `device=cuda` is requested and `torch.cuda.is_available()` is false.
4. **Record realised corruption and structure for image cells** (3.3), and
   teach the dashboard's image section to read it.
5. **Epoch ceiling for the 24 pixel arms.** `RosameIBaselineRunner` and
   `RosameIMilpRunner` take no `epochs`; add it, suffix `__ep=` when
   off-default, as the 26 runner does.
6. **Large-corpus memory in the pixel adapters.** Measure at the planned L;
   add streaming if the GPU or RAM figure in 4.3 is exceeded.
7a. **Parity variants (3.4)**: `anchor` knob on `rosame_i_24` (`gt_final` |
   `gt_init` | `vlm_final`, suffix `__anchor=` when not `gt_final`), and
   `anchor_endpoints` + `gt_anchoring: all_gt_states` wired into the
   `pisam_milp_loop` phase of `run_fold.py`.
7. **Labels and flags on the imaged ROSAME runners**: `__goal=none`,
   `__solve=<n>`, `--milp-gt-anchoring none`. Register the new labels in
   `dashboard_config.yaml` (`algorithms`, `modes: [image]`) and
   `paper_result_sources.yaml` (an image `paper_arms` block) before the first merge.
8. **Manifest and templates.** `scripts/cluster_image/make_manifest.py` (rows:
   `domain, data_dir, fold, run_name`; no mask/noise grid), `pixel_fold.sbatch`
   (GPU), `symbolic_fold.sbatch` (cpu256), reusing `pack_fold.sh`.
9. **Pilot on one domain (blocksworld, small), both families, on the cluster,
   before any array.** Read off: GPU type, `torch.version.cuda`, VRAM and RSS
   peaks, per-arm `learning_time_seconds` (does ROSAME-I_24 now finish its 100
   epochs inside 300 s?), `n_gt_goals`, the 26 arms' solve exit statuses, the
   realised rates. Then size `--time`/`--mem` and submit.
10. **Fresh small image run, not a backfill into `TO=600__*`.** New `run_name`
    per domain on the regenerated small corpora (3.1), then repoint
    `dashboard_config.yaml`'s `image.small.experiment_dir`.
11. **Timeouts the 26 arms do not honour**: `converge` mode or a hard
    `timeout_check` in `Rosame26Trainer._run_training`.
12. **Thread pinning.** CPU templates keep `OMP_NUM_THREADS=1`; the GPU template
    leaves torch's data-path threads alone but pins BLAS threads for CP-SAT.
13. **Data-generator changes for 3.1**: shared held-out example, classify
    frame 0, write the init-GT and VLM-state-0 variants, record
    `object_universe: gt` and the example problem id in `generation_info.json`.
14. **Usage logging**: write one JSONL line per VLM call (`problem, frame,
    prompt_tokens, cached_tokens, completion_tokens, latency_s, retries`) next
    to the corpus. The counters exist on the backend; nothing persists them.
15. **Batch API path (optional, halves the bill)**: a second backend that
    writes the requests of a whole corpus to a Batch job and reads the results
    back. Only worth it if 6.2's total matters; it trades a 24 h turnaround for
    50% off.
16. Keep `events_tracing: false`, `resume: true`, and `merge_provenance.json`.

---

## 6. The large image corpora

Decided: build them, five domains, VLM inference with `gpt-5.2` as in the small
corpora, trajectory lengths as in the symbolic large corpora. Blocksworld,
hanoi and npuzzle frames come from PDDLGym rendering; depot and gripper frames
from `planning-visualizer`.

### 6.1 Render the symbolic large corpora rather than walk again

Recommendation: do **not** run new random walks. The symbolic large corpora
(`benchmark/data/*/large_corpus*`, 2500 windows each, uniform object universe,
GT `.trajectory` per window) already are the trajectories; render their states
to frames and run the VLM over those frames. Then the image experiment and the
symbolic experiment share the **same trajectories**, and every image row has a
symbolic twin at every simulated `(mask, noise)` cell. That is a much stronger
comparison than two corpora that merely share a length range, and it removes
one source of variance.

What it needs:

- **PDDLGym domains**: a renderer entry point that takes a GT state (literals +
  objects) and returns the frame. `pddlgym_problem_generator.py` renders from
  the gym `State` it walked; the same renderer has to be fed a `State` built
  from a trajectory line (object names already match: blocksworld `a..e`,
  hanoi `d1..d4` / `peg1..3`, npuzzle `p_i_j`). Feasibility check: render three
  windows of each corpus and compare against walk-rendered frames.
- **depot and gripper**: `planning-visualizer/backend/planner/generate_images.py`
  renders from PDDL facts per step (`StateGenerator` → `RendererFactory` →
  `pil_renderer`), so it can take our GT states directly; its own generators
  produce *planner* traces and goal-directed problems, which is not what the
  symbolic large corpora are (random walks, `p_rnd 1.0`). Feed it the symbolic
  windows: `(domain, problem.pddl, action list)` → frames, GT kept from our
  side. The visualizer's domain files must equal `src/domains/{depot,gripper}/*.pddl`
  (the small corpora went through that path, so they do; re-check after any
  change). Object counts fixed to the symbolic universes (depot 2 cranes, 2
  depots, 3 packages, 2 piles, 1 truck; gripper 6 balls, 2 grippers, 2 rooms).

If a renderer cannot be fed from a trajectory, the fallback is the small-corpus
way: a long random walk cut into windows (`--gen-mode generate`) with the
symbolic length ranges (blocksworld 10–13, hanoi 9–20, npuzzle 10–18, depot
14–18, gripper 11–20). Then the trajectories differ from the symbolic ones and
only the length distribution is shared.

Pool sizes under discussion: 500, 1000, 2500 problems per domain (the symbolic
corpora are 2500 = 2000-problem train pool + 500 test under `kfold`). The
training-size list follows: 500 → L ∈ {10, 50, 100, 400}; 1000 → {10, 50, 100,
500, 800}; 2500 → the v4 list {10, 50, 100, 500, 2000}. Rendering is cheap at
any size; the VLM pass is the cost (6.2). A 2500-problem image corpus is a
strict superset of a 1000- and a 500-problem one if the problems are a prefix
of the symbolic corpus, so starting at 500 and growing later wastes nothing.

### 6.2 VLM cost and latency, 500 / 1000 / 2500 problems per domain

Measured inputs: mean actions per trajectory in the symbolic large corpora
(blocksworld 11.6, hanoi 15.0, npuzzle 14.1, depot 16.0, gripper 15.6) and the
frame sizes of 0.1. Frames to classify, **frame 0 included** (3.1):

| problems per domain | frames (5 domains) |
|---|---|
| 500 | 38,650 |
| 1000 | 77,300 |
| 2500 | 193,250 |

Pricing used (`gpt-5.2`, standard tier, as of the search on 2026-10-24):
$1.75 per 1M input tokens, $0.175 per 1M cached input tokens, $14.00 per 1M
output tokens; Batch API half of each. Image input is billed by 32×32 patches
(`ceil(w/32)·ceil(h/32)`: blocksworld 225, hanoi 140, npuzzle 36, depot and
gripper 475 per image), times a model multiplier that the public docs give for
other models and not clearly for 5.2, so it is bracketed 1× to 2× below. Text
prefix per call estimated from `src/llms/domains/*/prompts.py` sizes (1.1k to
2.3k tokens). Output per call is one relevance line per grounded predicate,
bracketed 300 to 600 tokens.

Per frame, with the shared prefix cached (3.1) and the target image uncached:

| domain | $ per frame, low (1×, 300 out) | $ per frame, high (2×, 600 out) |
|---|---|---|
| blocksworld | 0.0049 | 0.0095 |
| hanoi | 0.0049 | 0.0094 |
| npuzzle | 0.0047 | 0.0089 |
| depot | 0.0054 | 0.0105 |
| gripper | 0.0054 | 0.0105 |

Corpus totals, five domains, standard tier (Batch API: half):

| problems per domain | low | high |
|---|---|---|
| 500 | $196 | $380 |
| 1000 | $393 | $759 |
| 2500 | $982 | $1,898 |

Output tokens are 55 to 65% of the bill at every setting; the image tokens
matter less than expected, and the cached prefix is nearly free. Two
consequences: (i) a shorter answer format (only predicates the model asserts
true or false, nothing for `UNCERTAIN`) would cut the bill by about a third
but changes the classifier's contract; leave it unless the pre-check says the
answers are long; (ii) the Batch API is the one lever that halves everything
without touching the protocol.

If the prefix does **not** cache (today's per-problem example, or a prefix
under the 1024-token cache floor), add 25 to 45% to every figure.

Latency, per domain, sequential problems with 12 concurrent calls per window
(today's default): `problems × ceil(frames/12) × per-call latency`. Per-call
latency for a `gpt-5.2` image call is the one number we do not have; at 5 to
15 s it gives

| problems per domain | ≈ hours per domain at 5 s / 15 s |
|---|---|
| 500 | 1.4 / 4.2 |
| 1000 | 2.8 / 8.3 |
| 2500 | 7 / 21 |

Five domains in parallel keep the wall-clock at the per-domain figure, subject
to the account's rate limits (requests and tokens per minute), which the
pre-check also measures. Rendering is negligible beside this. Generation runs
locally, as before (`cluster_large/README.md`); the corpus then goes to the
cluster as one tarball per domain.

### 6.3 Pre-check protocol (to replace the brackets above with measurements)

One afternoon, about $5:

1. Implement 5.13 and 5.14 (shared example, frame 0, usage JSONL) on a branch.
2. Per domain, take **5 problems** of the symbolic large corpus, render them
   (6.1 feasibility at the same time), and run the VLM pass twice:
   once with the shared-example protocol, once with today's own-frame-0
   protocol, both with the usage JSONL on.
3. From the JSONL: `prompt_tokens` and `cached_tokens` per call (gives the
   real image multiplier and the cache hit ratio, expected > 0.9 after the
   first call), `completion_tokens` (the real output bracket), per-call
   latency and the retry rate, requests and tokens per minute at 12 workers
   (whether the rate limit binds and how far we can raise `--inference-workers`).
4. Diff the two protocols' trajectories against GT: realised mask and flip
   rates with and without the shared example, so the protocol change is
   quantified before the small corpora are regenerated.
5. Rerun the cost table with the measured per-call tokens; pick the pool size
   with Roni; decide standard vs Batch API from the measured latency.

### 6.4 Evaluation needs no frames

Test problems are scored from `gt_trajectories/` and the problem files
(`fold{N}_gtrate{R}_shared/` S_test, predictive-power states), so only training
frames go through the VLM. Under `kfold` every problem is in some fold's train
pool, so that is still every problem; no saving there.

---

## 7. Open questions (for the meeting)

1. Arm set: keep the ICAPS-26 arms? Present ROSAME-I+MILP with init+final fixed
   (upstream) or also with the final state free (the symbolic paper arm)?
2. Is the no-GT column presented for images (3.1 makes it available)?
2b. Supervision parity (3.4): add `ROSAME-I_24__anchor=init`,
   `ROSAME-I_24__anchor=vlm` and NPI-SAM with init+final fixed, and present
   the comparison by clean-states-per-trajectory?
3. Corpus size: 500, 1000 or 2500 problems per domain, given 6.2; standard or
   Batch API.
4. Render the symbolic corpora (6.1, recommended) or walk again?
5. Budget: 300 s small / 3600 s large as the symbolic runs, pixel arms on a GPU.
6. One GPU type for all pixel tasks: which one is plentiful on `main`?
