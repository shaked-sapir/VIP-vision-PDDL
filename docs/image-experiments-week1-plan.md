# Imaged experiments: the first week (9 to 13 Nov 2026)

Status: proposal, 2026-11-07, after the "Dilemmas & sub-experiments" slide and
the WhatsApp thread on it (Shaked, Yarin, Roni). Companion to
`docs/image-experiments-plan.md` (the full plan; section numbers below marked
"plan §" refer to it) and `docs/image-experiments-meeting-brief.md`.

Everything marked "checked" was read in the repo or in result files on
2026-11-07.

---

## 1. What the thread settled

| slide dilemma | outcome in the thread | what it means for the week |
|---|---|---|
| ICAPS-26 arms (new encoder + decoder) | **in**: compare against the 26 arms too | the two 26 defects (`docs/rosame-i-26-failure-suggestions.md`) must be fixed before any new 26 row is produced, or the rows are not what their label says |
| resolution (64 px default vs 800×600 depot/gripper) | **per-domain resolution** | `resize` is already a per-domain table (`rosame_i_runner.py:_RESIZE`, `None` = native); the work is choosing the value per domain and measuring VRAM, not code |
| which VLM (gpt-5.2 vs newer vs older) | **one model to start**; Yarin, with Roni's like: try the open Qwen VLM on the cluster on the small corpora first, for quality and throughput, and use it on the large corpus if it holds up | the week's data track is a **pre-check on two models**: `gpt-5.2` (the small corpora's model) and the Qwen model, same 25 problems, same prompt, realised rates compared |
| ROSAME stopping criteria | **the papers' epoch counts as the starting point**, time cap on top | 24 arms: the per-domain table (100/70/300/100/100) under a 300 s cap on GPU. 26 arms: `converge` mode with the ceiling, because the fixed count never fits a budget on these arms |
| fluent classification settings (1-shot today) | **start from the existing 1-shot**, widen later (zero-shot; the init image, maybe the goal image, given as an image instead of symbols) | week 1 fixes the 1-shot protocol's frame-0 leak; the variants are week 2+ and are **different corpora**, not different arms |

One thing the thread did not say and the week must: the one-shot example
today is **frame 0 of the same problem with its GT facts**, and frame 0 is
never classified (plan §0.2). That is why every image cell is an "init GT"
cell. The shared held-out example (plan §3.1) is the prerequisite for the
no-GT column on images and for any VLM comparison that includes frame 0.

---

## 1a. One list: the earlier plan's §5 items, their status, and who owns them

So nothing is done twice. "plan §5.n" is `docs/image-experiments-plan.md`.

| plan item | status (checked 2026-11-07) | this week |
|---|---|---|
| §5.1 decide 26 arms / pixel `goal=none` / no-GT column | 26 arms decided **in** (thread); the other two still open | A5 asks them with the model decision |
| §5.2 build the corpora after the pre-check | nothing built or coded | A1–A6, A7 |
| §5.3 `--device` plumbing + fail-fast | not done (`backfill_baseline.py` has no `--device`) | B2 |
| §5.4 realised corruption + structure in `fold_info.json`, dashboard reads it | not done | C1, C2 |
| §5.5 `epochs` override for the 24 pixel arms | not done | B2 |
| §5.6 large-corpus memory in the pixel adapters | not measured | B4 (pilot), week 2 |
| §5.7a parity variants (`anchor=init` / `vlm`, NPI-SAM both endpoints) | not done | C3 does `anchor=init`; the rest week 2 |
| §5.7 `__goal=none` / `__solve=` labels on the imaged ROSAME runners; register labels | not done | C2 registers; the runner labels with B5 |
| §5.8 `scripts/cluster_image/` manifest + templates | not done | B5 |
| §5.9 pilot on one domain on a GPU node | not done | B4 |
| §5.10 fresh small image run, repoint the dashboard | not done | C4 |
| §5.11 timeouts the 26 arms do not honour | not done | B3(b) |
| §5.12 thread pinning in the GPU template | not done | B5 |
| §5.13 shared held-out example, frame 0, two trajectory variants | not done | A1 |
| §5.14 usage JSONL per VLM call | counters exist on the backend, nothing persists them | A1 |
| §5.15 Batch API backend | optional; decide at A5 | week 2 if chosen |
| §5.16 `events_tracing: false`, `resume: true`, provenance | config hygiene | C4 |
| new: 26-arm fidelity (3.2b) | audited; one bug, four unregistered knobs, two doc errors | B3 |
| new: open VLM (Qwen) on the cluster | proposed in the thread, endorsed | A2, A4 |

---

## 2. The week in three tracks

The tracks touch different files and can run in parallel. Each person owns
one track; the Thursday checkpoint joins them.

### Track A: VLM data (the critical path)

| day | step | output |
|---|---|---|
| Sun | **A1. Shared held-out example + frame 0 + usage log** (plan §5.13, §5.14). One extra problem per domain is the example for the whole corpus; every frame 0..T is classified; two trajectory files per problem (state 0 = VLM; state 0 = GT); one JSONL line per call (`problem, frame, prompt_tokens, cached_tokens, completion_tokens, latency_s, retries`). `generation_info.json` records model id, example problem id, a prompt hash, `object_universe: gt`. | a branch, `test_data_generator.py` extended |
| Sun | **A2. `base_url` on the OpenAI backend.** `OpenAIImageLLMBackend.__init__` takes `api_key, model, temperature` only (checked). A `base_url` field in `config.yaml` makes a vLLM server serving Qwen a third "vendor" through the same code path, no new backend class. | one small change + a config entry |
| Mon | **A3. Pre-check, gpt-5.2** (plan §6.3): 5 problems × 5 domains, both protocols (own frame 0 as today, shared example), usage JSONL on. Cost about $5. | realised mask / flip rates per domain and protocol; tokens, cache hit ratio, latency, retry rate |
| Mon–Tue | **A4. Pre-check, Qwen** (Yarin proposed it; natural for them to own). vLLM on one GPU node, the same 25 problems through A2. | the same table for Qwen, plus frames/minute on one GPU |
| Wed | **A5. Decide** (with Roni, by message if needed): the model, the corpus size (500 / 1000 / 2500, plan §6.2), standard vs Batch API. Rule: pick the operating point on purpose, write down why (brief §2). | one line in this doc |
| Wed–Thu | **A6. Regenerate the five small corpora** under the shared-example protocol with the chosen model (about 650 frames). New dir names; the old ones stay for the `TO=600__*` results. | five corpora with `gt_trajectories/`, usage JSONL, generation info |

Independent of A1–A6 and doable by whoever is free (it needs no VLM):

- **A7. Renderer for the symbolic large corpora** (plan §6.1). `--render` in
  `data_generator.py` only copies frames that a trajectory source already has
  (checked: `emitter.py:_write_images`); it does not render from a state. Needed:
  a state→frame entry point for the PDDLGym domains, and feeding our windows to
  `planning-visualizer/backend/planner/generate_images.py` for depot/gripper.
  Feasibility test: three windows per domain, compare with walk-rendered frames.
  This decides whether the large image corpora are the symbolic corpora's twins.

  **Done 2026-11-08, gate passed** (branch `claude/render-pipeline-4f6e5c`).
  Rendering is now a strategy on the source: `src/trace_generation/frame_renderer.py`
  (`FrameRenderer`, `RenderedSource`), `pddlgym_renderer.py` (blocksworld, hanoi,
  npuzzle through `pddlgym.rendering.*`, fed from eval-schema states by a
  per-domain `GymSchema`), `visualizer_renderer.py` (depot, gripper) and
  `renderer_registry.py`. `benchmark/render_gate.py` re-renders a stored
  problem folder and counts differing pixels per frame;
  `benchmark/test_render_gate.py` is the same gate as a test. Result, three
  windows per domain:

  | domain | reference frames | result |
  |---|---|---|
  | hanoi | three windows of the old PDDLGym walk (`benchmark/data/hanoi/render_gate_reference_n3/`, seed 42) | identical, every frame |
  | npuzzle | same, `npuzzle/render_gate_reference_n3/` | identical, every frame |
  | blocksworld | same, `blocksworld/render_gate_reference_n3/` | identical when block colours are primed in the walk's own first-sight order (`--blocks-colors encounter`); with the renderer's sorted priming only blocks d and e swap colours (6052 px per frame), see below |
  | depot | `src/domains/depot/problems/problem{1,2,3}` (the small corpus's source frames) | identical, every frame |
  | gripper | `src/domains/gripper/problems/problem{1,2,3}` | identical, every frame |

  Three facts the gate turned up:

  1. The depot and gripper frames were **not** made by the visualizer's PIL
     path (`generate_images.py`) but by its Node canvas path
     (`backend/api/generate_images.ts --no-metadata`: the frontend's TypeScript
     renderers plus `truck.png` / `rooboot.png` sprites, no step overlay). The
     PIL path draws a different picture. `VisualizerRenderer` therefore runs
     the Python layout (`RendererFactory`) in-process and streams one JSON per
     frame to a persistent Node worker, `backend/api/render_states.ts`, added to
     the planning-visualizer checkout (commit b71a987 there). Rendering depot and
     gripper needs Node and the `canvas` package, so those two domains render
     on the laptop (about 12 frames/s), not on the cluster.
  2. The visualizer places same-typed objects in the order its own PDDL parser
     lists them, which for `d1 d2 - depot` is d2 before d1. `VisualizerRenderer.use_problem`
     takes that order from the problem file; the large corpora's problem files
     list one object per line, so there the order is the sorted one.
  3. PDDLGym colours blocks in order of first sight within a process
     (`pddlgym/rendering/blocks.py`, seeded RNG). A fresh walk from the gym's
     bundled problem10 meets the blocks as a, b, c, e, d, so **d was yellow and
     e green** in every old generate-mode blocksworld corpus, while
     `LLMBlocksFluentClassifier.imaged_obj_to_gym_obj_name` maps red→a,
     cyan→b, blue→c, green→d, yellow→e. The renderer primes the colours in
     sorted name order, which gives exactly the classifier's map and is the
     same in every process, so cluster chunks agree. The old small blocks
     corpora should be read with that swap in mind.

  The visualizer's domain files are not byte-equal to `src/domains/{depot,gripper}/*.pddl`
  (an `(and ...)` wrapper around single preconditions; the visualizer's depot
  `drop` lacks our `(not (clear ?p))` effect). They are never applied here:
  the renderer is fed our GT states, never actions, so only the predicate
  vocabulary has to agree, and it does.

  **All five large corpora rendered 2026-11-08** (laptop, one process,
  `scripts/cluster_large/render_local.sh all` equivalent, 1 h 40 min in all):
  every problem of the five corpora holds one `state_NNNNNN.png` per GT
  state, checked against the operator count of
  `gt_trajectories/problemN/problemN.trajectory` for all 2500 problems of each
  corpus; GT and problem files untouched. Since 2026-11-10 the frames live in
  the image pool, `training/image_trajectories/problemN/` (with a copy of the
  problem's pddl and GT json; the VLM output lands there too), and
  `training/trajectories/` is the symbolic pool only; see
  `src/utils/corpus_layout.py`. The ten existing corpora were moved over with
  `scripts/migrate_image_pool.py`.

  | corpus | frames | wall time |
  |---|---|---|
  | depot `large_corpus_n2500` | 42,579 | 33 min |
  | gripper `large_corpus_n2500` | 41,428 | 34 min |
  | blocksworld `large_corpus_L500_n2500` | 31,425 | 8 min |
  | hanoi `large_corpus_n2500` | 39,920 | 7 min |
  | npuzzle `large_corpus_n2500` | 37,828 | 19 min |

  `render_corpus.sbatch` + `render_manifest.csv` stay as the cluster route for
  a future corpus; nothing was submitted.

### Track B: pixel arms on the GPU

| day | step | output |
|---|---|---|
| Sun | **B1. GPU inventory.** `sinfo -o "%P %N %G %f %T" \| grep -i gpu`; pick the one type with the most idle nodes; on one node check `torch.version.cuda` in `vip_venv11` (plan §4.1). | the pinned GPU type and env |
| Sun–Mon | **B2. Device plumbing.** `--device` is not a `backfill_baseline.py` flag (checked: the flag list has `--resize`, `--epochs`, `--n-seeds`, `--budget-mode`, `--goal-mode`, `--milp-gt-anchoring`, no device). Add it, and make every imaged runner raise when `cuda` is requested and unavailable. Add `epochs` to the 24 runners (today `_instantiate` drops it). | flags + fail-fast |
| Mon–Tue | **B3. The 26 fixes**, in the order the audit (section 3.2b) ranks them. (a) **DONE 2026-11-07 on `milp-chain-probe`, re-run pending.** The spelling boundary lives in the repairer, not the runner: `CpSpelling` in `src/milp/rosame26_repairer.py` respells the contexts and the proposition index once, at construction, so anchors, the CV columns into the solver and the pseudo-label columns back out all cross in the CP dialect. `converter._resolve_fluents` raises on any unresolved anchor; `proposition_columns` (shared by `cv_predictions_to_trace` and `state_labels_for`) raises on a column naming an undeclared predicate and still skips one over an object the instance lacks. Tests: a gripper micro-fold through the real encoder in `test_rosame26_repairer.py` (before the fix it solved and wrote a wrong `carry` pseudo-label), plus raise cases in both test files. This was the real cause of the gripper/hanoi UNSAT, not `schema_nonempty`, which stays as upstream has it. (b) Enforce the wall clock inside the 26 training loop (the preflight projects from a 0.3 s solve; a cap-hitting solve overruns the cell silently; the CPU rows ran to 4960 s). (c) `n_seeds`: upstream runs once; three seeds under one budget cut the epochs three-fold and add a min-of-three selection. Run one seed with the whole budget, same for the 24 arms. (d) Record the empty-schema fraction per row beside gate 4, and count a `check_not_degenerate` drop as a row, not a hole. | a branch; `test_vendor_net_contract.py` and the repairer tests green on a hyphenated domain |
| Wed | **B4. Pilot, blocksworld small, one GPU node**, all four pixel arms, 300 s, `n_seeds` equal for 24 and 26. Read off: epochs reached vs the paper table, `learning_time_seconds`, VRAM and RSS, solve exit statuses per domain for both MILP arms, `n_gt_goals`. Then depot at native resolution to measure the 12× tensor. | the resource table for the sbatch templates (plan §4.3) |
| Thu | **B5. Templates.** `scripts/cluster_image/`: `make_manifest.py`, `pixel_fold.sbatch` (GPU), `symbolic_fold.sbatch` (cpu256), reusing `pack_fold.sh`; separate out roots per family (`one-job-per-output-cell`). | ready to submit when A6 lands |

### Track C: our arms, measurement, presentation

| day | step | output |
|---|---|---|
| Sun–Mon | **C1. Realised corruption per image fold** (plan §3.3): `realised_mask_rate`, `realised_flip_rate`, `n_atoms`, flip persistence across frames, frame-0 rate vs frames 1..T, written to `fold_info.json` from `gt_trajectories/` via `realised_rates.py`. | the bridge to the simulated grid |
| Mon | **C2. Dashboard reads it.** The image section derives its noise number from a simulated `noise_ratio` (`build_dashboard.py`); switch it to the recorded rates. Register the labels the week will produce: `__res=native` (or the chosen per-domain tag), `__mode=converge`, `__schema=none` on the 26 MILP arm, `__anchor=init` if C3 lands. `image.large.experiment_dir` is empty today; leave it until the large run. | config + one reader change |
| Tue | **C3. Parity variant, cheap version** (plan §3.4.2): `ROSAME-I_24__anchor=init`, the `gamma` anchor moved from the last to the first frame (`rosame_i_runner.py:320`), recorded in `algorithm_specific`. One knob, one label. The `vlm_final` anchor and NPI-SAM with both endpoints fixed wait for week 2. | the row that has the same information as NPI-SAM anchored |
| Thu | **C4. Fresh small image run** on the A6 corpora: our arms via `benchmark_runner` (new `run_name` per domain, `learning_timeout_seconds: 300`, not a backfill into `TO=600__*`), two-stage ROSAME arms via `backfill_baseline --out-root`, pixel arms via B5 on the GPU. Repoint `dashboard_config.yaml`'s `image.small.experiment_dir`. | the first comparable image grid |

C4 needs A6 and B5; everything else in track C is independent of A and B.

---

## 3. What to watch, per concern

### 3.1 Our arms' correctness on image data

- NPI-SAM and PI-SAM read the VLM trajectories exactly as the simulated ones;
  no image-specific code. The two things that can go wrong are in the data
  prep, not the learner: (i) the frozen observations' state 0 must be the GT
  state and the staged `__s0=noisy` copy the VLM's own frame 0 (never a
  redrawn simulated state, which is what `leading_state_corruption.py` does
  for symbolic cells); (ii) `frame_axiom_mode: after_gt_only`.
- `normalize.py` rewrites hyphens to underscores for image corpora; the GT
  lookup for realised rates and for the two-stage MILP's `n_gt_goals` is by
  problem name. The small symbolic corpora once lost every final state to a
  name-lookup miss (`rosame_milp_runner.py:190`, fixed since); check
  `n_gt_goals == num_trajectories` on image cells in the pilot anyway.
- Read `net_cost` / `net_fluent_patch_count`, never `cost`, when comparing
  repair magnitudes (`patch_accounting.py`).

### 3.2 The 26 arms' correctness

- **The UNSAT is our bug, confined to the 26 adapter** (section 3.2b): the
  26 runner canonicalises the init and goal fluents to underscores while the
  solver's domain keeps the PDDL hyphens, and the converter drops an unresolved
  fluent without an error. Hanoi loses all 25 anchor fluents, gripper loses
  `at-robby`; with an empty anchored state and `pre >= 1` the first action has
  no true precondition, hence UNSAT. Respelled, the upstream rule set is
  OPTIMAL in 0.3 s and the GT model is feasible (verified on
  `fold0_numtrajs3` of both domains, 2026-11-07). `rosame-i-26-failure-suggestions.md`
  §2 and the 26 plan's §8.1 are artefacts of this and are retracted. The 24
  MILP arms are not affected: the pixel one conforms its anchors to the
  reference dialect and the symbolic one has a `normalize_identifiers` policy;
  their image rows solved OPTIMAL on every round, and their v4 large-corpus
  rows on hanoi, gripper and depot are OPTIMAL or FEASIBLE with every goal
  anchored. Depot's predicates are hyphenated too, so its 89% UNSAT has the
  same cause.
- **Undertraining is the second, independent defect**; the "5000 epochs still
  UNSAT" test in the failure doc was run with the broken anchors and says
  nothing. A GPU changes how many epochs fit 300 s;
  record `epochs_reached` and the stop reason on every 26 row, and keep one
  `__ep=5000` control per domain outside the budget (gate 7) so "underperforms
  at the budget" and "undertrained at it" stay separable.
- **Extra supervision.** The 26 network anchors both endpoints at γ; the 24
  network anchors the final state only. State it beside every 24-vs-26 number
  (failure doc §3a). C3's `__anchor=init` row is the 24 arm with the same
  single clean state as NPI-SAM.
- **Same `n_seeds`, same GPU type, same `resize` per domain** for the 24 and
  26 arms, or the comparison is between settings, not architectures.
- The 26 trainer's `evaluate` is not called (it would reach
  `model_permutation`); emission goes through `rosame26_emitter.py` in the
  reference argument order. Keep `check_not_degenerate` and add the
  empty-schema fraction (B3c).

### 3.2a The authors' own recipes, and why "the paper's epochs" is not one number

Read from the two papers on 2026-11-07 (`thesis_related_work/icaps24-rosame.pdf`,
`...Visual Traces-icaps26.pdf`) and from upstream `train_common.py`:

| | ICAPS-24 (ROSAME-I) | ICAPS-26 (ROSAME-I + MILP) |
|---|---|---|
| training traces per domain | 100 blocks / 70 hanoi / 300 8-puzzle (synthesized); 800 to 2500 (grid) | 1,800 (2,000 split 90:10) |
| trace length (steps) | 5 to 10 | 3 to 9 |
| epochs | **100 / 70 / 300**, i.e. one epoch per trace | **5,000** |
| optimizer, lr, batch | Adam, 1e-3, per trace | Adam, 1e-4, batch 128 |
| λ, γ | 0.2 (0.4 on 8-puzzle), 10 | **paper 0.4, code 0.2**; γ 10 |
| MILP | none | after 50 DL-only epochs, every epoch, 3 to 4 traces, Gurobi, time limit 60 / 90 / 150 s by domain, ψ 0.99 |
| actions | observed | **unobserved** (predicted) |
| supervision | final state (γ) | init and goal (γ), plus MILP hard anchors |
| augmentation | image variations of the same state | same |
| hardware, cost | A100 40 GB | RTX 4090; 1.5 to 14.5 s per epoch, so 2 to 20 GPU-hours per model |

Consequences for dilemma 4:

- An "epoch" is a pass over 70 to 1,800 traces upstream and over 3 to 8 (small)
  or 10 to 2,000 (large) traces here. The paper's epoch count is not a
  transferable unit; what transfers is optimizer steps or wall-clock. Record
  both per row, and say in the paper which one the budget caps.
- The 24 recipe fits a 300 s GPU budget easily (its table is already our
  `_HYPERPARAMS`). The 26 recipe is 2 to 20 GPU-hours per model and the
  per-domain MILP limit is 60 to 150 s per solve, every epoch, for 4,950 epochs.
  No cell budget of ours contains it. So the grid runs the 26 arms under the
  budget with `converge`, and **one unbudgeted paper-recipe control per domain**
  (5,000 epochs, their time limits) says what the architecture can do when
  given what its authors gave it. Without that control, a weak 26 row is
  "undertrained at our budget", not "the 26 method".
- λ for the 26 arm: the paper says 0.4, the released code 0.2, and our port pins
  0.2 (`rosame26_training.py:89`). Pick one, say which, and keep the other as a
  one-knob check on one domain.
- Upstream's own setting predicts actions; ours observes them (the "option B"
  deviation in the 26 plan §8.1). That is the single largest difference between
  our 26 rows and the paper's, and it favours the baseline; it is stated, not
  removed, because every arm in our grid observes actions.

### 3.2b Fidelity audit of the 26 port against upstream (2026-11-07)

Two read-only comparisons against the authors' code at commit `95c733f`
(their `ROSAME+MILP` branch, since renamed `paper/icaps26`; it is the commit
`src/milp/vendor/` was taken from, so we hold their final code). The vendored
tree differs from upstream in exactly the nine edits `UPSTREAM.md` lists; the
network (`model.py`, encoder-decoder, action head, layers, dataset) is theirs
byte for byte, and the CP-SAT constraint set and objective in `encoder.py`
are line-for-line those of their `cp_sat.py`. The answer to "take their code
instead" is **no, and not needed**: their trainer needs homogeneous trace
lengths and at least ~20 traces (a 3-to-9-trace fold gives an empty
validation split and crashes), reads `max_t` from the first trace, zips the
head rows positionally onto the predicate arguments (wrong on four of five
domains), needs Gurobi, tensorboard and `lifted_pddl`, and runs unobserved
actions. The fidelity gap is not in the vendored parts.

What the audit found, by fairness impact:

| finding | where | verdict | favours |
|---|---|---|---|
| anchors dropped by predicate spelling (the UNSAT) | `rosame26_milp_runner.py:165-216`, `converter.py:205-283` | **bug** | us: the baseline's MILP was disabled on 3 domains |
| epochs vs steps: a fold ≤128 traces means one Adam step per "epoch"; 600 budget-lowered epochs against upstream's 5000 × ⌈N/128⌉ | `rosame26_runner.py:52`, `rosame26_budget.py:71` | documented, but the unit is not | us |
| three seeds share one budget, lowest-loss kept; upstream runs once | `rosame26_runner.py:558` | documented | mixed |
| `converge` mode emits the best-loss checkpoint; upstream emits the last epoch (its `best_model.pth` is saved and never loaded) | `rosame26_runner.py:494-515` | not in the deviation register | baseline |
| the plateau early stop, gate-4 seed dropping, `pre_mip_epoch = min(50, epochs-1)` | runner | not in the register | baseline / unknown |
| no wall-clock kill inside the loop | `rosame26_training.py` | documented gap | unknown |
| `check_not_degenerate` drops a row rather than scoring it | `rosame26_emitter.py:258-278` | documented | baseline (censors its worst cells) |
| per-domain image augmentation not ported | `network.py:218-231` upstream | documented | us, vs the paper's numbers |
| observed actions instead of predicted; `model_permutation` bypassed | `rosame26_model.py:113` | documented, the main design deviation | baseline |
| the docs say `loss_pseudo_a` was retargeted to the observed action; the code self-distils exactly as upstream | `rosame26_model.py:271` | doc error, code faithful | neutral |
| the 26 plan says one `Instance` per run; the code grounds per problem | `rosame26_milp_runner.py:173` | doc error; solutions equivalent | neutral |
| CP-SAT with 1 worker (cpmpy default 8); matters only at the 60 s cap | `encoder.py:77` | undocumented | us at the cap |
| normalisation std floored at 1e-6 vs 1e-20; endpoint frames included | `rosame26_data.py:64` | documented | mild, baseline |
| hyperparameters, loss terms, ψ decay, MILP cadence, trace selection (FIFO 3), pseudo-label paths, emission argmax | throughout | same | — |

Actions, in order: B3(a) the spelling fix with a raising converter and a
hyphenated-domain test; B3(c) seeds; B3(b) the in-loop clock; report Adam
steps and epochs reached per row and run the paper-recipe control (3.2a);
update the 26 plan's deviation register with the four undocumented knobs and
the two doc errors; retract the failure doc's §2; re-run every
`ROSAME-I_MILP_26` cell on gripper, hanoi and depot after the fix.

### 3.3 Data: where, how, and the GT

- **GT is already handled.** Both generation paths write
  `gt_trajectories/<problem>/` (true `.trajectory` + `_trajectory.json`) before
  the noisy rebuild: `_run_generation_inference` for PDDLGym, the visualizer's
  `trajectory_writer.py` output copied through for depot/gripper. The object
  universe is GT by construction (`inference_mode=False` in the detector).
  Nothing to add beyond the two artifacts of A1 and the generation metadata.
- **Where.** gpt runs locally against the API as before (the cluster has no
  outbound API access guarantee and the rate limit is the account's, not the
  node's). Qwen runs on a GPU node through vLLM; A2 lets the same
  `data_generator.py` drive it with `base_url`.
- **Reproducibility of the corpus.** Pin the model snapshot, temperature 0,
  the prompt hash, the example problem id, the seed, and the per-call usage
  log in `generation_info.json`. A corpus made with a different model or
  prompt protocol is a different corpus and gets its own dir name
  (`...__model=<m>__fs=shared`).
- **Large corpora.** Only after A5 and A7: render the symbolic large corpora
  (same trajectories as the symbolic experiment, every image row gets a
  symbolic twin) rather than walking again. Disk is file-count bound, not
  byte bound (plan §4.4): tarball per domain, unpack to the job's local SSD.

### 3.4 Budgets and stopping

- 300 s small / 3600 s large for every arm, wall-clock, as the symbolic runs.
  Pixel arms on one GPU type, everything else on cpu256; the hardware
  asymmetry is inherent and is stated once.
- 24 arms: the paper's epoch table under the cap; if the GPU finishes the
  table inside 300 s the budget is not binding and the row is at the paper's
  setting. 26 arms: `converge` (best-loss checkpoint) with a ceiling.
- Record `nvidia-smi -L`, `cpu_model`, `device`, `epochs_reached`,
  `stop_reason` in every pixel row.

### 3.5 Presenting the results

- The dashboard's image section already has one `experiment_dir` per domain
  and the 24/26 series registered (`ROSAME-I_24`, `ROSAME-I_MILP_24`,
  `ROSAME-I_26`, `ROSAME-I_MILP_26`, `__ep=5000`, `__res=224`, `__res=64x64`).
  A new suffix that is not registered does not render, so register before the
  first run, not after.
- Headline rows, by clean states per trajectory (plan §3.4): row "1 (init)"
  NPI-SAM vs `ROSAME-I_24__anchor=init` (upstream goal-anchored ROSAME-I
  beside it); row "2" the MILP pair; row "0" both sides with no GT. The
  two-stage `ROSAME_24` / `ROSAME_MILP_24` rows on the VLM output price the
  prompt separately from the learner.
- The realised mask / flip rates per domain are the sentence that places each
  image cell on the simulated grid; show them in the image section's header,
  not only in `fold_info.json`.
- No row from a mapped arm's variant goes into the paper or the thesis
  without asking (`paper-uses-mapped-arms-only`).

---

## 4. Thursday checkpoint: what "done" looks like

1. Realised rates for gpt-5.2 and Qwen on the same 25 problems, both
   protocols, with cost and throughput; the model and corpus size decided.
2. Five regenerated small corpora under the shared-example protocol.
3. The 26 arms producing rows whose MILP actually solves, under `converge`
   with the cap enforced, on a pinned GPU type; the pilot's resource table.
4. The cluster templates for both families, dry-run clean.
5. The fresh small image run submitted, or running locally for one domain if
   the GPU queue is slow; the dashboard repointed once it lands.

Week 2 then is: the remaining parity rows, the renderer in production, the
large corpora, and the fluent-classification variants (zero-shot; the init
image as the example).

---

## 5. Pre-check results, 2026-11-08 (A3 + A4, run on the cluster)

Both models classified the **same frames**: five fresh PDDLGym problems per gym
domain (`benchmark/data/<domain>/precheck_frames__seed7/`, `--gen-mode generate
--no-inference --seed 7`) and the first five of `src/domains/{depot,gripper}/problems`.
Driver: `python -m benchmark.vlm_precheck`; cluster job
`scripts/cluster_image/precheck.sbatch` (one CPU task per domain, the model
remote); Qwen served by `scripts/cluster_image/serve_vlm.sbatch` on one
rtx_4090 (`vllm 0.33.0`, `Qwen/Qwen3-VL-8B-Instruct` bf16, 12k context — a
24 GB card leaves 1.8 GB of KV cache after the 8B weights). Protocol: the
existing one-shot (frame 0 of the same problem with its GT facts), one call
in flight per task, five tasks sharing the one GPU.

| domain | model | frames | s/frame | tokens/frame | cache hit | mask rate | flip rate |
|---|---|---|---|---|---|---|---|
| blocksworld | gpt-5.2 | 73 | 3.1 | 1109 | 0.00 | 0.007 | 0.012 |
| blocksworld | qwen3-vl-8b | 73 | 1.2 | 1216 | 0.00 | 0.609 | 0.151 |
| depot | gpt-5.2 | 47 | 6.2 | 1983 | 0.16 | 0.000 | 0.016 |
| depot | qwen3-vl-8b | 47 | 4.3 | 1818 | 0.00 | 0.478 | 0.119 |
| gripper | gpt-5.2 | 65 | 4.3 | 1828 | 0.13 | 0.049 | 0.040 |
| gripper | qwen3-vl-8b | 65 | 2.4 | 1758 | 0.00 | 0.629 | 0.138 |
| hanoi | gpt-5.2 | 73 | 4.8 | 2084 | 0.66 | 0.162 | 0.020 |
| hanoi | qwen3-vl-8b | 73 | 5.1 | 1995 | 0.00 | 0.401 | 0.093 |
| npuzzle | gpt-5.2 | 73 | 29.1 | 1310 | 0.00 | 0.050 | 0.002 |
| npuzzle | qwen3-vl-8b | 73 | 27.3 | 1391 | 0.00 | 0.700 | 0.129 |

The existing small corpora (gpt-5.2, same protocol, ten problems) for scale:
blocksworld 0.000 / 0.016, depot 0.006 / 0.033, gripper 0.003 / 0.024,
hanoi 0.134 / 0.052, npuzzle 0.452 / 0.000 (mask / flip).

What the numbers say:

- **Qwen3-VL-8B is not a drop-in replacement under this protocol.** Mask
  40–70 %, flip 9–15 %, against gpt-5.2's 0–16 % and 0.2–4 %. Per frame it is
  about twice as fast on the small domains and no faster on hanoi/npuzzle,
  with five tasks sharing one card.
- **Half of Qwen's masking is a format failure, the rest is vision.** The
  rejected replies are bare fact lists copied from the example turn's format
  (`ontable(red:block) on(cyan:block,red:block) ...`, no `: <score>`), which
  the parser cannot read, so the whole frame is UNCERTAIN. A probe that
  rewrites the example turn as a complete scored list
  (`scripts/cluster_image/probe_scored_example.py`) brings the mask rate to
  0–12 % on depot/gripper but leaves the flip rate at 15 %. The model copies
  the example rather than reading the frame.
- **The in-context example contradicts the system prompt for every model:**
  the prompt asks for a 0/1/2 score on every grounded predicate, the example
  turn lists only the true facts. gpt-5.2 follows the prompt; an 8B model
  follows the example. A1's shared-example protocol should make the example
  turn a complete scored answer.
- **npuzzle under gpt-5.2 is a different corpus today** than the one on disk:
  mask 0.05 vs 0.45, and 29 s/frame. The time is output length, not
  reasoning: a measured npuzzle call returns 2,700 completion tokens and 0
  reasoning tokens (blocksworld: 234 and 0). gpt-5.2 ran with **no
  reasoning** in every run here: the backend sends no `reasoning_effort`,
  and the model's default is none (snapshot `gpt-5.2-2025-12-11`).
  Regenerated small corpora will not reproduce the old npuzzle profile.
- **Prompt caching mostly misses:** two domains never reach the 1024-token
  cacheable prefix, only hanoi exceeds half. A1's usage log matters.
- **Cluster facts (B1):** compute nodes reach api.openai.com and huggingface.co;
  rtx_4090 is the GPU to pin (30 nodes × 3, four idle, the ICAPS-26 card);
  `~/.config/pip/pip.conf` carries an unresolvable NVIDIA index that stalls
  every pip install unless `PIP_CONFIG_FILE=/dev/null`. The image-track
  checkout on the cluster is `~/projects/VIP-image` (rsync of the working
  tree, not a git clone); `vip_venv11` runs the pipeline as is.
- **A pipeline fix fell out of it:** a frame where nothing parsed is written as
  `(:state )`, and `trajectory_json_converter._parse_trajectory_file` dropped
  such states (its regex required one literal), so the problem failed with an
  action/state mismatch. Fixed, with a test; `llm_fluent_classifier` now logs
  an excerpt of a rejected reply.

Open for A5 (Wednesday): if an open model is still wanted, the next candidate
is a larger Qwen3-VL (32B on an rtx_6000 48 GB, or a 72B-class model on an
rtx_pro_6000 96 GB), with the scored-example protocol; the 8B result says the
one-shot protocol as it stands is the wrong test for small open models.

### 5.1 Protocol fixes landed on 2026-11-10 (A1 + the scored example)

- **Scored example.** `LLMFluentClassifier._scored_examples` renders the
  one-shot example's assistant turn as the full predicate universe (`: 2` for
  the frame's true facts, `: 0` for every other grounded predicate) at send
  time; `fewshot_examples` still stores bare facts. On Qwen3.8-27B this took
  the mask rate from 0.17 to 0.00 on blocksworld/depot problem0 with the flip
  rate unchanged; gpt-5.2 is unaffected.
- **Held-out example (A1).** `domains.<d>.example_dir` in `config.yaml` names
  `src/domains/<d>/example/` (one frame, its `*_trajectory.json`, its problem):
  blocksworld/hanoi/npuzzle from one generated problem at seed 1000,
  depot/gripper from problem9 (so their corpora run with `--problem-end 8`).
  With it set, the handler classifies frame 0 like every other frame, so
  state 0 is inferred; the GT trajectory beside it still carries the true
  state 0 for the "1 clean state" rows via GT injection. Output names carry
  `__fs=shared`. Unset, the old protocol (own frame 0, not classified) is
  reproduced exactly. Checked on all five domains with gpt-5.2: hanoi and
  gripper state 0 now differ from GT, as an inferred state should.
- Both are in the working tree, uncommitted; the cluster checkout
  `~/projects/VIP-image` has them.
