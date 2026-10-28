# Meeting brief: fairness of the imaged comparison, VLM choice, open points

For the 2026-10-29 meeting with Roni. Companion to
`docs/image-experiments-plan.md` (the plan; section numbers below refer to it).
Everything marked "code" was checked in the repo on 2026-10-28.

Roni's framing (WhatsApp, 2026-10-27): run a small sample first, see the
results, then plan the full run. The plan's pre-check (plan §6.3) is that
sample; section 4 below says what to decide before it runs.

---

## 1. Where the imaged comparison is not the symbolic one

In the symbolic experiments every arm read the same degraded trajectories and
the only information knob was the GT state column (`init` / `none`). On images
there are two pipelines (ours: VLM → symbolic trajectory → learner; ROSAME-I:
pixels → CV head + learner), and they are told different things. The gaps,
who each favours, and the proposed handling:

| # | gap | favours | proposal | needs a decision? |
|---|---|---|---|---|
| G1 | **GT states per trajectory.** ROSAME-I needs one labelled frame for its CV head (upstream: the final state); ROSAME-I+MILP and both 26 arms fix init **and** final; ours ran with init only and with none. | ROSAME-I family | Compare by *clean states per trajectory* (plan §3.4): add `ROSAME-I_24__anchor=init` (same information as NPI-SAM anchored), `ROSAME-I_24__anchor=vlm` (no GT: the anchor is the VLM's labelling of the last frame), and NPI-SAM with init and final fixed (beside ROSAME-I+MILP). Code: anchor index at `rosame_i_runner.py:320`; `anchor_endpoints` + `GtAnchoring.ALL_GT_STATES` exist but are not wired for the MILP phase. | yes: which rows are headline |
| G2 | **Object universe.** Both pipelines get the GT object list: the "LLM detector" never calls the model (`inference_mode=False`, `llm_object_detector.py:95`), and the pixel arms take the objects from the problem file. | neither | Keep, and say so in one sentence ("object identity is given; the learners recover fluents and dynamics"). ROSAME-I also needs *one* grounding per corpus (identical-grounding assert); the small blocksworld corpus has 4- and 5-block problems, but the adapter grounds the union and no trajectory was dropped (checked: `n_traces == num_trajectories` in all 300 rows). Large corpora are uniform by construction. | no |
| G3 | **Domain knowledge in the prompt.** Our VLM gets a hand-written system prompt that explains each predicate's visual meaning plus one labelled example problem; ROSAME-I gets the predicate vocabulary and action signatures only, and learns the pixels→predicate map from its anchor. | ours | This is the two-stage-vs-end-to-end question itself, not a confound to remove. Two things make it fair to report: (a) the two-stage `ROSAME_24` / `ROSAME_MILP_24` rows (ROSAME on *our* VLM output) separate "the VLM" from "the learner": if ROSAME on VLM output beats ROSAME-I, the prompt is doing the work; if NPI-SAM beats ROSAME on the same VLM output, the learner is. (b) Optional equaliser for the pixel arm: ImageNet-pretrained ResNet-18 instead of random init (`weights=None` today, one flag). | yes: (b) in or out |
| G4 | **Input resolution.** Pixel arms see 64 px (shorter edge); the VLM sees the native frame (depot/gripper 800×600, labels rendered as text). At 64 px depot's object identity is unreadable (`algorithm_comparison_analysis.md` §5.1: 40 px text → 4.3 px). | ours, on depot/gripper | Run the pixel arms at native resolution (or 224) on depot and gripper, label `__res=`, and report both. Cost: ~12× the pixel tensor, GPU makes it affordable; memory re-measured in the pilot. | yes: 64 everywhere vs per-domain |
| G5 | **Budget and hardware.** Equal wall-clock per arm (300 s small / 3600 s large), but pixel arms on a GPU node and the rest on cpu256. ROSAME-I spends its budget on epochs (3 seeds share it); ours on MILP solves. | depends | Inherent; state once. Pin one GPU type and one CPU type; record both in the row. The seeds already share the budget (`timeout_check` skips remaining seeds). | no |
| G6 | **Training schedule.** Symbolic large ROSAME ran batch 128, loss-normalised, to a convergence rule with a 2000-epoch ceiling. ROSAME-I has no `epochs` override (per-domain table is a hard ceiling: 100/70/300/100/100) and no batch knob (one step per trace). The CPU rows on disk hit the wall in blocksworld and npuzzle. | ours (undertrained baseline) | Add the `epochs` override and the convergence rule for the large run (plan §5.5); GPU for every pixel row (plan §0.4). | no |
| G7 | **MILP knobs.** ROSAME-I+MILP: 3 traces per solve, 60 s cap budgeted down, interval doubling; ours: `half` (small) / 4 (large) traces per subset, 60 s cap. Encoding rule-sets differ (upstream dialect vs ours) except `schema_nonempty`, aligned 2026-10-08. | roughly even | Set `mip_traces` 4 on the large corpus as the symbolic run did; keep the dialect difference documented. | no |
| G8 | **Frame 0.** Today state 0 of every image trajectory is GT (plan §0.2), so our `gt=none` image rows on disk are not no-GT rows. | ours, if reported as is | Classify frame 0 with a held-out shared example (plan §3.1); frozen = init-GT copy, staged = VLM-state-0 copy, as on the symbolic side. | yes: present a no-GT column for images? |
| G9 | **Masking.** Our arms exploit the VLM's `UNCERTAIN` answers as masks; ROSAME-I has no such channel from pixels. | ours | Not a confound: it is what the two-stage design buys. Report the realised mask rate so a reader sees how much was masked. | no |
| G10 | **Trajectory distribution, depot/gripper.** The small depot/gripper corpora are *planner* traces (visualizer plans); rendering the symbolic large corpora gives *random-walk* windows. Small and large image corpora would then differ in distribution for two domains, and blocksworld/hanoi/npuzzle small corpora are random walks already. | neither, but a confound across sizes | Either regenerate the small depot/gripper corpora as random-walk windows too (10 problems, cheap) or accept and footnote. | yes |
| G11 | **ICAPS-26 arms.** The MILP is UNSAT on every gripper/hanoi solve and 89% of depot's; the DL-only models have ~2% effect recall on three domains (`rosame-i-26-failure-suggestions.md`). | ours, if reported as MILP rows | Keep / DL-only with footnote / appendix (plan §2.4). If kept, `budget_mode converge` and per-domain solve usability in the table. | yes |
| G12 | **Seeds.** ROSAME-I keeps the best of 3 seeds by training loss; NPI-SAM is one run. | ROSAME-I slightly | Fine (training-loss selection touches no test data) but report it; use the same `n_seeds` for the 24 and 26 arms. | no |

The headline recommendation: present the image results as the table of plan
§3.4 (rows = clean states per trajectory), with the two-stage ROSAME rows in
every row so the VLM's contribution is separable from the learner's, and say
G2/G3/G5/G9 once in the setup paragraph.

---

## 1b. The experiment matrix

Same shape as the symbolic `paper_arms` grid: one row per arm, grouped by
the clean states each training trajectory carries. "frozen" = the corpus's
init-GT copy (state 0 replaced by GT, as today); "staged" = the VLM-state-0
copy (plan §3.1). Every row runs on both corpora unless the "corpus" column
says otherwise.

### Shared constraints (apply to every row)

| | small corpus | large corpus |
|---|---|---|
| data | 10 problems per domain, regenerated under the shared-example protocol (plan §3.1); montecarlo 80/20, 5 folds | the symbolic `large_corpus*` windows rendered to frames (plan §6.1), 500 → 2500 problems; kfold, 5 folds |
| training sizes L | 3, 4, 5, 6, 7, 8 | 10, 50, 100, 500 (2000 once the pool is 2500) |
| learning budget per arm | 300 s wall-clock | 3600 s |
| planning timeout | 60 s | 60 s |
| hardware | pixel arms: one GPU type; all else: cpu256 | same |
| MILP cap, both families | 60 s per solve (`__solve=60` on ours; bare on ROSAME) | same |
| MILP traces per solve | NPI-SAM `half`; ROSAME-I+MILP 3 | NPI-SAM 4 (`__m=4`); ROSAME-I+MILP 4 (align, as the symbolic run did) |
| NPI-SAM encoding | `schema_nonempty: pre_and_add` (`__schema=pre_and_add`), `eq16` off, frozen pool, random sampler, `w_prior` tiebreak | same, `stop.no_improvement_rounds: auto` |
| pixel resolution | 64 px shorter edge (default); see the resolution rows | same |
| pixel seeds / epochs | 3 seeds sharing the budget; per-domain epoch table (100/70/300/100/100) | 1 seed; convergence rule (window 40 / 0.002 / patience 3 / min 50) with a 2000-epoch ceiling (needs the `epochs` override, plan §5.5) |
| symbolic ROSAME | batch 0, no base-loss normalisation, convergence off, seed 8800 | batch 128, normalised, convergence rule, `epochs` 2000, `mip_traces` 4 |
| 26 arms | `budget_mode converge`, 3 seeds (small) / 1 (large), `mip_time_limit` 60, `mip_traces` 3 → 4 on large | same |
| recorded per row | realised mask and flip rate, flip persistence, device, `cpu_model`, GPU type, `n_gt_goals`, anchor, `gt_anchoring` | same |

### Arms × information: the grid

Rows are arms, columns the clean states a training trajectory carries. Each
cell gives the row label and how the GT enters: **D** = in the data copy
(the trajectory file has the GT content at that index and tags it in
`gt_indices`), **L** = fixed or anchored inside the learner (the MILP's
`gt_anchoring` / `goal_mode`, or ROSAME-I's loss anchor). Both channels must
agree on every row; the pilot checks tags, data content and `gt_anchoring`
together. Data copies: *staged* = VLM state 0 (plan §3.1); *frozen* = GT
state 0; *+sT* = GT final state injected and tagged
(`inject_gt_states_by_percentage(..., anchor_final=True)` replaces exactly the
tagged states). Symbolic ROSAME has no GT input of its own: information
reaches it through the data copy only.

| arm | 0 (staged) | 1, init (frozen) | 1, final (staged + sT, new copy) | 2, init + final (frozen + sT) |
|---|---|---|---|---|
| `PISAM` (ours, no MILP) | `PISAM__s0=noisy` · D none | `PISAM` · D s0 | `PISAM__gt=final__s0=noisy` · D sT | `PISAM__gt=init_final` · D s0 + sT |
| `PISAM_MILP_LOOP` (NPI-SAM) | `…__gt=none__s0=noisy…` · D none, L none | `…__schema=pre_and_add__solve=60[__m=4]` · D s0, L `init_only` · **paper arm** | `…__gt=final__s0=noisy…` · D sT, L `all_gt_states` with tags {T} (new wiring) | `…__gt=init_final…` · D s0 + sT, L `all_gt_states` with tags {0, T} (new wiring) |
| `ROSAME_24` (two-stage) | `ROSAME_24__s0=noisy` · D none | `ROSAME_24` · D s0 | `ROSAME_24__gt=final__s0=noisy` · D sT | `ROSAME_24__gt=init_final` · D s0 + sT |
| `ROSAME_MILP_24` [+`_TAG`] | `…__gt=none__s0=noisy` · L none, goal none | `…__goal=none` · L init fixed, final free · paper arm | `…__gt=none__goal=gt__s0=noisy` · L final fixed only | `ROSAME_MILP_24` · L init + final fixed (exists) |
| `ROSAME-I_24` (pixel) | `…__anchor=vlm` · L anchor = VLM labelling of the last frame, masked atoms excluded (new) | `…__anchor=init` · L anchor = GT s0 (new, one line) | `ROSAME-I_24` · L anchor = GT sT · **upstream** | `…__anchor=both` · L two anchor terms (new) |
| `ROSAME-I_MILP_24` (pixel) | `…__anchor=vlm__gt=none__goal=none` (new) | `…__anchor=init__goal=none` · L init in loss and MILP (new) | `…__gt=none` · L sT in loss and MILP, init free (new flag) | `ROSAME-I_MILP_24` · upstream (exists) |
| `ROSAME-I_26`, `ROSAME-I_MILP_26` | — | — | — | upstream only; keep / footnote / appendix (decision) |

Label note: the encoder's value is `all_gt_states` ("every *tagged* state",
not every state); the row label should read `gt=init_final` or `gt=final`
so nobody reads it as full observability.

Rows that exist today (symbolic or image): the paper arms, `ROSAME_MILP_24`
both-fixed, `ROSAME-I_24` and `ROSAME-I_MILP_24` upstream, the 26 arms. Every
other cell is a data variant plus a label, except the ROSAME-I anchor knob
(index and a two-term variant) and `--milp-gt-anchoring none` on the imaged
MILP runner.

Headline rows for the paper: column "1, init" (every arm at our setting),
column "2" (the MILP pair at upstream's setting), column "0" (no GT at all).
Column "1, final" is ROSAME-I's home setting with every other arm brought to
it; run it on the small corpus at least, on the large corpus if budget allows.

### Ablations (not in the headline table)

| arm | what it varies | corpus | why |
|---|---|---|---|
| `ROSAME-I_24__res=224` (or native) on depot, gripper | pixel resolution | small first | G4: object labels are unreadable at 64 px; the VLM sees 800×600 |
| `ROSAME-I_24__pretrained` | ImageNet-initialised ResNet-18 | small, one domain | G3(b): how much of the gap is "foundation model vs scratch" |
| `ROSAME-I_24__anchor=init` vs `ROSAME-I_24` | anchor position | both | shows whether upstream's goal anchor matters at all |
| VLM axis: a second model on the sample, then at most one small corpus | VLM quality | sample (5 problems × 5 domains), then small | section 2 |
| `PISAM_MILP_LOOP` `subset_size` on the image corpus | traces per solve | large, one domain | the one MILP knob the symbolic ablation never varied on real noise |

Headline figure/table rows: "1, init" (NPI-SAM vs `ROSAME-I_24__anchor=init`, with upstream `ROSAME-I_24` beside it), "2" for the MILP pair, "0" for both sides with no GT; the two-stage rows in every group.

---

## 2. Which VLM

The question is what the image experiment is for. Two readings:

- **Operating point.** The image corpus is one realistic point: "this is what a
  current perception pipeline produces, and this is how each learner copes."
  Then the model should be the one a practitioner would use today, the
  realised mask/flip rates place the point on the simulated grid (plan §3.3),
  and the simulated grid already covers the rest of the noise range.
- **Stress test.** The image corpus should discriminate between learners, so
  the VLM should be noisy enough that the no-repair arms fail and the repair
  arms do not.

Recommendation: **the operating point, with the model fixed to what the small
corpora used (`gpt-5.2`, pinned to its dated snapshot), plus one deliberately
weaker VLM on one or two domains as a second point.** Reasons:

1. If a frontier model makes classification near-perfect, that is a finding,
   not a failure: it says the symbolic grid's low-noise corner is where images
   live today, and the paper's claim moves to the grid. The pre-check tells
   us this for about $5 before any corpus is built (plan §6.3): realised
   rates per candidate model on 5 problems × 5 domains.
2. Picking an older model *because* it is noisier is the choice a reviewer
   will question ("you chose a weak VLM to need your repair step"). Picking
   the model the small corpora already used is consistency; adding a weaker
   one as an explicit axis ("VLM quality as the image analogue of the
   mask/noise axes") is a controlled comparison rather than a cherry-pick.
3. Cost (plan §6.2): output tokens dominate, so a frontier model mostly
   changes the per-token price, not the structure of the bill; a 2500-problem
   corpus is roughly $1k to $2k at `gpt-5.2` prices, less with the Batch API.
4. Determinism and availability: temperature 0, a dated snapshot, and a model
   that will still exist when a reviewer asks for a rerun. Frontier models
   churn fastest.

If Roni prefers the stress-test reading, the clean way is still the same
pre-check: run two or three candidate models on the sample, read their
realised rates, and choose the point on the grid on purpose, writing down why.

What the small sample can settle tonight with no new code: `data_generator.py
--gen-mode generate` with 3 problems per PDDLGym domain and `config.yaml`'s
`model_name` switched, once per candidate model, then `realised_rates` against
`gt_trajectories/`. The shared-example protocol and the usage log (plan §5.13,
§5.14) are needed for the cost numbers, not for the rate comparison.

---

## 3. Other points for tomorrow

1. **Corpus shape.** Render the existing symbolic large corpora rather than
   walk again (plan §6.1): same trajectories as the symbolic experiment, every
   image row gets a symbolic twin. Needs a state→frame renderer for PDDLGym
   domains and feeding the visualizer our windows for depot/gripper.
2. **Corpus size.** 500 / 1000 / 2500 problems per domain; the cost table
   (plan §6.2) and the disk table (plan §4.4: under 2 GB, but ~200k files).
   A prefix of the symbolic corpus means 500 now can grow to 2500 later with
   nothing wasted.
3. **Sample first (Roni's point).** The pre-check is the sample: 5 problems
   × 5 domains × 2 protocols, cost and latency measured per call, realised
   rates per model. Decide the model and the size from it, not before.
4. **Frame 0 protocol and regenerating the small corpora** under it, so small
   and large share one noise process (G8).
5. **Budgets and L list.** 300 s / 3600 s; L follows the pool size
   (plan §6.1).
6. **GPU type** for every pixel task, and whether `vip_venv11`'s torch is a
   CUDA build (plan §4.1).
7. **Where the image results go.** The AAMAS draft says images are future
   work; is this the thesis chapter, a follow-up paper, or both? That decides
   how many arms and how much ablation (e.g. G3(b), G4) we actually run.
8. **Reporting the realised corruption** (plan §3.3): mask and flip rates per
   domain, plus persistence of flips across frames, as the bridge to the
   simulated grid.
9. **What "fair" means in the paper's words.** One paragraph in the setup:
   same test problems and metrics for every arm; equal wall-clock per arm;
   object identity given to all; clean states per trajectory stated per row;
   the VLM prompt is part of our method, and ROSAME-on-VLM-output is there to
   price it.

---

## 4. What to do before the meeting / right after

- Tonight, optional, no code: the small sample with the current pipeline on
  two models, 3 problems per PDDLGym domain, to have realised rates in hand.
- After the meeting, in this order: §5.13 + §5.14 of the plan (shared
  example, frame 0, usage log) → pre-check proper → decide model and size →
  render corpora → pilot one domain on the cluster → arrays.
