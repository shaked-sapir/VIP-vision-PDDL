# CDPS-MILP P1–P5 — code change index

A per-file, per-line index of everything the CDPS-MILP work changed, for review.

This is deliberately **not** a narrative. `docs/CDPS-MILP-loop-PROCESS.md` (1219 lines)
is the reasoning log — why each decision was taken, what was measured, what was
withdrawn. `docs/cdps-milp-denoiser-design.md` is the spec. This file answers only
"what code moved, where, and what does it now do".

**Base commit:** `f86c60494` — everything below is `f86c60494..HEAD` on branch
`cdps-with-milp-implmenetation`. Line numbers are as of `1386a1629` (HEAD).

## Phase → commit map

| phase | commit | subject | net |
|---|---|---|---|
| P1+P2 | `4600b5b76` | Add `cdps_milp_single_round`: solve the CDPS repair problem as one CP-SAT program | +2441 / −252 |
| P3 | `b19fb8d69` | Add a ground-truth-free model score so the loop can select without cheating | +993 / −26 |
| P4 | `fa4c0fc1c` | Add `cdps_milp_loop`: pick a repair by a GT-free score across rounds | +2715 / −159 |
| — | `641cb97a3` | Dashboard: highlight one algorithm's trend on legend hover | +36 / −7 |
| P5.1 (D4) | `cdfa451bc` | Keep every loop round's model, not just the winner | +202 / −18 |
| P5.2 (D1) | `f2e609e42` | Backfill the MILP arms into existing cells, on their own input | +339 / −107 |
| P5.3 (D1) | `c0a8e95cc` | Measure the loop across n, withdraw the "run at 10–20 trajectories" advice | +85 / −1 (docs only) |
| P5.4 (D3) | `41c90aae8` | Give ROSAME per-epoch snapshots so it can appear on an anytime curve | +507 / −19 |
| P5.5 (D5/D7) | `2c0422252` | Add an offline anytime harness so the arms can be read against each other | +1024 / −3 |
| P5.6 (D6) | `84276d984` | Let one `cdps_milp` block name several MILP arms via per-knob ablations | +552 / −41 |
| P5.6a | `e53ac523e` | Normalise predicate type tags before CWA-completion, unblocking depot | +519 / −428 |
| P5.6b | `1060cd0f9` | Backfill: take the MILP work dir from the row label, not the algorithm key | +26 / −6 |
| P5.6c | `c1f164811` | Add the eq16=on config as a tracked file, not a `/tmp` throwaway | +14 |
| P5.7 | `ec263dd78` | Count patches by parity, and retire the bound they were feeding | +645 / −11 |
| P5.7 | `1386a1629` | Put `patch_accounting` on the module map | +1 / −1 |

---

# P1+P2 — the single-round MILP denoiser (`4600b5b76`)

The thesis claim being served: CDPS's repair problem has an exact formulation, so
CDPS's heuristic answer can be measured against a proven-optimal one.

## Module move — `benchmark/` → `src/`

The MILP encoder was living under `benchmark/algorithm_adapters/rosame_milp/`, but
`src/` may not import from `benchmark/` (CLAUDE.md). Once CDPS needed the encoder,
it had to move or the rule broke.

| from | to |
|---|---|
| `benchmark/algorithm_adapters/rosame_milp/encoder.py` | `src/pi_sam/plan_denoising/milp_version/encoder.py` |
| `benchmark/algorithm_adapters/rosame_milp/converter.py` | `src/pi_sam/plan_denoising/milp_version/converter.py` |
| `benchmark/algorithm_adapters/rosame_milp/encoding_config.py` | `src/pi_sam/plan_denoising/milp_version/encoding_config.py` |
| `benchmark/algorithm_adapters/rosame_milp/vendor/**` | `src/pi_sam/plan_denoising/milp_version/vendor/**` (0-line moves) |

`benchmark/algorithm_adapters/rosame_milp/__init__.py` (−24) keeps the ROSAME-MILP
baselines working off the relocated encoder.

## New files

| file | lines | what it is |
|---|---|---|
| `src/pi_sam/plan_denoising/milp_version/single_round.py` | 355 | the driver |
| `src/pi_sam/plan_denoising/milp_version/config.py` | 609 | `CdpsMilpConfig` + enums |
| `src/pi_sam/plan_denoising/milp_version/trajectory_extraction.py` | 192 | solved MILP → re-masked T′ |
| `src/pi_sam/plan_denoising/milp_version/encoding_config.py` | 173 | `MilpEncodingConfig` + presets |
| `src/pi_sam/plan_denoising/milp_version/test_cdps_milp.py` | 501 | 12 tests |

### `single_round.py` — the driver

| lines | symbol | logic |
|---|---|---|
| 68–110 | `SingleRoundResult` | result record; `is_conflict_free` (93), `as_report` (96) |
| 113–155 | `_build_traces` | masked observations → `planning_structs` traces |
| 156–172 | `_reindex_gt_states` | remaps GT indices after any trace is dropped |
| 173–196 | `_learn_with_pisam` | repaired T′ → PI-SAM, the step shared with CDPS |
| 197–265 | `save_artifacts` | writes CDPS's artifact shape + `milp_repair_log.json`, `milp_masked_completion.json` |
| 266–355 | `run_single_round` | encode → solve → extract → learn |

### `encoding_config.py` — the three dialects

`MilpEncodingConfig` (73) turns the encoder's behaviour into data so one encoder
serves both ROSAME-MILP and CDPS without branching on a caller flag.

| lines | preset | used by |
|---|---|---|
| 115–120 | `upstream()` | `rosame_milp`, `rosame_milp_base` — the paper's rules |
| 121–129 | `tag()` | `rosame_milp_tag` |
| 130–157 | `cdps_dialect()` | the CDPS arms |
| 103–114 | `__post_init__` | validation |
| 35–49 | `SchemaNonemptyRule` | enum, replaces a bool |
| 50–72 | `PriorWeightMode` | enum |

**The CDPS dialect is the key technical point** — see PROCESS §2.2. It is a
different *objective*, not a different solver: the upstream encoding trusts the
observation and fits a model to it; the CDPS dialect treats observed fluents as
soft and pays to flip them, which is what makes the two comparable at all.

### `trajectory_extraction.py` — reading the solution back out

| lines | symbol | logic |
|---|---|---|
| 34–71 | `FluentFlip` | one edit; `as_patch_dict` (51) emits CDPS's patch record shape so downstream tooling is shared |
| 72–94 | `ExtractionResult` | `repair_cost` (81) = number of flips |
| 106–163 | `extract_repaired_observations` | solved vars → new `Observation`s, **re-applying the original mask** so the learner sees the same observability CDPS saw |
| 164–177 | `_solved_value` | one variable's solved value |
| 178–192 | `save_extraction_artifacts` | |

### `config.py` (P1 portion) — `CdpsMilpConfig`

Validated on every run even when the arm is off, so a typo fails immediately
rather than three hours in. Enums at 31–97; the dataclass at 310.

## Modified — benchmark integration

| file | Δ | change |
|---|---|---|
| `benchmark/algorithms.py` | +50 | `CDPS_MILP_SINGLE_ROUND` key (50) + name (51); `resolve_algorithms` (248) returns a 4-tuple |
| `benchmark/experiment_running_helpers/learning_helpers.py` | +202 | `_learn_cdps_core` (94) extracted so CDPS and MILP share one path; `learn_cdps_milp_single_round` (330) |
| `benchmark/experiment_running_helpers/run_fold.py` | +133 | `run_cdps_phase` (184) gains `milp_config` (215): **one evaluation path, denoiser swapped** |
| `benchmark/benchmark_runner.py` | +14 | plumbs the `cdps_milp:` block |
| `benchmark/experiment_runner.py` | +19 | ditto |
| `benchmark/run_config.yaml` | +11 | the `cdps_milp:` block (36) |
| `benchmark/baselines/rosame_milp_runner.py` | +13 | import path after the move |

---

# P3 — the ground-truth-free score (`b19fb8d69`)

The loop must choose between candidate models *during* a run. Every existing
metric (`evaluate_model` in `benchmark/experiment_running_helpers/evaluation.py`)
needs the GT domain, so using one would make the loop cheat. P3 builds a score
that reads only the frozen original observations.

## New — `src/pi_sam/plan_denoising/evaluator.py` (541 lines)

V = w₁·(effect mismatches) + w₂·(inapplicability events), measured by replaying
each observation against the candidate model.

| lines | symbol | logic |
|---|---|---|
| 61–72 | `EvaluationWeights` | w₁, w₂ |
| 73–129 | `TraceEvaluation` | per-trace; `v_raw` (99), `v_per_transition` (107), `success_rate` (114), `one_step_success_rate` (121) |
| 130–268 | `EvaluationResult` | fold-level aggregate, 15 derived properties, `to_dict` (209) |
| 269–313 | `_fluent_key`, `_project_to_positive_state`, `_positive_fluent_keys` | CWA projection; `is_init` (275) is special-cased because the init state is GT |
| 314–351 | `_grade_state` | grades **only unmasked slots** — a masked slot is not evidence |
| 352–397 | `_build_operator`, `_precondition_fluent_keys`, `_effect_fluent_keys` | |
| 398–436 | `_evaluate_rollout` | apply-anyway: on an inapplicable action, **charge it and continue** rather than truncating, so one early error doesn't hide every later one (PROCESS §3.3) |
| 437–493 | `_evaluate_one_step` | re-seeds from the observed state each step, isolating per-transition error from drift |
| 494–541 | `observations_reconstruction_score` | public entry point |

Both rollout and one-step are computed because they fail differently: rollout
compounds, one-step does not.

**Exit gates, both passed before P4 was allowed to start** (PROCESS §4bis.3/§4bis.4):
V(GT model) equals the injected-noise count, and V correlates with the GT metrics.

## New — `src/pi_sam/plan_denoising/test_evaluator.py` (315 lines, 17 tests)

## Modified

| file | Δ | change |
|---|---|---|
| `CLAUDE.md` | 1 line | records that `evaluator.py` is GT-free and `evaluation.py` is not — the distinction is easy to get wrong |
| `run_fold.py` | 1 line | import |

---

# P4 — the loop (`fa4c0fc1c`)

One solve over every trace is brittle: a single unrepairable trace drags the
whole fold. The loop solves a *subset* per round, learns from it, scores with V,
and keeps the best — so selection never touches ground truth.

## New — `src/pi_sam/plan_denoising/milp_version/loop.py` (902 lines)

| lines | symbol | logic |
|---|---|---|
| 105–162 | `RoundLog` | one round's record |
| 163–223 | `LoopResult` | `is_conflict_free` (179), `as_report` (183) |
| 224–260 | `_canonical_action`, `model_hash` | **structural** hash. `LearnerDomain.to_pddl` rebuilds `:requirements` from a set, so its text is unstable across runs; identity must not key on it |
| 261–288 | `_subset_gt`, `_evaluate` | |
| 289–347 | `_TraceCache` | instance/trace reuse; `invalidate` (340) for non-frozen pools |
| 348–389 | `_sample_random`, `_sample_hardest_first` | the two samplers |
| 390–412 | `_per_trace_scores` | feeds `hardest_first` |
| 413–438 | `_stop_reason` | budget / no-improvement / perfect-fit / max-rounds / fixpoint |
| 439–451 | `_remaining_budget` | |
| 452–471 | `_solve_subset` | |
| 472–493 | `_learner_input` | `subset_only` vs `accumulated` |
| 494–618 | `run_loop` | the driver |
| 619–682 | `_LoopState` | `subsets_seen_for` (634) + `draw` (638); round identity is `(subset, M_best)`, giving both dedup and a `math.comb` fixpoint rule |
| 683–758 | `_run_round` | |
| 759–787 | `_hint_traces` | `frozen_with_hints` |
| 788–868 | `_learn_and_score` | |
| 869–892 | `save_round_model` | (added in P5.1) |
| 893–902 | `save_round_log` | |

Three pool policies are implemented, not one, because `replace` **voids dedup and
the fixpoint rule** — worth being able to demonstrate rather than assert.

## New — `src/pi_sam/plan_denoising/milp_version/model_prior.py` (171 lines)

Projects a learned model into the encoder's `ObservationM` so a round can be
biased toward the incumbent.

| lines | symbol | logic |
|---|---|---|
| 51–69 | `PriorProjection` | `is_lossless` (65) — flags when the projection dropped something |
| 70–105 | `_position_map`, `_binding` | parameter-position mapping |
| 94–105 | `_flatten_preconditions` | |
| 106–171 | `learner_domain_to_observation_m` | |

## Modified

| file | Δ | change |
|---|---|---|
| `config.py` | +321 | `SubsetSize` (99) incl. `half`; `StopRules` (164) 5 rules; `EvalWeights` (234); `PoolPolicy` (66), `Sampler` (52), `LearnerInput` (59); `pool_is_frozen` (408), `dedup_rounds` (417), `effective_stop_rules` (427) |
| `encoder.py` | +60 | `_model_prior_terms` (340), `_prior_bit_scale` (325), `make_solution_hints` (392); **`solve` (431) pins `random_seed` and `num_workers=1`** — without it CP-SAT races between equally-optimal solutions and round-to-round comparison is noise. This also made `rosame_milp` reproducible for the first time |
| `algorithms.py` | +147 | `CDPS_MILP_LOOP` (52); `cdps_milp_algorithm_name` (108) computes arm-suffixed row labels |
| `learning_helpers.py` | +127 | `learn_cdps_milp_loop` (375); `_prepare_milp_driver_inputs` (282), `_milp_outcome` (322) shared |
| `run_fold.py` | +84 | `_milp_specific` (141) flattens the report into result columns |
| `run_config.yaml` | +28 | loop-only knobs |
| `converter.py` | +22 | |
| `single_round.py` | +9 | |

### `milp_repair_cost` has two scopes — deliberate

`milp_repair_cost` stays **single-round-only**. It carries design §7.1's
`cost(MILP) ≤ cost(best CDPS CFM)` check, which needs one solve covering every
trace. The loop has no such solve, so it reports `None` there and fills
`milp_loop_best_round_repair_cost` + `milp_loop_best_round_subset_size` instead
(`run_fold.py:167`, `178–179`). Putting a subset cost in the fold-wide column
would read as "repaired more cheaply" when it only repaired *less*.

## New — `src/pi_sam/plan_denoising/milp_version/test_loop.py` (672 lines, 65 tests)

---

# P5 — anytime profile, backfill, ablations, audit

## P5.1 / D4 — per-round models (`cdfa451bc`)

The loop returned one incumbent, so rejected candidates vanished. An anytime
curve is a claim about what was on the table at each point in time — which those
rejected candidates are the evidence for — and re-scoring them later needs the
model itself, not the log's summary of it.

| file | lines | change |
|---|---|---|
| `loop.py` | 869–892 | `save_round_model` — every round that produced a model writes `round_{i}/model.pddl` |
| `loop.py` | (`_LoopState`) | the path rides on the state object; `_run_round`/`_learn_and_score` already carry 15+ args each |
| `test_loop.py` | +48 | incl. `_RenderableDomain`, kept separate from `_FakeLearnedDomain` (whose `to_pddl` raises on purpose to keep identity away from text) |

Duplicates are written too. Suppressing them would save ~70 small files at n=8
while making "round 4 had a model" indistinguishable from "round 4 was infeasible".

**The artifact is not the identity.** An end-to-end run confirmed `model_hash`'s
docstring: the winning round's file and the returned model differ in text, entirely
in `:requirements` ordering. Re-parsing yields identical preconditions and effects.
Readers must key on the hash.

## P5.2 / D1 — backfill the MILP arms (`f2e609e42`)

`benchmark/backfill_cdps.py`, +392/−107 → 613 lines.

| lines | symbol | logic |
|---|---|---|
| 104–111 | `_WORK_SUBDIRS` | |
| 112–126 | `AlgorithmSpec` | |
| 127–146 | `_read_milp_config` | accepts a `run_config.yaml`, a file with a top-level `cdps_milp:`, or the bare block |
| 147–171 | `resolve_algorithm_spec` | `--algorithm` ∈ {`cdps_anchored`, `cdps_milp_single_round`, `cdps_milp_loop`} |
| 188–246 | `_degraded_files`, `_frozen_fold_trajectories` | the MILP arms read the cell's **frozen degraded files unchanged** |
| 247–297 | `_stage_anchored_inputs` | the anchored path only |
| 332–341 | `_fold_inputs` | dispatch |
| 342–431 | `backfill_cell` | |
| 432–539 | `_backfill_cell_worker`, `resolve_experiment`, `_gather_tasks`, `_run_parallel` | `--workers` over cells |

**The flag is not the substance.** Reusing the anchored staging path would have
produced rows labelled `CDPS_MILP_*` that were actually init+final-anchored — a
different algorithm, silently non-comparable, voiding design §7.1. So
`anchor_endpoints` follows the arm (`backfill_cdps.py:396`): anchoring is a
property of the trajectories, not of the denoiser.

`gt_rate != 0` is **refused, not guessed**. A cell records problem names, masking
files and test problems but not which state indices had GT injected. Every existing
cell is `gtrate0`, so the branch is unreachable today — it exists to stay unreachable.

## P5.3 / D1 — the loop across n (`c0a8e95cc`, docs only)

Withdrew the earlier "run at 10–20 trajectories" advice. The deciding variable is
the **domain**, not n.

## P5.4 / D3 — ROSAME per-epoch snapshots (`41c90aae8`)

### New — `benchmark/algorithm_adapters/anytime_snapshots.py` (152 lines)

| lines | symbol | logic |
|---|---|---|
| 28–38 | `SnapshotRecord` | |
| 39–65 | `SnapshotWriter` | |
| 66–84 | `start`, `overhead_seconds` (73), `elapsed_seconds` (77) | **overhead is measured and subtracted**, so instrumentation does not inflate the curve it is drawing |
| 85–104 | `maybe_capture` | every Nth epoch |
| 105–136 | `capture` | |
| 137–152 | `close` | |

### Modified

| file | Δ | change |
|---|---|---|
| `po_rosame_runner.py` | +122 | optional `snapshot=` on `learn_per_trajectory` (207) and `learn_pooled` (266); `_serialize_current_model` (325) deep-copies and mutates nothing, so a snapshot cannot perturb its own run; clock start/close at 349/358 |
| `rosame_runner.py` | +26 | `snapshot_interval` (88), **off by default** |
| `test_po_rosame_runner.py` | +226 | 4 tests |

## P5.5 / D5/D7 — the offline anytime harness (`2c0422252`)

`benchmark/evaluation/anytime/` — reader → scorer → curves, all offline, so it can
be re-run against existing artifacts without re-running any experiment.

| file | lines | key symbols |
|---|---|---|
| `checkpoints.py` | 216 | `Checkpoint` (58); `read_cdps_checkpoints` (91), `_read_final_model_fallback` (118) for arms with no intermediate artifacts, `read_loop_checkpoints` (136) reads P5.1's `round_{i}/model.pddl`, `read_snapshot_checkpoints` (163) reads P5.4's, `read_fold_checkpoints` (193) dispatches |
| `score.py` | 207 | `ScoredCheckpoint` (45); `_canonical_model_text` (70) so `:requirements` churn does not create phantom distinct models; `load_fold_observations` (84); `score_checkpoints` (112), `_score_one` (143), `score_fold` (172), `write_scores` (186) — scores with **P3's GT-free V** |
| `curves.py` | 148 | `CurvePoint` (38), `running_best` (46), `step_series` (68) — a step function, since a model is the incumbent until replaced; `plot_fold` (92) |
| `run_anytime.py` | 83 | `find_folds` (31), `process_fold` (41), `main` (60) — CLI |
| `test_anytime.py` | 254 | 9 tests |

## P5.6 / D6 — config-driven ablations (`84276d984`)

One `cdps_milp:` block can name several arms.

| file | lines | change |
|---|---|---|
| `config.py` | 529–583 | `expand_cdps_milp_ablations` — crosses the listed knobs |
| `config.py` | 584–609 | `_validate_ablations` — **rejects `seed`, `stop.*`, `eval.*`**, because two arms differing only there would be averaged into one row naming neither |
| `config.py` | 492–528 | `arm_identity` |
| `algorithms.py` | 66–107 | `_shared_milp_suffix_parts`, `_loop_suffix_parts` |
| `algorithms.py` | 108–129 | `cdps_milp_algorithm_name` |
| `algorithms.py` | 139–204 | `milp_configs_for`, `milp_work_subdir` |
| `algorithms.py` | 205–240 | `cdps_family_names` |
| `run_fold.py` | (−41/+60) | loops over `milp_configs_for` |
| `run_config.yaml` | +19 | the `ablations:` template, lines 60–76 — shipped commented out, so a default run keeps the bare `cdps_milp_loop/` it always had |
| `test_milp_ablations.py` | 226 | 10 tests |

A loop-only knob collapses under `cdps_milp_single_round` (one SR arm, not four),
so listing both algorithm keys does not multiply the SR cost.

## P5.6a — the depot defect (`e53ac523e`)

Not a data problem. Every frozen depot trajectory is clean; the contradiction is
**manufactured in memory on each load**, so no choice of cell avoids it.

`GroundedPredicate.__eq__` walks the type hierarchy (`is_sub_type`, one-directional)
while `__hash__` is `hash(str(self))`, which embeds the type tag. So
`(clear p1 - object)` and `(clear p1 - package)` hash apart but compare equal —
an `__eq__`/`__hash__` contract violation. The set-membership probe in
`ground_all_predicates_in_state` lands in the wrong bucket, never consults `__eq__`,
and CWA-completion appends the negative beside the present positive.

| file | lines | change |
|---|---|---|
| `src/utils/pddl_state.py` | 278–318 | **new** `normalize_predicate_types_in_state` — rewrites type tags to the canonical spelling *before* CWA-completion |
| `src/utils/pddl_state.py` | 319–339 | `ground_all_predicates_in_state` now documents the precondition |
| `src/utils/pddl_state.py` | 340–355 | `ground_all_states_in_observation` calls it on both states (348, 351) |
| `src/utils/pddl_state.py` | 234–265 | `get_all_possible_groundings` comment (249) |
| `src/utils/test_pddl_state.py` | 198 | **new**, 8 tests |
| `converter.py` | +3/−1 | |
| deleted | | `src/depot-polarity-test/` — the throwaway repro (README, `repro.py`, 2 state dumps, 380 lines) |

The fix is in `src/utils/`, so it is **not MILP-specific** — it affects every
consumer of CWA-completion.

## P5.6b — backfill work dir from the row label (`1060cd0f9`)

`backfill_cdps.py` +9/−1. With ablations live, the work dir must follow the
*arm-suffixed row label*, not the algorithm key, or two arms collide in one dir.

## P5.6c — `benchmark/milp_eq16_on.yaml` (`c1f164811`, 14 lines)

The eq16=on config as a tracked file. A `/tmp` throwaway makes the run
unreproducible.

## P5.7 — patch parity, and retiring the bound (`ec263dd78`, `1386a1629`)

### The finding

Design §4.1's converse claim — *every CDPS-reachable conflict-free (T′, Φ) is
MILP-feasible, therefore the MILP optimum lower-bounds CDPS* — is **false**, for
two independent reasons.

**(a) The logged cost was inflated.** `cost` counted raw patch-set members, but
patches are applied by *toggling*: `utils/pddl_state.flip_fluent_in_state` matches
under `{f, ¬f}` and inverts. So a same-key polarity pair is two toggles = zero net
edits, charged as two. On the worst hanoi fold the logged 949 was a realized 647
(151 self-cancelling pairs).

**(b) The comparison was not like-for-like.** CDPS can also buy conflict-freeness
with REQUIRE/FORBID **model** constraints, which cost 0.0 by `CDPSConfig` default.
The MILP has no such move, so a CDPS model carrying Φ ≠ {} is not a point the MILP
could have reached. Evidence: **259/259 violating folds carry model constraints**;
only 269/629 passing folds do.

Restricting to the comparable Φ = {} subset leaves 360 folds — and **all 360 are
`0 ≤ 0`**. The check is not merely weak, it is vacuous. It is retired, not patched.

### New — `src/pi_sam/plan_denoising/patch_accounting.py` (80 lines)

| lines | symbol | logic |
|---|---|---|
| 41–47 | `_normalize_fluent` | strips `(not …)` |
| 48–57 | `patch_key` | `(observation_index, component_index, state_type, normalized_fluent)` |
| 58–67 | `record_key` | same, from a serialized record |
| 68–72 | `_odd_parity_count` | the rule: odd count ⇒ one realized edit |
| 73–77 | `net_patch_count` | |
| 78–80 | `net_patch_count_from_records` | for logs on disk |

### Modified

| file | lines | change |
|---|---|---|
| `conflict_search.py` | 44 | import |
| `conflict_search.py` | 245–261 | `_weighted_cost` now takes `net_patch_count(...)`; the docstring records that the **search** cost stays over-charging on purpose — changing it would change search behaviour, which is a separate experiment |
| `conflict_search.py` | 302, 614 | logs `net_fluent_patch_count` beside `cost`, so the corpus becomes self-describing |
| `single_round.py` | +15 | same accounting on the MILP side |
| `docs/cdps-milp-denoiser-design.md` | +171 | §4.1 corrected; §7.1a "Check 1 is vacuous as specified"; §7.1b arm comparison; §8 two open questions |
| `CLAUDE.md` | 2 lines | `patch_accounting.py` on the module map (`1386a1629`) |

### New — `benchmark/evaluation/milp_arm_audit.py` (245 lines)

The reproducible source for the §7.1a/§7.1b numbers — re-run it before citing them.

| lines | symbol | logic |
|---|---|---|
| 65–71 | `load_json` | |
| 73–82 | `entry_net_cost` | prefers the new `net_fluent_patch_count`, recomputes it for older logs so the existing corpus stays readable |
| 84–99 | `best_cfm_costs` | returns (cheapest overall, cheapest Φ={}) |
| 105–197 | `audit_domain` | row coverage, the bound, the eq16 and SR→loop A/Bs — **paired over folds carrying both rows**, since the arms share data, masking and fold split |
| 200–224 | `report_loop_rounds` | round counts + stop reasons; distinguishes a **budget-limited** null from an **exhaustive** one |
| 226–245 | `main` | argparse: `--root`, `--domains`, `--cells` |

### New — `src/pi_sam/plan_denoising/test_patch_accounting.py` (110 lines, 11 tests)

### Result on the full grid (1080 folds, 4 domains × 270)

- **eq16 on is a small, consistent, free win** (≤ +0.02 solving ratio, no time cost) → should be the default.
- **The loop does not pay for itself**: ~9.4× runtime for ≤ +0.0007 precision.
  Not a budget artifact — stop reasons are `fixpoint` 720 / `perfect_fit` 360,
  **zero timeouts**.
- All three arms sit on an **identical per-domain precision ceiling** (hanoi 0.73,
  npuzzle 0.78, gripper/blocksworld ~0.92–0.94) with recall ≈ 1.0 — the signature
  of over-general preconditions. That is a **learner** limit, not a repair-quality
  one, which is why better repair moves none of these numbers.

---

# Cross-cutting

## Test inventory (all added by this work)

| file | lines | tests |
|---|---|---|
| `src/pi_sam/plan_denoising/milp_version/test_loop.py` | 672 | 65 |
| `src/pi_sam/plan_denoising/milp_version/test_cdps_milp.py` | 501 | 12 |
| `src/pi_sam/plan_denoising/test_evaluator.py` | 315 | 17 |
| `benchmark/evaluation/anytime/test_anytime.py` | 254 | 9 |
| `benchmark/algorithm_adapters/test_po_rosame_runner.py` | 226 | 4 |
| `benchmark/test_milp_ablations.py` | 226 | 10 |
| `src/utils/test_pddl_state.py` | 198 | 8 |
| `src/pi_sam/plan_denoising/test_patch_accounting.py` | 110 | 11 |
| | **2502** | **136** |

## Known gaps

1. **The MILP encoder has no soundness test.** The §7.1 bound was the standing
   check and P5.7 retired it as vacuous. The cheapest replacement is a
   fluent-patch-only CDPS mode (Φ forced empty), which would make the two
   directly comparable again. Not implemented.
2. **`cdps_anchored` cannot run live on a simulated source** —
   `run_fold.py:636-640` raises whenever `pre_built_observations is not None`. It
   is backfill-only. Easy to trip over when composing an `algorithms:` list.
3. **The live runner overwrites `fold_result.json` wholesale.**
   `run_fold.py:701-704` writes `baseline_results + cdps_results` through
   `resume.py:36-39` (mode `"w"`), with no merge by algorithm — any row not in the
   current run is erased. `resume` is a per-fold *skip*, not a merge, so it cannot
   protect those rows. The backfills are the opposite:
   `backfill_common.merge_row` (93) does read-modify-write under an `flock` and
   replaces only the same-algorithm row. **That lock does not protect against the
   live runner**, which does not take it.
4. **`milp_repair_cost` is single-round-only** by design (P4 above) — the loop's
   `None` there is not missing data.
