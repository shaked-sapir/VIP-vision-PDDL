# ROSAME arm rework — working process log

Scratch/working document. Tracks where the six-arm rework stands, what is done,
what is half-done, and what comes next. Not a design doc — the design lives in
`docs/rosame-i-milp-26-implementation-plan.md`.

---

## 0. The target

A 2×2×2 factorial over input × network × MILP. Two cells are void (there is no
symbolic ROSAME in the ICAPS-26 paper), leaving six arms:

| key | input | network | MILP | exists today? |
|---|---|---|---|---|
| `rosame_24` | symbolic | ICAPS-24 | — | yes |
| `rosame_milp_24` | symbolic | ICAPS-24 | ICAPS-26 | yes |
| `rosame_milp_24_tag` | symbolic | ICAPS-24 | ICAPS-26 (tag rule set) | yes — **scoped to this arm only** |
| `rosame_i_24` | image | ICAPS-24 | — | yes |
| `rosame_i_milp_24` | image | ICAPS-24 | ICAPS-26 | yes |
| `rosame_i_26` | image | ICAPS-26 | — | **no — to build** |
| `rosame_i_milp_26` | image | ICAPS-26 | ICAPS-26 | **no — to build** |

Constraints agreed with the user:

- The two `*_milp_24` arms are **not in any paper**. Keep them, but say nothing
  about their provenance in the code — the user reports that himself.
- `rosame_milp_base` is **deleted** — from code *and* from the dashboard's
  `exclude_algorithms`.
- The `tag` variant exists for `rosame_milp_24` **and no other arm**.
- Standing principle: fidelity to upstream over local invention. No invented
  ablations.

Agreed order: **the rename first** (§1–§2, now done), then everything in §4.
§4 is the single authoritative list from here to completion; it follows the
plan doc's phase numbering rather than inventing a second one.

**Where it stands.** Step 1 done and landed. Step 2 (the imaged-24 re-run) done,
dashboard rebuilt, and **the two series moved** — `ROSAME-I_24` materially in
all five domains, `ROSAME-I_MILP_24` barely at all, for a reason that checks out
(§4, Step 2). Phase 1 done and **landed** in `0453e95f5`; Phase 1½ **done** —
decided and its equivalence gate run and green, which unblocks Phase 2. Phase 2
is under way: **2.1 done** (the fold walk extracted to
`benchmark/baselines/image_fold_inputs.py`, now returning both GT endpoint
states) and **2.2 done** (`src/milp/head_alignment.py`, CP index → DL head index
through `rosame_argument_permutation`, mutation-checked on all five domains).
**2.3a done** (`src/milp/trace_tensors.py`, the frame arithmetic and the padded
tensors, fed through the real vendored `Net.forward`) and **2.4 done** with it —
gate 1 now imports `interior_frame_count` instead of restating it. **2.3b done**
(`benchmark/baselines/rosame26_data.py`, the fold-level half that touches disk),
mutation-checked and run end to end on a real blocksworld cell. All that is left
in Phase 2 is 2.5. 304 new tests; full suite **735 passed, 1 skipped** from a
432-passed baseline. Two Phase-1 findings amend the plan — see §6 — and four
more found since: the padded loss while scoping Phase 2 (plan §6.1a, §8 4c);
in 2.2, that the align-by-name bug is a loud miss rather than a silent mis-hit
on our five domains, and that the ICAPS-24 arms were never exposed to it; and in
2.3b, the nullary trailing-space dialect gap (§8 11c) and the constant-pixel
`std` floor (§8 7a), neither of which the unit tests found — the real cell did.

---

## 1. Step 1 — the rename: what is DONE

### 1.1 Runner classes (registry key → row name / display name)

| file | key | `name` | `display_name` |
|---|---|---|---|
| `baselines/rosame_runner.py` | `rosame_24` | `ROSAME_24` | `ROSAME (24)` |
| `baselines/rosame_i_runner.py` | `rosame_i_24` | `ROSAME-I_24` | `ROSAME-I (24)` |
| `baselines/rosame_i_milp_runner.py` | `rosame_i_milp_24` | `ROSAME-I_MILP_24` | `ROSAME-I+MILP (24)` |
| `baselines/rosame_milp_runner.py` | `rosame_milp_24` | `ROSAME_MILP_24` | `ROSAME+MILP (24)` |
| `baselines/rosame_milp_runner.py` | `rosame_milp_24_tag` | `ROSAME_MILP_24_TAG` | `ROSAME+MILP (24, tag)` |

Verified live:

```
rosame_24              name=ROSAME_24          display=ROSAME (24)
rosame_i_24            name=ROSAME-I_24        display=ROSAME-I (24)
rosame_i_milp_24       name=ROSAME-I_MILP_24   display=ROSAME-I+MILP (24)
rosame_milp_24         name=ROSAME_MILP_24     display=ROSAME+MILP (24)
rosame_milp_24_tag     name=ROSAME_MILP_24_TAG display=ROSAME+MILP (24, tag)
BASE abstract OK
```

### 1.2 `rosame_milp_base` retired

Dropped from `BASELINE_REGISTRY`. The class `RosameMilpBaseRunner` **stays** as
the shared plumbing both MILP runners inherit.

**Bug caught here and fixed.** Deleting its `name` override did *not* make it
abstract — it extends `RosameBaselineRunner`, which defines `name` concretely, so
the base silently started reporting as `ROSAME_24`. That would have mislabelled
rows rather than raising, i.e. exactly the failure a migration cannot detect
after the fact. Fix: re-declare `name` as `@property @abstractmethod` on
`RosameMilpBaseRunner`. Verified:
`Can't instantiate abstract class RosameMilpBaseRunner with abstract method name`.

### 1.3 Every other reference updated

- `benchmark/baselines/__init__.py` — registry keys + docstring example
- `benchmark/algorithms.py` — two docstring references
- `benchmark/evaluation/cfm/dashboard_config.yaml` — all five keys;
  `ROSAME_MILP_BASE` removed from `exclude_algorithms` (only
  `ROSAME-I_24__res=64x64` remains)
- `benchmark/experiment_runner.py` — `--algorithms` default + help
- `benchmark/benchmark_runner.py` — fallback `["cdps", "rosame_24"]`
- `benchmark/backfill_baseline.py` — `--baselines` default + help, `--resize`
  help example, two module-docstring usage examples
- `benchmark/evaluation/anytime/checkpoints.py` — `ARM_SUBDIRS`, `_READERS`,
  docstring table
- `benchmark/evaluation/anytime/curves.py` — `ARM_STYLES`
- `benchmark/evaluation/anytime/test_anytime.py` — 5 occurrences
- `benchmark/baselines/test_rosame_i_runner.py` — 10 assertions
- `scripts/cluster/run_backfill_baselines.sbatch`
- `benchmark/run_config.yaml` — key list in the trailing comment
- `benchmark/algorithm_adapters/rosame_milp/__init__.py` + `IMPLEMENTATION.md`
- `src/milp/encoder.py`, `src/milp/encoding_config.py` — `rosame_milp_tag` →
  `rosame_milp_24_tag`
- `CLAUDE.md` — registry key list

---

## 2. Step 1 — the on-disk migration: DONE

### 2.1 `benchmark/migrate_arm_names.py`

Extended to carry both the ROSAME renames and the retired arm's purge.

Why the purge is mandatory, not cosmetic: `algAllowed()` checks EXCLUDE first,
and `bases()` builds the dashboard's series set from the data actually present.
Removing `ROSAME_MILP_BASE` from `exclude_algorithms` while its rows sat on disk
would have made it **start** rendering. So "remove it from excluded algorithms"
*required* purging the rows.

- `LABEL_RENAMES` += the five ROSAME entries (as string literals — importing the
  runners drags in torch, which a filesystem migration has no business needing)
- `LABEL_PURGES = frozenset({"ROSAME_MILP_BASE"})` + `is_purged()`
- `Plan` gained `dropped_labels` + `deletions`
- `plan_fold_result` / `plan_run_params` both return `(rewrites, dropped)`
- `build_plan` skips inside a purged directory and collects both the directory
  and the fold-level `learned_domain_*.pddl` into `deletions`
- **directory renames consult both maps** — our own learner's directories carry
  the registry key, a baseline's the row *label*:
  ```python
  new_dirname = (rename_suffixed(directory.name, KEY_RENAMES)
                 or rename_suffixed(directory.name, LABEL_RENAMES))
  ```
- `apply_plan` deletes first, then rewrites each touched JSON **once** (a file
  can appear under both a rewrite and a drop), then renames
- `summarize` reports `DROPPED labels` and `DELETED paths`, so `--dry-run` shows
  the destructive half
- `rename_domain_filename` deleted — `build_plan` now has to branch on the purge
  before renaming, so the helper had no production caller left

`benchmark/test_migrate_arm_names.py` rewritten accordingly: **46 pass**. New
coverage for the ROSAME renames, `is_purged`, label-keyed baseline directory
renames, the purge itself, a half-migrated baseline directory aborting, and the
dry-run actually printing the deletions.

### 2.2 Applied

Backed up the only irreversible part first (the renames invert trivially):

- `/tmp/rosame_milp_base_purged.tar.gz` — the 60 deleted paths (90 entries)
- `/tmp/rosame_milp_base_rows.tar.gz` — the 30 `fold_result.json` losing a row

Applied in 52 s. Post-migration census, matching the prediction exactly:

| name | fold_result rows | dirs | `learned_domain` files |
|---|---|---|---|
| `ROSAME_24` | 4620 | 4620 | 4620 |
| `ROSAME_MILP_24` | 4320 | 4050 | 4050 |
| `ROSAME_MILP_24_TAG` | 4050 | 4050 | 4050 |
| `ROSAME-I_24` | 210 | 180 | 180 |
| `ROSAME-I_MILP_24` | 180 | 180 | 180 |
| `ROSAME-I_24__res=64x64` | 120 | 120 | 120 |
| `ROSAME_MILP_BASE` | **0** | **0** | **0** |

Re-running the migration reports "nothing to migrate", so it is idempotent.

All 13230 ROSAME directories live under `baseline_models/`. There are **no**
`anytime_snapshots/` directories on disk at all, so the label-aware directory
rename had nothing to catch there.

### 2.3 Verified

- full suite: **432 passed**
- dashboard rebuilt (1.1 MB, 5 domains × 9 sim cells + 5 image experiments).
  Occurrence counts in the HTML: `ROSAME_24` 47, `ROSAME_MILP_24` 90,
  `ROSAME_MILP_24_TAG` 45, `ROSAME-I_24` 12, `ROSAME-I_MILP_24` 7,
  `ROSAME-I_24__res=64x64` 5 (excluded from curves, still named in the config
  echo), and **`ROSAME_MILP_BASE` 0**. No bare `"ROSAME"` / `"ROSAME_MILP"` /
  `"ROSAME-I"` anywhere.
- two stale docstrings fixed: `IMPLEMENTATION.md` on how
  `RosameMilpBaseRunner` is made abstract, and `rosame_runner.py`'s
  `anytime_snapshots/ROSAME/` → `anytime_snapshots/<name>/`

**Step 1 is complete.**

---

## 3. Known, deliberately left alone

- `ROSAME_MILP_24` and `ROSAME_MILP_24_TAG` have no `algorithms:` entry in
  `dashboard_config.yaml`. Pre-existing. An unregistered baseline defaults to
  whichever mode it has data in, and both are symbolic-only, so behaviour is
  already correct. Registering them would only matter if an image-mode run ever
  produced rows under those labels, which it cannot.
- **tensorboard is not a project dependency and will not become one.** Decided
  with the user: guard the import in `dl/network.py`, the same treatment
  `constraint_opt/__init__.py` already gives gurobipy. Upstream's
  `SummaryWriter` logging is inert for us. Recorded as a `VENDOR MODIFICATION`.
- **`dl/util/tuning.py` is a stub.** Upstream's is 655 lines of grid/genetic
  hyperparameter search that spawns subprocesses and re-runs whole experiments —
  none of which may happen inside a fold. Exactly one symbol is reachable from
  the code we vendor: `dl/main/normalization` imports its `parameters` dict and
  uses it as an image mean/std cache. The stub is that dict and nothing else.
  Note the upstream cache is **unkeyed** and carries the image shape, so one
  process touching two domains reuses a wrong-shaped array (plan §4.4); the
  Phase-2 adapter owns a keyed on-disk cache and writes through this dict. Also
  a `VENDOR MODIFICATION`.

---

## 4. The remaining work, to completion

Phase numbers below are **the plan doc's** (`docs/rosame-i-milp-26-implementation-plan.md`
§10), not a second scheme. Step 1 above was preparatory and sits outside that
table. Phase 0 (the `Resize(64)` fix + the resize A/B) is already **CLOSED** —
shipped, run on 4 domains × 30 paired cells, hypothesis **refuted**
(analysis §5.3).

### Step 2 — re-run both imaged 24 arms with `--force`  ← **DONE**

Not part of the plan's phase table; a debt from the `train_per_trajectory`
deletion (`3a9d1b67e`). Every existing `ROSAME-I_24` / `ROSAME-I_MILP_24` row was
produced by the per-trajectory schedule, which is *not* what ICAPS-24 `main/`
does. The migration renamed those rows; it did not make them upstream-faithful.

- [x] re-run `rosame_i_24` and `rosame_i_milp_24` with `--force` across the image
      experiments. Verified by mtime, not by report: every `ROSAME-I_24` row in
      all seven image experiment directories (five domains) was rewritten
      2026-08-21 between 01:13 and 04:22 — blocksworld 60, depot 30 + 60
      (`__groundfix`), gripper 60, hanoi 30 + 60, npuzzle 30, and 180
      `ROSAME-I_MILP_24` rows alongside.
- [x] rebuild the dashboard; confirm the two series moved. **Rebuilt** — bare
      build, 29 s, 1.1 MB. `--regen-plots` / `--refresh-stats` were *not* needed
      and would have been no-ops: of the 150 files newer than the old page, all
      150 are `fold_result.json` and every one is in the five configured image
      experiments. `all_solutions_metrics.json`, `conflict_free_solutions_log.json`
      and `original_observations/` are untouched, and the trend PNGs are built
      only from the CFM series (`build_dashboard.py:405`), never from baseline rows.

**Not a null result.** Diffing the embedded `DATA` blob of the old page against
the new, over all five image domains and all reported metrics:

| arm | rows changed | size of change |
|---|---|---|
| `ROSAME-I_24` | 18 of 25 domain×metric cells | **large** — up to +0.35 |
| `ROSAME-I_MILP_24` | 5 of 25 | ≤ 0.017, four of them float-ulp |

`ROSAME-I_24`, mean over folds, before → after:

| domain | app_prec | app_rec | eff_rec |
|---|---|---|---|
| blocksworld | 0.153 → 0.487 | 0.657 → 0.946 | 0.053 → 0.228 |
| depot | 0.125 → 0.041 | 0.726 → 0.954 | 0.071 → 0.105 |
| gripper | 0.220 → 0.220 | 0.560 → 0.345 | 0.178 → 0.333 |
| hanoi | 0.057 → 0.154 | 0.609 → 0.963 | 0.000 → 0.095 |
| npuzzle | 0.048 → 0.072 | 0.300 → 0.232 | 0.000 → 0.000 |

Two things to read off this rather than the direction of any single arrow.

**The asymmetry is explained, not mysterious.** `3a9d1b67e` rewrote
`benchmark/algorithm_adapters/rosame_i_runner.py` (−36 lines) but touched
`rosame_i_milp_runner.py` by three. The MILP arm runs through
`rosame_milp/milp_loop_i.py::learn_pooled_with_milp`, which was **already**
pooled, so the schedule deletion could not move it. Only the DL-only arm's
training changed, and only the DL-only arm's numbers changed. That the two agree
is a check on the causal story, not a coincidence.

**`ROSAME-I_24` is now more degenerate, not less.** `pred_eff_precision` is
1.000 in four of five domains against `pred_eff_recall` of 0.00–0.33 — vacuous
precision over a near-empty effect set — while `pred_app_recall` runs to
0.95–0.96 at `pred_app_precision` of 0.04–0.15, i.e. preconditions saturating
toward "assert everything". This is precisely the shape **gate 4** (§9.4,
degenerate-model guard) exists to catch, and it is now firing on the *24*
baseline before the 26 arm has run at all. Do not report a 26-vs-24 delta
against this without saying so.

Two secondary observations worth keeping:

* The **fold count changed** for three domains — blocksworld 22 → 28 usable
  `pred_*` rows, depot 29 → 22, npuzzle 10 → 9, while `solving_ratio` stays at
  30 everywhere. So the re-run changed *which* folds yield a scoreable model,
  not only what those models score. Means taken over differently-sized
  populations; the table above is indicative, not a paired test.
* `ROSAME-I_24__res=64x64` — the frozen Phase-0 A/B series — did **not** move,
  as it should not have: `bases()` keeps `__`-suffixed rows separate and the
  re-run wrote no rows under that name.

Until this lands, the 26 arm has no honest 24 baseline to be measured against,
which is what Phase 4 exists to do.

One caveat the re-run inherits: it is **unpinned**. The Phase-1 bullet asking for
ICAPS-24's `main/` to be vendored — so `rosame_i_24` gets the parity test
`rosame_24` already has — is still open (below), so nothing mechanically checks
that the new rows match upstream's schedule. They are believed-faithful, not
shown-faithful.

### Phase 1 — vendor + assets + parity tests *before* any adapter  ← **substantially DONE**

**143 tests** across the six files below (142 passed, 1 skipped). Together with
Phase 1½'s 7 that is **150 new tests**, and the arithmetic closes: the full suite
now reports **581 passed, 1 skipped** against a 432-passed baseline, so every new
test is accounted for and nothing pre-existing broke.

All of it **landed** in `0453e95f5` "Vendor the ICAPS-26 DL tree and pin the
grounding scope" — 50 paths, including Phase 1½. (This paragraph previously read
"still uncommitted, `git status` shows every path untracked"; that was written
before the commit and never updated.)

- [x] vendor `dl/` + `convertor/` + `util/model_perm.py` (§2.1 fixes the exact
      boundaries). Vendored `dl/` (incl. `dl/main/normalization.py`,
      `dl/mixins/`, `dl/util/ROSAME/`), `convertor/` (all four modules),
      `planning_structs/util.py`, and `util/` — which is `model_perm.py` as the
      plan says **plus `pddl_parsing.py`**, not in the plan but pulled in
      transitively by `convertor.convertor`.
      Nine `VENDOR MODIFICATION` markers now exist;
      **eight are new this phase**, only the gurobipy guard predates it:
      - `dl/network.py:8` — tensorboard guarded (§3)
      - `dl/util/tuning.py` — stubbed (§3)
      - `dl/util/ROSAME/rosame.py:375`, `dl/mixins/action_model.py:10` —
        `domain_assets_root` (§6.1)
      - `dl/__init__.py`, `dl/main/__init__.py` — upstream re-exports its
        argparse CLI layer (`dl/main/common.py`, `rosame_full.py`), which we do
        not vendor because the adapter replaces it; without these edits the
        package is unimportable. `dl/main/` survives only to host
        `normalization.py`.
      - `util/pddl_parsing.py:6` — `lifted_pddl` guarded. It is not a project
        dependency and this module sits on the import path of *every* consumer
        of `dl/network.py` (via `convertor.convertor`), so an unguarded import
        makes the entire vendored DL tree unimportable. `Parser` is reachable
        only from `parse_pddl_domain`, which raises.
      - `util/pddl_parsing.py:76` — upstream reuses the outer double quotes
        inside an f-string, which only parses under PEP 701 (Python ≥ 3.12).
        We are on 3.11. Rewritten with single quotes; identical output.
- [ ] **also vendor `main/`** — our extension, not in the plan's table. It is what
      lets `rosame_i_24` get an upstream-parity test of the kind
      `test_po_rosame_runner.test_local_loop_matches_vendored` already gives
      `rosame_24`. Without it, step 2's re-run is unpinned. **Still open** —
      and step 2 has now run anyway, so the debt is live rather than theoretical.
      Note this is ICAPS-24's `main/` (branch `main` @ `6573e7d`), a different
      checkout from the ICAPS-26 tree everything else here came from; the
      `dl/main/` directory in the vendor tree is ICAPS-26's and is not it.
- [x] generate all five domain specs + `pddl/<domain>/domain.pddl` from **our**
      domains (§5) — not copied from upstream. `src/milp/domain_assets.py`,
      re-runnable as `python -m src.milp.domain_assets`; 21 tests.
- [x] import-only smoke test — 49 tests.
- [x] **write verification gates 1–3 now, against the vendored code, before the
      adapter exists** (§9). Three of the five errors an external review found in
      the plan's first draft are things these catch mechanically:
  - [x] **gate 1 — shape/parity**: `z` is `[B,T,S]`, `a_logit` `[B,T+1,adim]`,
        `z_suc_aae` / `p_applicable` `[B,T+1,adim,S]`, `state_traces` `[B,T+2,S]`,
        `z ∈ [0,1]`; assert `T = N−1` explicitly; assert a 2-image trace is
        **rejected**, not silently `T = 0`. All confirmed empirically against the
        real `Net`, not asserted from the plan.
  - [x] **gate 2 — loss parity**: tiny synthetic trace through vendored
        `dl/model.py` and through our harness, equal to float tolerance. Cover
        **both** `loss_pseudo_a` regimes (no MILP labels; some MILP labels). This
        is the test that actually protects the port.
        Gates 1+2 share `src/milp/test_vendor_net_contract.py`, 23 tests.
  - [~] **gate 3 — MILP parity**: `src/milp/test_vendor_translator_contract.py`,
        9 tests — **two of the four obligations, not four**. Landed: the
        `trans_full_state` zip alignment, and `state_label`/`action_label` sizing
        under a ragged bundle, plus the `chosen = 0` fallback. Deferred to
        Phase 2 because they need the adapter that does not yet exist: that the
        §0.1 identity mappings reach `extract_sol_*`, and that `run_fixer`
        agrees with a direct translator call.
- [x] `check_predicate` round-trip over all five domains (§5.1) — 20 tests
      (1 skipped: gripper has no predicate with repeated argument types).

Two findings that change the plan, both detailed in §6 below: a **third asset
family** the plan does not mention, and an **argument reordering** that is a
Phase-2 obligation the plan does not record.

`src/milp/vendor/UPSTREAM.md` **has been brought up to date**. It described only
`planning_structs/` and `constraint_opt/` and named none of the eight new
modifications; it now carries per-file verbatim/MODIFIED status for every
vendored package, a "not vendored, deliberately" list, the nine-row modification
table every `VENDOR MODIFICATION` comment points at, the two asset families
(§6.1), and the sorted-type reordering (§6.2). It is the authoritative record of
the vendor boundary — the plan doc summarises, it does not.

### Phase 1½ — PIN THE GROUNDING SCOPE  ← **DONE**

**Decision, from the user: one `Instance` for the whole run, as upstream does.**
Head width == CP width. No 0.5/0.0 bridge, no §8 deviation entry, because
following upstream is not a deviation. The equivalence gate below was retained
with its status changed: a **confirmation** of a choice already made, not the
tiebreak between two live options. It has now been run and confirms it.

This was the blocker on Phase 2 — no code for `rosame_i_26` or
`rosame_i_milp_26` could be written until it was settled (plan §4.2a, open
decision 19). It is settled; Phase 2 is unblocked.

One `Instance` for the whole run, as upstream does
(`convertor/convertor.py:48-49` builds it from the ROSAME domain's object
universe and shares it across every trace), or one per problem, as our 24 arm
does (`rosame_i_milp_runner.py:155`, surplus union columns dropped at
`converter.py:473-486` — the `N of M CV propositions have no counterpart`
warning)?

Upstream never had to choose: its corpora are object-homogeneous. **Ours are
not** — blocksworld problems carry 4 or 5 blocks, so on a 4-block trace 11 of
the union's 36 propositions name the absent block `e`. Following upstream is
therefore a decision, not an inherited default.

- [x] pin it. It sizes the DL symbol head, `extract_sol_label`'s width and the
      `loss_pseudo_s` BCE, so it is **not** deferrable to Phase 5 — discovering
      it there means rebuilding the adapter and discarding anything already
      measured. **Pinned: one shared `Instance`.**
- [x] either way: take the union over the whole `data_dir`, **not the fold**.
      Measured — all 30 blocksworld cells ground at `n_props = 36`, but by
      composition (every fold happens to include a 5-block problem), not by
      construction. A 4-block-only fold would ground at 25 and the grid would
      average two vocabularies. Same argument and same cache key as §4.4's
      normalisation statistic. **Pinned: `data_dir`-wide union.**
- [x] **equivalence gate**: solve one 4-block blocksworld trace under both
      groundings, assert the recovered model is identical. The claim that the
      phantom propositions are inert (pinned false by hard init + frame axioms,
      decoupled from the lifted schema variables) was a *prediction*; if it had
      failed, the choice would have been a measurement decision, not a fidelity
      one.

      **Run, and the prediction holds.**
      `src/milp/test_grounding_scope_equivalence.py`, 7 tests, 1.4 s. One clean
      4-block trace exercising all four schemas, grounded on `a b c d` and on
      `a b c d e`:

      | | per-problem | union |
      |---|---|---|
      | propositions | 25 | 36 (11 name `e`) |
      | `hol` variables | 125 | 180 |
      | lifted model variables | 78 | 78 |
      | lifted variables that differ | — | **0** |
      | phantom propositions true at any step | — | **0** |
      | repaired live states | identical | identical |

      The lifted count is 78 either way *by construction*, not by luck:
      `predicate_arguments` is built from the `Domain`, so the 26 (schema,
      predicate, binding) triples cannot see how many objects were grounded.
      What the gate adds is that the *solution* over them does not move either,
      and that the state channel does not move on the propositions the two
      groundings share.

      Two non-vacuity guards, because an equivalence result is only as good as
      the difference it was given a chance to detect. The `hol` row above shows
      the two solves really are two problems (55 extra variables, not a
      no-op). And the equality is not CP-SAT tie-breaking: `solve` pins
      `random_seed` so two solves of one encoding break ties alike, which would
      make the gate pass for free if the optimum were underdetermined. It is
      not — re-solving the narrow encoding under seeds 7 and 1234 returns the
      same 78 values.

      Scope, stated so it is not over-read: one domain, one clean fully-observed
      trace, one absent object, and every phantom handed to the solver at ε.
      That last part is the limit. Here the phantoms are false *because the
      observation says so*; the frame argument (no observed action mentions
      `e`, so nothing can make it true) is what should carry the result, but the
      gate cannot separate the two.

      It matters because in `rosame_i_*` the per-step values are the network's
      sigmoid outputs, not a CWA completion — and the head has no way to know
      `e` is absent from *this* problem, so it will emit some value for
      `(clear e)`, possibly a confident one. A phantom pulled towards true by
      the observation channel and pushed towards false by the frame is a
      contradiction the solver pays for, and it pays in the state objective,
      which is a reported number. **Re-run the inertness half of this gate on
      real head outputs in Phase 2**, once the adapter exists to produce them.
- [x] ~~register the outcome in the deviation register (§8) if per-problem
      wins~~ — moot; per-problem was not chosen. What §8 *should* get instead is
      the opposite entry: that on our object-heterogeneous corpora the shared
      instance carries phantom propositions the 24 arm does not, and that this
      is a fidelity choice made knowingly. **Landed as §8 item 11a**, with the
      Phase-2 re-check on real head outputs named in the entry itself so it
      cannot be quietly skipped.
- [x] correct the plan doc's Phase-4 table row for `grounding scope` — **already
      done**, at `docs/rosame-i-milp-26-implementation-plan.md:1281`, which now
      reads `one Instance over the data_dir union (**DECIDED**, Phase 1½)`
      against a 24-arm cell of `per problem, then surplus union columns dropped`.
      This bullet was left unticked; it describes work that had landed.

### Phase 2 — the data adapter  ← **the real work**

Four scoping decisions taken before starting, all recorded below and in the plan
doc: the fold walk is **extracted to a shared module** rather than duplicated;
padding is **pad + zero-action mask with a length-aware loss subclass**
(§6.1a, §8 item 4c); the two remaining §9.3 gate-3 obligations are **deferred to
Phase 5**, since both need `Convertor` wired to `src/milp/encoder.py`; and the
dashboard rebuild landed first, so Phase 4 has a settled 24 baseline to point at.

**2.1 — extract the fold walk.** ← **DONE**

- [x] `RosameIBaselineRunner._resolve_inputs` and its eight helpers
      (`_resolve_images`, `resolve_final_state_path`, `_resolve_final_state`,
      `_domain_uses_hyphens`, `_conformed_tempfile`, `_parse_problem_normalized`,
      `_parse_trajectory_normalized`, `_state_positive_predicates`,
      `_infer_domain_name`) move to a shared module. `_resolve_inputs` touches
      `self` only to dispatch to static/class methods, so the extraction is
      mechanical.

      Landed as `benchmark/baselines/image_fold_inputs.py`. The walk is
      `resolve_fold_inputs(partial_domain, prepared_trajectories, bench) ->
      List[ResolvedTrace]`, and `ResolvedTrace` is a frozen dataclass rather
      than the old 4-tuple, because the 26 arm needs a fifth field the 24
      adapter must not see. `RosameIBaselineRunner._resolve_inputs` survives as
      the shape converter — it maps `ResolvedTrace` back to the positional
      4-tuple `RosameI_Runner.prepare_traces` unpacks — so `rosame_i_milp_24`,
      which calls it on `self`, needed no edit at all.

      One simplification taken while moving `_infer_domain_name`: its loop
      checked `if key in _HYPERPARAMS: return key` before the alias lookup, and
      that branch is dead — every `_HYPERPARAMS` key is also a `_DOMAIN_ALIASES`
      key mapping to itself, so the alias lookup already returns the same value.
      Dropping it decouples bench-key inference from the hyperparameter table,
      which is a *paper* artefact and does not belong in a shared module. The
      invariant that made it safe is now a test rather than an assumption:
      `test_every_tuned_domain_is_a_resolvable_bench_key`.

- [x] the shared version additionally returns the **GT init state**, which the 24
      arm never reads and the 26 arm needs as a hard anchor (§4.3)

      Both anchors come from **one** parse of the GT trajectory —
      `components[0].previous_state` and `components[-1].next_state` — rather
      than the init coming from the problem's `:init` as §4.3 describes it. One
      parse means one grounding and one dialect policy for the two anchors,
      which is the hazard `rosame_i_milp_runner.py:139-147` already documents
      for the goal channel. The two sources being equal is not assumed: it is
      checked on all ten problems of the checked-in blocksworld cell
      (`TestGtInitAgreesWithTheProblem`), and they do agree.

- [x] the 24 arm's tests stay green, unchanged — that is the check that the
      extraction was behaviour-preserving

      Green, and with only one substantive change. The GT-anchor and
      input-dialect cases moved verbatim into
      `benchmark/baselines/test_image_fold_inputs.py` with call sites retargeted
      at the free functions; the row-naming and resize cases stayed in
      `test_rosame_i_runner.py` untouched. The exception is
      `TestResolveFinalState`, which monkeypatched
      `RosameIBaselineRunner._parse_trajectory_normalized` to inject an empty
      observation. That patch had no target after the move, so the test was
      rewritten to call `gt_anchors_from_observation` on a stub directly — a
      strictly better test, since it no longer depends on which method the walk
      happens to dispatch through, and it now also pins which component each
      anchor is read off.

      Beyond the tests: a real end-to-end smoke of the 24 arm through the
      extracted walk on `blocks_predefined_problems1-10_final-version` — 10
      problems resolved, 10 traces prepared, one epoch trained, a four-schema
      model out. Full suite **594 passed, 1 skipped**, up from 581 by the 13 new
      tests.

      Two facts the new real-data tests pin, both already predicted by the plan:
      frames outnumber actions by exactly one on every problem (§4.1), and
      `problem1` carries 2 frames for 1 action, so its `T = N-1 = 0` and it is
      the trace §8 item 11b says the 26 arm must drop.

**2.2 — head↔CP index alignment.** ← **DONE**

- [x] **head alignment through `rosame_argument_permutation`** — new, §6.2 below.
      Not in the plan. Map CP proposition index → DL head index; aligning by
      predicate name alone is silently wrong on depot, gripper, hanoi and
      npuzzle. Gate 3 shows why this cannot be caught downstream:
      `trans_full_state` zips positionally, so a permuted vector of the right
      width produces the wrong propositions and no error

      `src/milp/head_alignment.py`. `head_key(name, args, types)` is the
      primitive — args in PDDL order out as the head's space-joined key —
      and `proposition_head_indices` / `action_head_indices` lift it to a whole
      grounding, returning CP index → head index. Both **raise** rather than
      skip, on a size mismatch, on a key the head lacks, and on a collision;
      `invert` gives the other direction and raises unless the map is a
      bijection. Actions are covered as well as propositions, because §2.3's
      `action_traces` is one-hot in head order.

- [x] tested on all five domains, with a **mutation check** — perturb the map and
      assert the test fails. blocksworld has permutation 0 and would pass a broken
      implementation

      `src/milp/test_head_alignment.py`, 45 tests. The DL side is the real
      vendored `Domain_Model` via `get_domain_model`, not a reimplementation of
      its ordering rule. Bijectivity alone would be satisfied by *any*
      permutation, so `test_the_aligned_head_key_names_the_same_objects` also
      pins that each CP proposition lands on the column holding its own objects.
      The mutation is `rosame_argument_permutation` monkeypatched to the
      identity — exactly the align-by-name bug — and it raises on all four
      reordering domains; `test_blocksworld_cannot_detect_the_mutation` pins the
      converse so the parametrisation is never trimmed back to blocksworld.
      Beyond the synthetic universe, `TestOnARealObjectUnion` runs the whole
      alignment over the real object union of the checked-in depot and
      blocksworld cells, where object names are arbitrary rather than named
      after their type.

**Two findings from 2.2, both worth recording.**

*The align-by-name bug is loud on our domains, not silent.* Measured, propositions
then actions, on two objects per leaf type:

| domain | propositions | actions |
|---|---|---|
| blocksworld | 9/9 correct | 8/8 correct |
| depot | 30/42, 12 miss | 0/84, 84 miss |
| gripper | 12/12 correct | 2/18, 16 miss |
| hanoi | 12/16, 4 miss | 8/12, 4 miss |
| npuzzle | 4/8, 4 miss | 0/4, 4 miss |

Every failure is a **miss**, never a wrong hit — on the real depot and
blocksworld unions too. Our five domains declare disjoint types, so a mis-ordered
tuple is not type-valid and the head simply has no such key. That is a property
of these domains, not of the scheme, and it does not make name alignment safe: a
wrong tuple that happened to be type-valid would decode silently, and even a miss
only reaches `cv_predictions_to_trace`'s `unmapped` counter, which **warns and
defaults the column to 0.5**.

*The ICAPS-24 arms carry no defect from this.* Worth stating explicitly, because
the reordering would otherwise implicate already-reported `ROSAME-I_MILP_24`
numbers on four of five domains. AMLGym's fork carries the sort commented out and
recovers PDDL order by accident of dict insertion order, and
`benchmark/algorithm_adapters/test_check_predicate.py` pins the identity
round-trip on all five domains — 19 passed. No permutation step is needed there,
and none was silently missing.

**2.3 — the adapter.** Split in two, because `src/` may not import `benchmark/`
and gate 1 lives in `src/milp/`. **2.3a is the tensor contract** — pure, no I/O,
`src/milp/trace_tensors.py`. **2.3b is the fold adapter** — object union, image
loading, resize, normalisation cache — and lives under `benchmark/`.

*2.3a* ← **DONE** (`src/milp/trace_tensors.py`, 39 tests)

- [x] resolve the one-frame-too-long mismatch (§4.1): `interior_frame_count` is
      the single implementation, both endpoints dropped so `T = N−1`, and
      **T < 1 raises**. `interior_frame_indices` is the frame-selection form
- [x] pad to a uniform T, emit a **per-trace length array**, and zero the action
      one-hot on padded steps (§6.1a). The loss subclass that consumes the
      lengths is Phase 3 work; Phase 2 owes it the data
- [x] `state_traces` per §4.3: GT init at row 0, **zero filler** interior, GT
      goal **held from each trace's true final row to the end** — so the
      vendored `state_traces[:, -1, :]` slice is that trace's goal whatever its
      length, and no VLM state ever enters
- [x] PDDL-order names → head columns via `head_alignment`'s new
      `proposition_index_by_name` / `action_index_by_name`, which is the exact
      string form `image_fold_inputs` already emits (paren-free, space-separated)
- [x] the batch is fed through the **real vendored `Net.forward`** and its
      outputs re-checked against gate 1's shapes — a shape contract cannot be
      satisfied by agreeing with itself

*2.3b* ← **DONE** (`benchmark/baselines/rosame26_data.py`, 42 tests)

- [x] fold → their contract (§4.1–4.4). `build_fold_batch` is the one entry
      point: traces in, a `FoldBatch` out carrying the `PaddedTraces` 2.3a
      defines, the grounding, the stats, and the kept/dropped split
- [x] proposition space is upstream `Instance`, **no repeated args** (§4.2) —
      unlike our `RepeatedArgsInstance`
- [x] grounding assets (`domain_model.json` + `objects.json`) written per run
      into a scoped root via `write_grounding_assets`, §6.1 below
- [x] image normalisation computed once over the whole `data_dir` (§4.4, DECIDED)
      — cached to `.rosame26_norm__<bench>__res=<tag>.pt` beside the corpus, and
      a cache hit reads **one** frame to check the shape rather than all of them
- [x] resize per-domain configurable, default `Resize(64)` (§4.6, DECIDED).
      `_build_image_tf` was promoted to `build_image_tf` so the 24 and 26 arms
      share one definition of PNG → tensor rather than two that can drift
- [x] report the **dropped set** when `T < 1` rejects a trace — it costs
      blocksworld `problem1` (§8 item 11b). The raise is 2.3a's; the reporting
      is the fold walk's. `FoldBatch.dropped` names the problem and the reason;
      only that documented case is dropped, and an action count that does not
      span its frames still raises

**Two things the tests alone would not have caught.** Both came out of running
the module on the real `blocks_predefined_problems1-10_final-version` cell.

*A `KeyError` on `handempty`.* `untyped_representation` renders a nullary as
`"(handempty )"`, so the fold walk's `[1:-1]` slice yields a **trailing space**
that no head key has. The fix widened the dialect normaliser from
`replace("-", "_")` to `" ".join(name.replace("-", "_").split())` — but the
useful half was changing the shared test fixture to emit `"handempty "`, so the
suite now reproduces what `state_positive_predicates` actually produces instead
of an idealised form. Registered as plan §8 item 11c.

*A `std` floor.* Upstream divides by `std + 1e-20`. On our renders that is not
cosmetic: **51.8% of blocksworld pixels and 70.5% of depot's are constant**, and
that epsilon standardises each to a magnitude of order **2.4e13 / 3.6e13**.
Floored at `1e-6` instead. On pixels with real variance the two agree to exactly
`0.0`. Registered as plan §8 item 7a.

**Mutation-checked**, because a green suite over code and tests written together
proves little:

| mutation | tests that failed |
|---|---|
| drop `.clamp_min(STD_FLOOR)` | 3 |
| skip key canonicalisation | 1 |
| never read the cache | 1 |
| keep the endpoint frames | 9 |

**And run end to end on the real cell**, through the vendored `Net.forward`:
object union `{block: a..e}`, `n_props` **36** (the value §4.2a documents) and
`n_actions` **50**; `problem1` dropped for having two frames, exactly as §8 item
11b predicts, leaving 9 traces; `images (9, 11, 3, 64, 64)`,
`states (9, 13, 36)`, `actions (9, 12, 50)`; per-trace interior filler zero and
the goal held from each true final row; padded action rows all-zero and real
rows one-hot; `z (9, 11, 36)`, `a (9, 12, 50)`, `z_suc_aae (9, 12, 50, 36)`,
head width equal to `n_props`, all finite.

**2.4 / 2.5 — close the Phase-1 loose ends.**

- [x] must pass gate 1 — and **repoint it**: `test_vendor_net_contract.py` now
      imports `interior_frame_count` from `src.milp.trace_tensors` instead of
      restating it. The gate keeps the *contract* (what `T` must be for the
      vendored shapes to line up); the arithmetic has one definition
- [ ] re-run the **phantom-inertness** half of the Phase-1½ gate on real head
      outputs (§4.2a, §8 item 11a) — the symbolic result does not transfer,
      because the head emits a sigmoid for `(clear e)` whether or not `e` exists

**Deferred out of Phase 2** (into Phase 5, where `Convertor` is wired to
`src/milp/encoder.py` and both become cheap): the two §9.3 gate-3 obligations —
that the identity mappings actually reach `extract_sol_*`, and that `run_fixer`
agrees with a direct translator call.

### Phase 3 — harness replacement

- [ ] override one method, keep everything else (§3)
- [ ] pin the run-level settings explicitly (§1.3), including **augmentation
      disabled** (the 24 arm horizontally flips on blocksworld)
- [ ] batching built for any N, not just today's N (§1.1)
- [ ] **length-aware `ROSAMEGoal` subclass** consuming Phase 2's length array
      (§6.1a, §8 4c): endpoint γ terms gathered per trace, `weight_mask_a` and
      `loss_pseudo_s` zeroed past each trace's end, normaliser over true steps
- [ ] must pass gate 2 — **extended**: on a length-homogeneous batch the masked
      loss must return values bit-identical to the vendored one, which is what
      confines 4c to the ragged case

### Phase 4 — `rosame_i_26`, DL-only, one blocksworld fold

The first real milestone: the old-vs-new DL comparison *before* any MILP work,
and where the empty-effects question gets answered.

- [ ] one blocksworld fold, `pre_mip_epoch ≥ epochs` (gate 5, sanity run)
- [ ] **gate 4 — degenerate-model guard**: reject a learned model with zero
      add-and-delete effects across all schemas, loudly. This would have caught
      the ICAPS-24 empty-effects collapse on day one.
- [ ] **gate 7 — budget control**: one cell per domain at `epochs: 5000` outside
      the timeout (§1.2), so *"underperforms at 750"* and *"undertrained at 750"*
      stay distinguishable. A **26-arm item only** — 5000 is the 26 code default
      and the value §1.2's pre-flight calibrates away from; the 24 arm runs the
      ICAPS-24 paper's own 70/100/300.
- [ ] run the §1.2 pre-flight budget check before committing to any epoch count

**Phase 4 is not an architecture comparison unless what else moved is stated.**
A 24→26 delta has eight candidate causes and only the first is the one people
will assume:

| | ICAPS-24 arm | ICAPS-26 arm |
|---|---|---|
| state head | raw logits, no sigmoid | `z = sigmoid(...)` ∈ [0,1] |
| loss reduction | `sum` | normalised by `B(T+1)` |
| augmentation | h-flip on blocksworld | disabled |
| images per trace | N + 1 | **N − 1** |
| proposition space | `RepeatedArgsInstance` | upstream `Instance` |
| argument order | PDDL signature order | **sorted by type name** (4 of 5 domains) |
| grounding scope | per problem, then surplus union columns dropped | one `Instance` over the `data_dir` union (**DECIDED**, Phase 1½) |
| epoch budget | per-domain 70/100/300 | one calibrated value |
| batch size | 1 | 128 |

- [ ] report the delta **with this table attached**, or hold the movable ones
      fixed in a dedicated ablation

Presenting it as "old architecture vs new architecture" would repeat the resize
confound of analysis §5 — a real effect attributed to the wrong cause.

### Phase 5 — `rosame_i_milp_26`

- [ ] turn the MILP on, using **`src/milp/encoder.py`**, not the vendored solvers
      (§6.1, DECIDED)
- [ ] MILP cadence per §6.2
- [ ] one fold; verify the `mip_gt_dist` logs
- [ ] must pass gate 3 — **including its two deferred halves**: that the §0.1
      identity mappings reach `extract_sol_*`, and that `run_fixer` agrees with a
      direct translator call. Neither could be written in Phase 1; both are
      write-once-the-adapter-exists, so they belong to whoever does Phase 2.

### Phase 6 — full grid

- [ ] backfill across all 5 domains. Compute only; no new risk.
- [ ] rebuild the dashboard; register the two new keys in
      `dashboard_config.yaml` `algorithms:` with `modes: [image]`

### Phase 7 — out of scope here

- option A, predicted actions (§0, §7). High risk, explicitly deferred.

### Closing out

- [ ] ship the **deviation register** (§8) in the thesis, plus §8.1 *"what this
      arm will not fix"*
- [ ] resolve or restate whatever is left under the plan's *"Still open"*
      (§11, end)

---

## 5. Notes worth not re-deriving

- **Two name spaces.** Registry KEYS are lowercase (CLI/YAML); row/display names
  are the `name` property, written to `fold_result.json`'s `algorithm` field and
  used as dashboard series keys. Directories under `baseline_models/` and
  `learned_domain_<NAME>.pddl` files are named by the **row name**, not the key.
- **`__` suffixes.** `rename_suffixed` partitions on `__` and carries the suffix
  over. `bases()` (`build_dashboard.py:716`) does *not* strip suffixes, so
  `ROSAME-I_24__res=224` is its own series. That is why `ROSAME_MILP_24_TAG` was
  chosen over `ROSAME_MILP_24__tag` — behaviourally identical, smaller diff.
- **Class lineage.** `Rosame_Runner` (AMLGym vendored) ← `PORosame_Runner` ←
  `RosameI_Runner` ← `MilpRosameI`. So `rosame_i_milp` is the **24 network**
  driven by the **26 MILP**. The `rosame_i_milp_runner.py` docstring used to
  overclaim this as "the ICAPS-26 competitor in its native setting"; corrected.
- **Two different upstreams.** Symbolic `rosame_24` reproduces AMLGym's vendored
  `learn_rosame` — per-trajectory training *is* upstream there, pinned by
  `test_po_rosame_runner.test_local_loop_matches_vendored`. Image `rosame_i_24`'s
  upstream is ICAPS-24 `main`/`train.py`: pooled, shuffled, batch 128. That is
  why `train_per_trajectory` was deleted from the image arm only (`3a9d1b67e`)
  and deliberately kept on the symbolic one.

---

## 6. Phase-1 findings that change the plan

Two things surfaced while vendoring that the plan doc does not account for.
Both are Phase-2 obligations; neither is optional.

### 6.1 There is a *third* asset family

Plan §5 lists two per-domain assets, and both are what `Convertor.__init__`
reads, relative to the vendor root:

    planning_structs/specs/<domain>/domain.json   the CP domain spec
    pddl/<domain>/domain.pddl                     the `gt_am` reference

But `ROSAMEMixin._build_around` reads a **third pair**, through
`get_domain_model`, and the network cannot be constructed without it:

    <assets_root>/<domain>/domain_model.json      the DL head's vocabulary
    <assets_root>/<domain>/objects.json           its grounding universe

The first pair is static and checked in. **The second is not, and cannot be.**
Upstream's `objects.json` is a checked-in constant, because its corpora are
object-homogeneous. Ours is the per-run object union — precisely the quantity
Phase 1½ just pinned — so it has to be generated per run, into a run-scoped
directory.

Upstream's own `models/domains/` tree is therefore **not vendored**: we would
never read it. `root=None` preserves upstream's path for fidelity, but nothing
in our code takes that branch.

`src/milp/domain_assets.py:write_grounding_assets(domain_key, objects, root)`
writes that pair. It validates two things that would otherwise fail much later
and much less legibly: that every object type is declared in the domain PDDL, and
that the universe is non-empty (an empty one grounds nothing and yields a
zero-width head).

This cost **two vendor modifications**, both minimal and both additive:

- `dl/util/ROSAME/rosame.py:375` — `get_domain_model(domain, root=None)`.
  Upstream hardcodes a path next to the module; `root=None` keeps that exact
  behaviour, so nothing upstream-facing changes.
- `dl/mixins/action_model.py:10` — `domain_assets_root` threaded through
  `_build_around`.

Directories are keyed by **benchmark domain key** (`blocksworld`, `npuzzle`,
...) rather than upstream's names, which is what makes `Convertor`'s hardcoded
`"blocks" -> "blocksworld"` alias a no-op and leaves that file verbatim.

### 6.2 ICAPS-26 reorders predicate and schema arguments

`dl/util/ROSAME/rosame.py` stores a signature as a `{Type: count}` map rather
than an ordered list, and then does:

    self.params_types = sorted(params.keys(), key=lambda x: x.name)

`Predicate.ground` emits one `itertools.permutations` group per key in that
order. **A PDDL signature that is not already grouped and name-sorted is
therefore grounded as a permutation of itself**, and the DL head's proposition
list is not positionally comparable to the CP list built from the PDDL.

Measured on our five domains — signatures whose order differs:

| domain | reordered |
|---|---|
| blocksworld | **0** |
| depot | 10 |
| gripper | 2 |
| hanoi | 2 |
| npuzzle | 2 |

blocksworld being 0 is the trap: it is the domain every other gate in this
package uses, so a blocksworld-only check would show nothing.

Two consequences worth stating plainly.

**First, this is a divergence between the two upstreams, not a quirk of ours.**
AMLGym's ICAPS-24 fork carries the same line *commented out*:

    # self.params_types = sorted(params.keys(), key=lambda x: x.name)
    self.params_types = list(params.keys())

so it recovers PDDL order by accident of dict insertion order, and the 24 arms
never see the reordering while the 26 arms always do. `check_predicate`'s
permutation fallback is what absorbs this on the 24 side.
`benchmark/algorithm_adapters/test_check_predicate.py` passes green **for the
24 fork only** — its docstring previously implied the count map was
order-preserving, which is false, and has been corrected so nobody reads that
green as evidence the 26 arms need no permutation step.

**Second, the collapse is recoverable.** `rosame_argument_permutation` is a
*stable* argsort by type name, which keeps same-typed arguments in PDDL order.
That matters for a signature repeating a type at non-adjacent positions —
hanoi's `move_peg_disc(disc, peg, disc)` is exactly that, and I had earlier
asserted no domain had one. Because `ground` permutes rather than combines,
the two `disc` slots stay ordered and distinct, so the mapping is a bijection
and no argument identity is lost. Pinned as `[0, 2, 1]`.

Pinned in `src/milp/test_rosame_argument_order.py` (21 tests), which grounds a
real `Domain_Model` and checks `model.propositions` *and* `model.actions` —
both are needed, since gripper's reordering is schema-only and a
predicate-only test would have been vacuous there. Mutation-checked: replacing
the permutation with identity fails 12 tests and correctly leaves blocksworld
passing.

The link to gate 3: `trans_full_state` zips the DL vector against the CP
proposition list **positionally**. A permuted vector has the right width, so it
raises nothing — it just means different propositions. That is the failure the
Phase-2 head alignment exists to prevent, and it is why the alignment must be
by index and not by name.
