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

**Where it stands.** Step 1 done and landed. Step 2 (the imaged-24 re-run) done
but the dashboard is stale. Phase 1 substantially done and **uncommitted**;
Phase 1½ **done** — decided and its equivalence gate run and green, which
unblocks Phase 2. 150 new tests; full suite **581 passed, 1 skipped** from a
432-passed baseline. Phase 2 is next and has not started. Two Phase-1 findings
amend the plan — see §6.

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

### Step 2 — re-run both imaged 24 arms with `--force`  ← **DONE, dashboard stale**

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
- [ ] rebuild the dashboard; confirm the two series moved (if they did not, say so
      — a null result here is still a result). **Not done.**
      `benchmark/running_results/results_dashboard.html` was built at 01:27, and
      **five of the seven experiment directories finished writing after that** —
      the last at 04:22. It is currently a mix of re-run and pre-re-run numbers
      and must not be read until rebuilt.

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

All of it is still **uncommitted** (`git status` shows every path untracked);
nothing here has been reviewed or landed.

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
- [ ] ~~register the outcome in the deviation register (§8) if per-problem
      wins~~ — moot; per-problem was not chosen. What §8 *should* get instead is
      the opposite entry: that on our object-heterogeneous corpora the shared
      instance carries phantom propositions the 24 arm does not, and that this
      is a fidelity choice made knowingly.
- [ ] correct the plan doc's Phase-4 table row for `grounding scope`: it reads
      `undecided`, and its 24-arm cell is wrong besides (the 24 arm grounds per
      problem *and then drops surplus union columns*, which is not the same
      thing as per-problem grounding). Mirrored in §4 above.

### Phase 2 — the data adapter  ← **the real work**

- [ ] fold → their contract (§4.1–4.4). Risk: **medium, the bulk of the effort**
- [ ] resolve the one-frame-too-long mismatch (§4.1): our traces carry N+1 images,
      theirs N−1
- [ ] proposition space is upstream `Instance`, **no repeated args** (§4.2) —
      unlike our `RepeatedArgsInstance`
- [ ] **head alignment through `rosame_argument_permutation`** — new, §6.2 below.
      Not in the plan. Map CP proposition index → DL head index; aligning by
      predicate name alone is silently wrong on depot, gripper, hanoi and
      npuzzle. Gate 3 shows why this cannot be caught downstream:
      `trans_full_state` zips positionally, so a permuted vector of the right
      width produces the wrong propositions and no error
- [ ] grounding assets (`domain_model.json` + `objects.json`) written per run
      into a scoped root via `write_grounding_assets`, §6.1 below
- [ ] image normalisation computed once over the whole `data_dir` (§4.4, DECIDED)
- [ ] resize per-domain configurable, default `Resize(64)` (§4.6, DECIDED)
- [ ] must pass gate 1

### Phase 3 — harness replacement

- [ ] override one method, keep everything else (§3)
- [ ] pin the run-level settings explicitly (§1.3), including **augmentation
      disabled** (the 24 arm horizontally flips on blocksworld)
- [ ] batching built for any N, not just today's N (§1.1)
- [ ] must pass gate 2

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
