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

---

## 4. The remaining work, to completion

Phase numbers below are **the plan doc's** (`docs/rosame-i-milp-26-implementation-plan.md`
§10), not a second scheme. Step 1 above was preparatory and sits outside that
table. Phase 0 (the `Resize(64)` fix + the resize A/B) is already **CLOSED** —
shipped, run on 4 domains × 30 paired cells, hypothesis **refuted**
(analysis §5.3).

### Step 2 — re-run both imaged 24 arms with `--force`

Not part of the plan's phase table; a debt from the `train_per_trajectory`
deletion (`3a9d1b67e`). Every existing `ROSAME-I_24` / `ROSAME-I_MILP_24` row was
produced by the per-trajectory schedule, which is *not* what ICAPS-24 `main/`
does. The migration renamed those rows; it did not make them upstream-faithful.

- [ ] re-run `rosame_i_24` and `rosame_i_milp_24` with `--force` across the image
      experiments
- [ ] rebuild the dashboard; confirm the two series moved (if they did not, say so
      — a null result here is still a result)

Until this lands, the 26 arm has no honest 24 baseline to be measured against,
which is what Phase 4 exists to do.

### Phase 1 — vendor + assets + parity tests *before* any adapter

- [ ] vendor `dl/` + `convertor/` + `util/model_perm.py` (§2.1 fixes the exact
      boundaries)
- [ ] **also vendor `main/`** — our extension, not in the plan's table. It is what
      lets `rosame_i_24` get an upstream-parity test of the kind
      `test_po_rosame_runner.test_local_loop_matches_vendored` already gives
      `rosame_24`. Without it, step 2's re-run is unpinned.
- [ ] generate all five domain specs + `pddl/<domain>/domain.pddl` from **our**
      domains (§5) — not copied from upstream
- [ ] import-only smoke test
- [ ] **write verification gates 1–3 now, against the vendored code, before the
      adapter exists** (§9). Three of the five errors an external review found in
      the plan's first draft are things these catch mechanically:
  - [ ] **gate 1 — shape/parity**: `z` is `[B,T,S]`, `a_logit` `[B,T+1,adim]`,
        `z_suc_aae` / `p_applicable` `[B,T+1,adim,S]`, `state_traces` `[B,T+2,S]`,
        `z ∈ [0,1]`; assert `T = N−1` explicitly; assert a 2-image trace is
        **rejected**, not silently `T = 0`
  - [ ] **gate 2 — loss parity**: tiny synthetic trace through vendored
        `dl/model.py` and through our harness, equal to float tolerance. Cover
        **both** `loss_pseudo_a` regimes (no MILP labels; some MILP labels). This
        is the test that actually protects the port.
  - [ ] **gate 3 — MILP parity**: one `run_fixer` call matches a direct
        invocation of the vendored translator; the §0.1 identity mappings reach
        `extract_sol_label` / `extract_sol_model` (**not** that
        `model_permutation` returns identity — it would not); `trans_full_state`'s
        zip is index-aligned with the DL symbol vector; `state_label` /
        `action_label` shapes match for a **ragged** bundle; the `chosen = 0`
        action fallback never fires

### Phase 1½ — PIN THE GROUNDING SCOPE  ← **blocking, decision only**

**No code for `rosame_i_26` or `rosame_i_milp_26` may be written until this is
settled** (plan §4.2a, open decision 19).

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

- [ ] pin it. It sizes the DL symbol head, `extract_sol_label`'s width and the
      `loss_pseudo_s` BCE, so it is **not** deferrable to Phase 5 — discovering
      it there means rebuilding the adapter and discarding anything already
      measured
- [ ] either way: take the union over the whole `data_dir`, **not the fold**.
      Measured — all 30 blocksworld cells ground at `n_props = 36`, but by
      composition (every fold happens to include a 5-block problem), not by
      construction. A 4-block-only fold would ground at 25 and the grid would
      average two vocabularies. Same argument and same cache key as §4.4's
      normalisation statistic
- [ ] **equivalence gate**: solve one 4-block blocksworld trace under both
      groundings, assert the recovered model is identical. The claim that the
      phantom propositions are inert (pinned false by hard init + frame axioms,
      decoupled from the lifted schema variables) is a *prediction*; if it
      fails, the choice is a measurement decision, not a fidelity one
- [ ] register the outcome in the deviation register (§8) if per-problem wins

### Phase 2 — the data adapter  ← **the real work**

- [ ] fold → their contract (§4.1–4.4). Risk: **medium, the bulk of the effort**
- [ ] resolve the one-frame-too-long mismatch (§4.1): our traces carry N+1 images,
      theirs N−1
- [ ] proposition space is upstream `Instance`, **no repeated args** (§4.2) —
      unlike our `RepeatedArgsInstance`
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
| grounding scope | per problem | **undecided — Phase 1½** |
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
- [ ] must pass gate 3

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
