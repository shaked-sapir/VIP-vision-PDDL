# ROSAME training convergence fix

Stop the ROSAME arms on **training-loss convergence**, with a fixed epoch
ceiling behind it and the cell timeout as the outer bound. Today the
symbolic arms in the large-corpora sweep stop on ROSAME/MILP agreement (the
MILP arms) or on a hardwired 100 epochs (the DL-only arm); neither is a
convergence criterion.

Status: **plan only — nothing implemented.**

Every result tree on disk (`large-corpora__L-sweep__v2`, `__v3`, the
`epprobe__*` runs) predates commits `4997d76fa` (base-loss normalisation where a
pseudo-label CE competes with it) and `c5e43f8c0` (batched optimizer steps).
Their loss curves do not reflect the current training dynamics and are **not**
used anywhere in this plan. The constants below are starting points; the probe
in §6 is where they get measured.

---

## 0. Scope

**In:** the three symbolic arms of the large-corpora sweep — `rosame_24`,
`rosame_milp_24`, `rosame_milp_24_tag`. This is the deliverable.

**Also in, code only:** the four imaged arms get the same rule and the same
stop-reason vocabulary (§5), so that "trained to convergence" means one thing
across the paper. Nothing imaged is re-run by this task; existing image rows
were produced under fixed epochs and stay as they are.

**Out:** the warmup. `pre_mip_epochs` stays a constant 50, as in the upstream
ROSAME repository. Making the warmup itself convergence-driven is not part of
this change.

---

## 1. What the code does today

### 1.1 `ROSAME_24` has no convergence signal

`benchmark/run_config_large.yaml` does not set `train_per_trajectory`, so the
runner default (`True`) applies: `PORosame_Runner.learn_per_trajectory` trains
each trace with a fresh Adam for 100 epochs, weights carried over. The recorded
per-step "loss" is a saw-tooth of per-trace curves, not one training curve —
there is nothing to converge on. `RosameBaselineRunner.learn` also does not
accept `epochs`; 100 is hardwired through `learn_full`'s default.

### 1.2 The MILP arms stop on agreement, and agreement is trivially reached

`MilpPORosame.learn_pooled_with_milp`
(`benchmark/algorithm_adapters/rosame_milp/milp_loop.py`) trains 50 warmup
epochs, solves, and returns as soon as `model_agreement >= agreement_stop`
(default 1.0). The MILP objective pulls its solution toward the network's own
argmax (`obs_m`), so the first round already agrees. In practice these arms
train the warmup and stop.

### 1.3 The 1 h budget is not enforced on the symbolic arms

`run_fold._run_baselines` passes `timeout_seconds=conflict_search_timeout`
(3600 in the large config) to every runner's `learn`. Both symbolic runners
ignore it. Only `RosameIMilpRunner` has the `timeout_check` / `seconds_left`
pattern (`benchmark/baselines/rosame_i_milp_runner.py:288`). Nothing external
kills a slow arm.

### 1.4 A convergence rule exists, but only for the ICAPS-26 imaged arm

`has_converged` (`src/milp/rosame26_budget.py:332`) is a windowed
relative-plateau rule with `CONVERGE_WINDOW=40`, `CONVERGE_MIN_IMPROVEMENT=0.002`,
`CONVERGE_PATIENCE=3`, `CONVERGE_MIN_EPOCHS=60`, and `_BestModelTracker`
(`benchmark/baselines/rosame26_runner.py:78`) checkpoints the schema heads at
the best loss and restores them at the end. `Rosame26Trainer` exposes a
`stop_check` hook; `docs/large-corpora-experiments-plan.md` §4 says the rule
"needs switching on", which is true only for that arm. The symbolic loops
(`learn_pooled`, `learn_pooled_with_milp`) have no hook.

---

## 2. Stopping rules after the change

Any-of, first satisfied wins, in every ROSAME arm:

| rule | stop reason | role |
|---|---|---|
| training loss has plateaued (§3) | `converged` | the main criterion |
| `epochs` reached | `epochs_exhausted` | ceiling; should bind rarely |
| fold learning budget spent (`timeout_seconds`) | `timeout` | outer bound; checked between epochs, so a solve in flight can overrun by at most one solve |

Removed as a rule: agreement. `agreement_stop` becomes `Optional[float]`, `None`
= off, and is `null` in the large config. Agreement stays **recorded** per round
(`milp_rounds[].agreement`) for the dashboard's convergence panel.

The emitted model is the **best-loss checkpoint**, not the final epoch's
(the CONVERGE behaviour of the 26 arm, applied everywhere). For the MILP arms
"best" is measured on the post-first-solve series (§3.2).

---

## 3. The rule, and what each parameter means

The rule reads the per-epoch **training** loss, where one epoch is one pass
over every pooled transition. It never reads test data. It is `has_converged`
as it stands, moved to a neutral module (§4.1).

- **`window`** (epochs). The loss series is cut into consecutive blocks of this
  many epochs; each block is summarised by its lowest loss. This is the noise
  filter: a single epoch can tick up from batch shuffling, so single epochs are
  never compared. Larger = more smoothing, slower reaction (decisions happen
  only at block boundaries).

- **`min_improvement`** (fraction). A block counts as progress only if its
  lowest loss beats the *running best* — the lowest loss over all earlier
  blocks — by at least this fraction of that best. Relative, not absolute, so
  one value serves domains whose loss scales differ. Too high stops while the
  curve is still descending; below the noise level lets random dips reset the
  count and prolong training (bounded by the timeout, never past it).

- **`patience`** (blocks). Consecutive blocks without progress required to
  stop. `window × patience` is the plain-language quantity: **the number of
  epochs the loss may go without a meaningful improvement before training
  stops.** At least 2, because one flat block happens mid-descent.

- **`min_epochs`** (epochs). A floor before the rule may fire, so a flat start
  cannot end the run. For the MILP arms it is counted **from the first
  successful solve** (§3.2).

- **`epochs`** (epochs). A ceiling only. High enough that runs end by
  convergence or by time.

- **`timeout_seconds`**. The fold's learning budget (1 h in the large config).
  Checked between epochs; also caps each MILP solve at the seconds still left.

### 3.1 Starting values

| parameter | symbolic arms | imaged arms | combined meaning |
|---|---|---|---|
| `window` | 10 | 40 | decisions every 10 / 40 epochs |
| `patience` | 3 | 3 | stop after 30 / 120 epochs without progress |
| `min_improvement` | 0.002 | 0.002 | progress = beating the running best by 0.2% |
| `min_epochs` | 50 (DL-only); 30 after the first solve (MILP) | same | at least the old warmup; at least 30 pseudo-label epochs |
| `epochs` | 2000 | per-domain table (24) / preflight projection (26), as ceilings | binds rarely |
| `pre_mip_epochs` | 50 | 50 | upstream constant, unchanged |

Why the window is the one constant that differs: it is the noise filter, and
the noise regime is set by what is trained. The symbolic arms fit small schema
MLPs over hundreds of pooled transitions per step; the imaged arms fit a CNN
(24) or ResNet (26) on 3-9 traces with tiny batches, and the 26 arm's 40 was
tuned because a 20-epoch window let one transient excursion fill the whole
patience buffer. The two imaged families share the imaged value until the probe
says otherwise. Everything else is identical by design, so the criterion means
one thing across arms.

### 3.2 Where the MILP arms' series starts

When the pseudo-label cross-entropy joins the loss at the first successful
solve, the logged loss shifts scale. A rule comparing post-solve losses against
the pre-solve running best would never see an improvement and would fire
`window × patience` epochs later, without the pseudo-label phase having
trained. So for every MILP arm the convergence series — running best,
windows, `min_epochs`, and the best-checkpoint tracker — starts at the first
successful solve. The warmup epochs are recorded (snapshots, loss) but not
scored. Base loss and CE are recorded as separate fields so the jump is
visible in the convergence panel.

---

## 4. Changes to make

### 4.1 Extract the rule: `src/milp/loss_convergence.py` (new)

Move `window_best`, `relative_improvements`, `has_converged` out of
`rosame26_budget.py`. Add a frozen `LossConvergenceRule` dataclass
(`window`, `min_improvement`, `patience`, `min_epochs`) with `from_config` and
`as_stats`. `rosame26_budget.py` imports from here and keeps its constants, so
the 26 arm is untouched.

### 4.2 Share the best-model checkpoint

Move `_BestModelTracker` from `rosame26_runner.py` to a shared adapter module
(`benchmark/algorithm_adapters/best_checkpoint.py`). Both networks expose
`action_schemas` as modules, so the same state-dict snapshot and restore works
for the 24 schema MLPs. Add `reset()` so the MILP arms can restart the tracker
at the first solve.

### 4.3 Hooks in the pooled loop: `po_rosame_runner.learn_pooled`

Gains `stop_check: Callable[[List[float]], bool]` (called with the loss history
after each epoch) and `timeout_check: Callable[[], bool]`. Returns a small
`TrainingReport` (epochs run, per-epoch losses, stop reason, best epoch)
instead of a step count. `learn_full` forwards both and returns the report
alongside the PDDL. `learn_per_trajectory` is left as is; it is no longer the
large-sweep schedule.

### 4.4 Rework `milp_loop.learn_pooled_with_milp`

Same two hooks plus `seconds_left` to cap each solve at the remaining budget;
`agreement_stop: Optional[float] = None`; the convergence series starts at the
first successful solve (§3.2). Report gains `stop_reason` in the §2
vocabulary, `epochs_run`, `first_solve_epoch`, `best_epoch`, `best_loss`.
Snapshot records gain `base_loss` and `ce_loss`.

### 4.5 Wire the runners

`RosameBaselineRunner`: accept `epochs` and a `LossConvergenceRule`; honour
`timeout_seconds`; run the tracker; restore the best checkpoint before
`rosame_to_pddl`; write `stop_reason`, `epochs_run`, `best_epoch`, `best_loss`
and `rule.as_stats()` into the result row.

`RosameMilpRunner` (and `_TAG` through it): the same, plus the budget callbacks
into the loop, `agreement_stop` forwarded as `None` by default. The
`RosameMilpBaseRunner.learn` one-shot path gets the same treatment for its DL
phase.

### 4.6 Config plumbing

`benchmark_runner._RUNNER_KWARG_KEYS` rejects unknown keys, so add
`agreement_stop` and a `rosame_convergence:` block (parsed into
`LossConvergenceRule`; a flat dict is fine for `_instantiate`).

`benchmark/run_config_large.yaml`:

```yaml
train_per_trajectory: false     # pooled: one persistent optimizer, one loss curve
epochs: 2000                    # ceiling only
agreement_stop: null            # recorded, not a stop rule
rosame_convergence:
  window: 10
  min_improvement: 0.002
  patience: 3
  min_epochs: 50                # MILP arms count this from the first solve
```

and a new `run_name`: the `ROSAME_24` rows change schedule, so nothing is
comparable with v3.

### 4.7 Tests

- Rule, on synthetic curves: a noisy plateau stops at the expected block; a
  steady descent does not; a post-solve jump does not fire early when the
  series is restarted.
- Loop, with a fake `milp_round`: each stop reason; agreement off; the series
  restarts at the first *successful* solve, not the first attempted one;
  solves are capped by `seconds_left`.
- Runner: kwargs reach the loop; the emitted model is the best-epoch
  checkpoint; the result row carries the new fields.
- The existing `test_po_rosame_runner.test_local_loop_matches_vendored` must
  still pass: hooks default to `None` and change nothing when absent.

### 4.8 Docs and dashboard

`docs/large-corpora-experiments-plan.md` §4, the loop's `IMPLEMENTATION.md`,
and the loop line in `CLAUDE.md`. `benchmark/evaluation/cfm/convergence.py`
reads the curves unchanged; marking the stop epoch and the first-solve epoch
on the panel is a cheap addition.

### 4.9 Cluster sizing

The per-fold job (`scripts/cluster_large/sweep_fold.sbatch`, 24 h) now has to
hold three ROSAME arms × five L values at up to 1 h each, plus the two PI-SAM
loops and evaluation. Typical stops will be well under the hour, but the worst
case needs either a longer `--time` or a lower per-arm ceiling at L=2000.
Decide after the probe reports seconds per epoch at L=2000.

---

## 5. The imaged arms

Same rule, same knobs, same stop-reason vocabulary; only the window differs,
and it differs by input kind, not by paper year.

| arm | key | loop | today | after |
|---|---|---|---|---|
| ROSAME_24 | `rosame_24` | `learn_pooled` | fixed 100 epochs, per-trajectory | rule, symbolic constants |
| ROSAME_MILP_24 | `rosame_milp_24` | `learn_pooled_with_milp` | agreement, else 100 | rule from first solve, symbolic constants |
| ROSAME_MILP_24_TAG | `rosame_milp_24_tag` | same | same | same |
| ROSAME-I_24 | `rosame_i_24` | `rosame_i_runner.learn_full`, 3 seeds | per-domain fixed epochs (70-300); timeout skips later seeds | `stop_check` hook, imaged constants, table = ceiling |
| ROSAME-I_MILP_24 | `rosame_i_milp_24` | `milp_loop_i.learn_pooled_with_milp`, 1 seed | agreement, else fixed; timeout between epochs | rule from first solve, imaged constants |
| ROSAME-I_26 | `rosame_i_26` | `Rosame26Trainer` | `budget_mode` preflight / fixed / converge | `converge` uses the shared module; `min_epochs` 60 → 50 |
| ROSAME-I_MILP_26 | `rosame_i_milp_26` | same | same | same, series from first solve |

Seeds: each seed converges on its own and keeps its best-loss checkpoint; the
seed with the lowest best loss is emitted. The between-seed budget check
already exists.

Code lands with this change so the vocabulary is consistent; a re-run of the
image grid under convergence gets a new experiment name.

---

## 6. Calibration probe

One job per domain, per input kind, rule **off**, `snapshot_interval: 1`,
via `scripts/cluster_large/epoch_probe.sbatch` extended with the arm list and
`train_per_trajectory: false`:

1. Run the DL-only arm and the MILP arm for a fixed 500 epochs at L=10, 100
   and 2000, one fold each.
2. From each recorded loss series read: the epoch where the loss comes within
   0.5% of its eventual minimum; the median relative epoch-to-epoch uptick
   against the running best; the size of the jump at the first solve.
3. Set `min_improvement` above the uptick band; set `window` so a block spans
   several upticks; check `window × patience` exceeds the longest flat stretch
   that later resumed descending.
4. Replay `has_converged` offline over the recorded series with the chosen
   values; confirm the stop epoch lands after the plateau and the restored
   best-epoch model equals the full run's best.
5. Read seconds per epoch at L=2000; confirm `window × patience` epochs cost a
   small fraction of the hour and the ceiling is unreachable there before the
   budget.

Expected outcome: the symbolic and imaged probes agree on everything except
the window.

---

## 7. Pilot gate

One domain, L=10 and L=2000, rule on with the probe's values,
`snapshot_interval: 1`. Pass when every ROSAME row reports `converged`, the
stop epoch sits after the plateau on the recorded curve, and the L=2000 arms
finish inside the hour. Then the full sweep under the new run name.
