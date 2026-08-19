# Algorithm comparison: what each method assumes, and how each one fails

Working analysis of every arm in the benchmark — what it consumes, what it assumes, where its
strength comes from, and the specific ways it breaks. Evidence is from the image experiments
listed in `benchmark/evaluation/cfm/dashboard_config.yaml` unless stated otherwise.

Written to be read before designing an experiment or writing up a comparison table. The
failure-mode section (§3) is the part that matters most: two arms can share a `solving_ratio`
of 0.000 for completely different reasons, and the fix is different in each case.

---

## 1. The arms

| arm | state channel | actions | model search | GT used |
|---|---|---|---|---|
| **CDPS** | VLM-classified PDDL states (masked/noisy) | observed | conflict-directed patch search over trace repairs | init anchored |
| **PISAM_MILP_SR** | same | observed | one CP-SAT projection | init anchored |
| **PISAM_MILP_LOOP** | same | observed | iterative subset-sampled CP-SAT + PI-SAM | `init_only` / `none` |
| **ROSAME** (24, symbolic) | observed PDDL states | observed | gradient descent on relaxed schemas | — |
| **ROSAME-I** (24, imaged) | ResNet-18 over raw frames | observed | gradient descent, `argmax` read-out | init + final (γ anchor, soft) |
| **ROSAME-I_MILP** (ours: 24 net + 26 loop) | same ResNet-18 | observed | CP-SAT projection each epoch | init + final (hard) |
| **ROSAME-I_MILP_26** (planned) | AAE encoder + sigmoid symbol net | observed (adapted) | same loop, upstream-faithful | init + final (hard) |

The single most important axis is **what the state channel can see**. Everything else follows
from it.

## 2. Per-method pros and cons

### CDPS (ours)

**Pros.** Operates on symbolic observations, so any fluent the VLM names is visible to it —
including fluents that are hard to see in pixels. Repairs are explicit and auditable
(`conflict_free_models/`, patch accounting). Degrades gracefully: returns best-so-far on
timeout rather than failing.

**Cons.** Search, so it is budget-bound rather than convergent — on the hanoi image experiment
it hit `terminated_by: timeout_exceeded` in **30/30 cells** at 600 s, finding 1–19 conflict-free
models after 3.7k–14.3k nodes. Reported numbers are therefore a lower bound, and its curve can
*decline* with more trajectories (hanoi solving 0.50 → 0.70 → 0.50 across numtrajs) because the
tree grows against a fixed budget. Model edits being free also invites overfitting to a
conflict-free-but-wrong model (see `conflict-search-free-model-overfitting`).

### PI-SAM + MILP, single round

**Pros.** Fastest arm by a wide margin (0.6–2.8 s/cell on hanoi) and the strongest overall
there: P 0.725 / R 0.994 / solving 0.950. Converges rather than exhausting a budget.

**Cons.** One shot — no opportunity to revise once the projection is made. Inherits whatever
the VLM channel got wrong.

### PI-SAM + MILP, loop

**Pros.** Slightly more robust than SR on noisy sets; stops on `fixpoint` in 29–30/30 cells
rather than on a timeout, so its cost is data-determined, not budget-determined.

**Cons.** More expensive (1.3–61.8 s) for a small quality gain on our domains. The
`gt_anchoring` ablation shows the init anchor buys almost nothing here: `init_only` vs `none`
differ in 14/30 hanoi cells and only in repair cost (83→77, 92→86), not in final metrics
(P 0.723 vs 0.721). Useful as an ablation, weak as a headline.

### ROSAME (24, symbolic)

**Pros.** The apples-to-apples symbolic baseline; no vision confound.

**Cons.** On depot it produced `unsolvable_ratio` 0.433 — over-constrained models that prove
goals unreachable. Reasonable P/R (0.823/0.849) with solving 0.000.

### ROSAME-I (24, imaged)

**Pros.** The published competitor, learning end-to-end from pixels with no symbolic channel at
all. That is a genuinely harder and more ambitious setting than ours.

**Cons.** In our low-data regime it collapses — see §3.1. `solving_ratio` 0.000 in **all five
domains, 150/150 cells**.

### ROSAME-I + MILP (ours)

**Pros.** The MILP rescues the collapse: hanoi P 0.835 / solving 0.850 against its parent's
0.533 / 0.000. Highest precision of any arm on hanoi. Strong evidence that the projection, not
the network, is doing the work.

**Cons.** Still bound by what the pixels contain — see §3.2 (gripper) and §3.3 (depot). It is
also *not* a port of ICAPS-26: it is ICAPS-24's network inside ICAPS-26's MILP loop, and should
be labelled that way.

## 3. Failure modes — the section to read twice

Three distinct mechanisms have produced `solving_ratio = 0.000` in this benchmark. They look
identical in a table and require completely different responses.

### 3.1 Optimisation collapse — "empty effects" (ROSAME-I, no MILP)

**Symptom.** The learned PDDL has `:effect (and )` on most actions. Empty/total: hanoi 4/4,
gripper 3/3, npuzzle 1/1, depot 6/7, blocksworld 2/4.

**Metric signature.** Effect *precision* ≈ 1.000 with effect *recall* ≈ 0.000–0.044 — precision
is vacuous when you predict nothing. Precondition recall stays healthy (0.25–0.77).

**Mechanism.** The consistency term `MSE(domain_preds[:-1], preds[1:-1])` is trivially minimised
by `add = del = 0` — identity dynamics — plus a CV head that predicts near-identical vectors for
consecutive frames. That is a genuine optimum, reachable with no learning. Nothing in the loss
pushes effects up, while `lambda_ * MSE(pre, 1)` actively pushes preconditions up. The
asymmetry *is* the precondition/effect split above.

**Why it is a collapse and not a limit.** The information is present; the optimiser found a
degenerate basin. More data, mini-batching instead of per-trajectory training, or a bounded
state range would plausibly escape it. This is fixable.

**Detection.** Assert the learned model has ≥1 add-or-delete effect across all schemas. One
line, and it would have caught 150 void rows immediately.

### 3.2 Identifiability limit — "static fluent" (ROSAME-I + MILP, gripper)

**Symptom.** `solving_ratio` 0.000 with **`false_plans_ratio` 1.000** — the planner finds a plan
for every test problem and every one fails validation. Nothing unsolvable, nothing timed out.
The model is otherwise excellent: precondition recall **1.000**, effect precision **1.000**.

**The concrete case.** Learned vs GT gripper:

```
GT   pick: … (not (at ?b ?r))  (not (free ?g))   (carry ?b ?g)
ours pick: … (not (at ?b ?r))                    (carry ?b ?g)      ← missing (not (free ?g))

GT   drop: pre (at-robby) (carry)          eff (at ?b ?r) (free ?g) (not (carry ?b ?g))
ours drop: pre (at-robby) (free ?g) (carry) eff (at ?b ?r)          (not (carry ?b ?g))
```

The model **never changes `free`**. One gripper can therefore carry unboundedly many balls;
plans are valid under the model and rejected by VAL every time.

**Mechanism.** The encoder hard-fixes exactly two states — the GT init and the GT final. In
gripper, `free(g)` is **true in both** (the robot starts and ends empty-handed). Consider:

* `M_true` — `pick` deletes `free`, `drop` adds it. `free` is false in the interior.
* `M_static` — neither action touches `free`. `free` is true everywhere.

Both satisfy the hard init. Both satisfy the hard goal. Both are consistent with the observed
action sequence. **They differ only in the interior value of `free`** — which is soft, driven by
the CV head's per-frame probabilities. If the head has no reliable signal for "the gripper is
holding something" (a subtle cue at 64×64), the objective pays nearly the same for both and the
model prior breaks the tie arbitrarily.

**Why it is a limit and not a bug.** No amount of extra training, better optimisation, or a
bigger network fixes it, because **two different action models generate the same observations**.
Escaping it requires new information, not better search:

1. a frame where the fluent is visually unambiguous;
2. traces where the fluent differs at an *anchored* endpoint (not just in the interior);
3. a structural prior preferring models that consume their preconditions.

**Why our arms don't suffer it.** PI-SAM reads `free(g:gripper)` as an explicit predicate from
the VLM classifier (`llm_gripper_fluent_classifier.py:103`) and scores R 1.000 / solving 1.000
on gripper. The information is in the symbolic channel and not in the pixel channel. **This is
a real contribution claim** — a named-predicate observation channel is strictly more
identifiable than a pixel channel for fluents with no visual signature — and it is the cleanest
argument in the benchmark for why our pipeline exists.

**Prediction.** The planned ICAPS-26 arm will *not* fix gripper unless its AAE encoder genuinely
extracts `free` better. If it does fix it, that itself is the interesting result.

### 3.3 Encoding exclusion — constraints that rule out the truth (depot)

**Symptom.** Degraded across the board — depot ROSAME-I_MILP: R_precs 0.760, R_eff_pos 0.550,
and P_eff_pos **0.619** (spurious effects, unlike gripper), `false_plans_ratio` 1.000.

**Mechanism.** `src/milp/vendor/UPSTREAM.md` documents it, naming depot:

> *No redundant adds* (`StepAddPre`) … With observed actions this can make the ground-truth
> model infeasible in domains with legal redundant adds (e.g. depot's `drop` adding
> `(at ?p ?d)` after `lift`, which does not delete it).

**And an asymmetry we must declare.** `ROSAME-I_MILP` runs `MilpEncodingConfig.upstream()`
(`rosame_i_milp_runner.py:88` — `forbid_redundant_adds`, `delete_implies_precondition`,
`schema_nonempty` all ON), while our PI-SAM MILP runs `cdps_dialect()`
(`milp_denoiser/config.py:446` — all OFF). On depot that is exactly the constraint that can
exclude the GT model. Part of the depot gap therefore measures an upstream modelling
assumption, not the algorithms.

**Required before depot goes in a table.** Re-run depot's `ROSAME-I_MILP` with
`MilpEncodingConfig.tag()` (`forbid_redundant_adds=False`) and report how much of the gap
closes. Fidelity to upstream is defensible; an undeclared asymmetry is not.

## 4. Reading the metrics correctly

| pattern | means | look at |
|---|---|---|
| eff-precision ≈ 1, eff-recall ≈ 0 | empty predicted set (§3.1) | the learned PDDL |
| `false_plans` = 1.0, `unsolvable` = 0 | over-permissive model — missing delete or precondition (§3.2) | diff against GT domain |
| `unsolvable` high | over-constrained model | precondition precision |
| all four ratios 0 | **model failed to parse** — `syntax_errors` in `problem_solving`, a field our result schema drops | the model file |
| `learning_time` pinned at the timeout | budget-bound, not converged | `terminated_by` |

The fourth row is worth fixing in the schema: `amlgym.problem_solving` returns a
`syntax_errors` ratio that `evaluate_model` discards, so a model that unified-planning cannot
parse is indistinguishable from one that simply solved nothing.

## 5. Open items

1. Re-run all ROSAME-I rows once §3.1 is addressed — the current 150 are void as a fair baseline.
2. Depot ablation with `forbid_redundant_adds=False` (§3.3).
3. Is CDPS timeout-limited or capability-limited? One cell at a much larger budget answers it.
4. Surface `syntax_errors` in `result_schema.py`.
5. Add the degenerate-model guard (§3.1) to every learner's output path, not just ROSAME-I's.
