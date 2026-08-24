# ROSAME-I (26): the MILP is silently inert on three domains

Found while reading Phase 6's full grid (300 rows, 5 domains, both 26 arms).
This document records the diagnosis and the options; the decision is open.

---

## 1. The symptom

`ROSAME-I_26` and `ROSAME-I_MILP_26` came out **byte-identical on all 30 cells**
of gripper and hanoi — same precision, same recall, same solving ratio. The MILP
was contributing nothing at all.

It is not a null result. Every solve on those domains fails:

| domain | OPTIMAL | UNSATISFIABLE | usable |
|---|---|---|---|
| blocksworld | 7605 | 0 | 100% |
| npuzzle | 7290 | 0 | 100% |
| depot | 768 | 6522 | **11%** |
| gripper | 0 | 7209 | **0%** |
| hanoi | 0 | 6642 | **0%** |

`UNSATISFIABLE`, **not** a timeout: CP-SAT *proves* no model exists, in
0.2–1.1 s. A timeout would have returned `UNKNOWN` and been visible in the solve
time; this is a genuine infeasibility, which is why it went unnoticed.

**Consequence for reporting.** `ROSAME-I_MILP_26` on gripper and hanoi is a
DL-only run wearing a MILP label, and depot's is 89% so. Those rows must not be
reported as MILP results.

---

## 2. The cause, isolated

Disabling each upstream rule in turn, on real gripper and hanoi folds:

```
upstream (all on)                   UNSATISFIABLE
schema_nonempty = NONE              OPTIMAL          <- this one, alone
forbid_redundant_adds = False       UNSATISFIABLE
delete_implies_precondition = False UNSATISFIABLE
```

`schema_nonempty` is the sole cause; the other two rules are irrelevant here.

Narrowing further — it is the **`add >= 1` half**, not the precondition half:

```
PRE_AND_ADD                    UNSATISFIABLE
ADD only (drop the pre half)   UNSATISFIABLE   <- still fails
NONE                           OPTIMAL
```

### What the rule is

`src/milp/encoder.py:215-222`, per action schema:

```python
sum(pre[a, p, x] for all p, x) >= 1    # every schema has >=1 precondition
sum(add[a, p, x] for all p, x) >= 1    # every schema has >=1 add effect
```

Plainly: **no action may be learned as a no-op.**

The code comment beside it matters: **"NOT in the paper — upstream code extra."**
It is present in the released ICAPS-26 implementation and absent from the
paper's formulation, so it is an inherited modelling choice rather than a
published claim. `encoding_config.py` already warned that it "can make the GT
model infeasible", which is why the CDPS dialect drops it.

### Which schemas cannot satisfy it

Adding `add >= 1` one schema at a time against a real solved encoding:

| domain | schema with no feasible add effect |
|---|---|
| gripper | `move` |
| hanoi | `move_disc_disc`, `move_disc_peg` |

### Why — and two hypotheses that were wrong

Both were checked and rejected, and they are recorded so they are not
re-proposed:

* **"Schemas absent from the sampled traces are forced empty."** False. gripper
  and hanoi use **every** schema in their traces; only depot has an unused one
  (`stack`).
* **"The GT models violate the rule."** False. All five GT domains satisfy
  `PRE_AND_ADD`, and the affected schemas have real GT add effects — gripper's
  `move` adds `(at-robby ?to)`; hanoi's `move_disc_disc` adds `(clear-disc ?from)`
  and `(on-disc ?disc ?to)`.

**The actual mechanism is the rule *combined with the observations*.** The
network's predicted interior states, hard-anchored at both endpoints by the GT
init and goal, admit no consistent assignment in which those schemas add
anything. The observations say "nothing became true across this transition"; the
constraint says "something must have". Hence a domain- and data-dependent
infeasibility rather than a static incompatibility — and hence one that a
different fold, or a better-trained network, might not exhibit.

---

## 3. A second, larger failure: the models have almost no effects

Independent of the UNSAT issue, and affecting **every** arm including the ones
whose MILP works. This is why `ROSAME-I` solves nothing.

Splitting the metrics separates it immediately (`ROSAME-I_26`, 150 cells):

| domain | precision (pre) | precision (eff) | **recall (eff)** | solving |
|---|---|---|---|---|
| blocksworld | 0.896 | 0.739 | **0.279** | 0 |
| depot | 0.397 | 0.619 | **0.023** | 0 |
| gripper | 0.421 | 0.723 | **0.022** | 0 |
| hanoi | 0.869 | 0.817 | **0.024** | 0 |
| npuzzle | 1.000 | 0.622 | **0.767** | **0.533** |

Effect recall is **2–3%** on three domains. The one domain that recovers effects
is the only one that solves anything.

The learned models confirm it directly:

| domain | schemas with **zero** effects | median effects/schema | GT effects/schema |
|---|---|---|---|
| blocksworld | 42% | 1 | 4.5 |
| depot | 27% | 2 | 4.6 |
| gripper | 48% | 1 | 2.7 |
| **hanoi** | **69%** | **0** | 4.0 |
| npuzzle | 0% | 6 | 4.0 |

**hanoi's median schema has no effects at all.** An action that changes nothing
can never appear usefully in a plan, so the domain is unplannable however good
its preconditions look (0.869 precision).

This also explains why the *precision* numbers look respectable: the arm asserts
very little and is right about the little it asserts. That is abstention, not
accuracy — the same trade gate 7 exposed on blocksworld, where 0.94 precision at
131 epochs came with two of four schemas completely empty.

**Note gate 4 does not catch this.** It rejects a model only when *every* schema
is empty; a model with 69% empty schemas passes.

### The cause: undertrained, not data-starved

Two tests, both free — the data was already on disk.

**Test 1 — does training longer add effects? Yes, decisively.** Same cell
(`fold0_numtrajs3`), gate 7's controls:

| domain | epochs | total effect literals | empty schemas |
|---|---|---|---|
| blocksworld | 161 | 7 | 1 |
| blocksworld | **5000** | **10** | **0** |
| depot | 131 | 19 | 2 |
| depot | **5000** | **24** | **1** |
| hanoi | 131 | **2** | 2 |
| hanoi | **5000** | **16** | **0** |

hanoi recovers **8× more** effect literals, and empty schemas fall to zero on
every domain.

**Test 2 — does more data add effects? No. Perfectly flat.** Effect recall by
trajectory count, all 150 cells:

| domain | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| blocksworld | 0.282 | 0.266 | 0.250 | 0.282 | 0.298 | 0.298 |
| depot | 0.028 | 0.000 | 0.000 | 0.042 | 0.042 | 0.028 |
| gripper | 0.066 | 0.066 | 0.000 | 0.000 | 0.000 | 0.000 |
| hanoi | 0.072 | 0.000 | 0.024 | 0.024 | 0.000 | 0.024 |
| npuzzle | 0.700 | 0.600 | 0.900 | 1.000 | 0.700 | 0.700 |

Nearly doubling the traces changes nothing on any domain.

**So the arm needs epochs, not traces** — and the §1.2 pre-flight floors it at
131 to fit the 600 s cell budget every arm shares. Gate 7 was written to keep
exactly these two apart ("so 'underperforms at the budget' and 'undertrained at
it' stay distinguishable"); the answer is **undertrained**.

The tension is real and should be stated rather than resolved silently: the
equal-budget rule is what makes the comparison fair, and at that budget the 26
arm never learns effects, so what is being compared cannot plan. Meanwhile the
24 arm runs per-domain budgets its own authors tuned (70/100/300).

### How much evidence there actually is

The undertraining finding rests on **3 rows**, and it is worth being precise
about that before it carries weight in a thesis:

| arm | 5000-epoch rows on disk |
|---|---|
| `ROSAME-I_26` | **3** — blocksworld, depot, hanoi; 1 cell each, all `numtrajs3` |
| `ROSAME-I_24` | none |
| `ROSAME-I_MILP_24` | none |
| `ROSAME-I_MILP_26` | none |

The asymmetry is deliberate: gate 7 is a 26-arm item by design (5000 is the
ICAPS-26 code default; the 24 arm runs its own paper's 70/100/300), and the MILP
arm did not exist when those controls ran.

Gaps, cheapest first:

| to establish | cost |
|---|---|
| it holds on all 5 domains (+gripper, +npuzzle) | ~2 h |
| it holds beyond `numtrajs3` | ~1 h per larger fold |
| the MILP arm behaves the same | ~2.5 h for 5 controls |

gripper's MILP control would be uninformative until §2 is settled, but its
DL-only control is fine.

### Tested: more training does NOT fix the UNSAT (§2)

The obvious hypothesis — that §2's infeasibility is really §3 in disguise, since
UNSAT arises from the *network's* predicted states — is **refuted**. Retraining
one hanoi cell at four budgets and solving five times at each, with a fresh
trace subset per attempt:

| epochs | final loss | solves OPTIMAL |
|---|---|---|
| 131 | 14.028 | **0/5** |
| 500 | 11.021 | **0/5** |
| 1500 | 9.866 | **0/5** |
| 5000 | 9.775 | **0/5** |

38x the training, loss down 30%, 20 attempts across different subsets, not one
satisfiable. **The two failures are independent and need separate fixes.**
Training longer improves the learned model (§3) but will never make the MILP
contribute on gripper, hanoi or most of depot.

This also strengthens §2's diagnosis: it is the *constraint*, not the quality of
the data feeding it. Even a well-trained network cannot predict states in which
those schemas add anything, which is consistent with the observations genuinely
showing nothing become true across those transitions.

---

## 3a. A confound the plan's table did not have: the 26 arm gets more supervision

Not a failure, but it belongs beside the numbers above, because it inflates the
26 arm relative to the 24 arm on every metric.

**ICAPS-24 (`main`) supervises on the final state only.** `train.py:86` is the
sole supervised use of the GT label:

```python
loss += gamma * F.mse_loss(domain_preds[:, -1], label[:, -1], reduction="sum")
```

The only other reference to `label` is `compute_correctness` at line 109 — an
accuracy *metric*, not a loss term. There is no initial-state anchor anywhere.

**ICAPS-26 anchors both endpoints**, each at γ:

| term | anchor | site |
|---|---|---|
| `loss_pred` | **goal** state | `dl/model.py:184` |
| `loss_app` | **init** state | `dl/model.py:195` |

Both are verified faithful to their respective upstreams — our `rosame_i_24`
drops the GT init deliberately (`rosame_i_runner.py`), and `rosame_i_26` passes
both. So this is an **architecture difference between the two papers**, not a
porting defect.

**But it is extra supervision the 26 arm receives for free**, and it cannot be
held equal without deviating from one upstream or the other. It is now a ninth
row in the eight-factor confound table in `rosame_i_26_fixing_PROCESS.md` §4,
and any 24-vs-26 comparison should carry it.

For completeness, since the four arms differ here:

| arm | init anchor | final anchor |
|---|---|---|
| `rosame_i_24` | no | yes (γ) |
| `rosame_i_26` | **yes** (γ) | yes (γ) |
| `rosame_i_milp_24` | yes (hard row in the MILP) | yes (hard row) |
| `rosame_i_milp_26` | yes (hard row) | yes (hard row) |

Note the MILP arms both receive both endpoints as *hard* constraints regardless
of their network, so this confound applies to the DL-only pair.

---

## 4. Options for the UNSAT issue

Not a decision to take on fidelity grounds alone: the rule is not in the paper,
so "upstream fidelity" here means fidelity to their *code*, not their method.

| option | effect | cost |
|---|---|---|
| **Keep `upstream()`** | Report `ROSAME-I_MILP_26` as DL-only on 3 of 5 domains, and say so | none; but the arm is not what its name says on most of the grid |
| **`schema_nonempty=NONE` only** | Minimal change: keeps `forbid_redundant_adds` and `delete_implies_precondition`. Verified to fix both domains | re-run the 26 MILP arm (~11 h) |
| **Full `cdps_dialect()`** | Drops all three rules | re-run, plus a larger deviation to defend |

The middle option is the smallest edit that makes the arm function, and it is
the one the evidence points at.

---

## 5. Open, and worth checking first

**`rosame_i_milp_24` uses `upstream()` too.** Its 150 image rows may carry the
same defect. That check is cheap — read the `exit_status` counts already stored
in `milp_rounds` on those rows — and it decides whether this is a 26-only
problem or grid-wide. Do it before choosing, since it changes the cost of every
option above.
