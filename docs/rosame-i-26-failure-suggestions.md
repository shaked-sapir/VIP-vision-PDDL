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

## 3. Options

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

## 4. Open, and worth checking first

**`rosame_i_milp_24` uses `upstream()` too.** Its 150 image rows may carry the
same defect. That check is cheap — read the `exit_status` counts already stored
in `milp_rounds` on those rows — and it decides whether this is a 26-only
problem or grid-wide. Do it before choosing, since it changes the cost of every
option above.
