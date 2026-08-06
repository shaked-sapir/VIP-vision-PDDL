# Depot both-polarity corruption — reproduction & explanation

## What this is

A minimal, inspectable reproduction of the bug that made **all 270 depot
`ROSAME_MILP` cells fail** with OR-Tools' *"solution hint contains duplicate
variables (index #242)"*, and that silently feeds contradictory states to
**every** depot consumer of `ground_observation_completely` — the ROSAME
baselines, the MILP adapter, **and the CDPS/PI-SAM pipeline**
(`run_fold.py:58`, `masking.py:221`).

## Exact provenance (what to open to verify)

| What | Where |
|---|---|
| Experiment cell | `benchmark/running_results/depot/sim_run__mask=0.0__noise=0.0/testing/fold0_numtrajs3_gtrate0/` |
| Trajectory | `original_observations/original_observation_problem3.trajectory` (first of the cell, sorted) |
| Problem PDDL | `benchmark/data/depot/multi_problem_04-07-2026T01:32:08__model=gpt-5.2__steps=100/training/trajectories/problem3/problem3.pddl` |
| Domain PDDL | `src/domains/depot/depot.pddl` — note `(clear ?x - object)` |

The `mask=0.0 / noise=0.0` cell is chosen deliberately: **no masking, no
noising** — the corruption appears with completely clean inputs, proving it is
introduced by `ground_observation_completely` itself, not by any noise
machinery.

## What to look at

`state_dump.txt` in this directory contains:

1. **State-by-state summary** — every one of the 10 states of the trajectory
   gains 2–3 both-polarity fluents. Always `clear(...)`, never any other
   predicate, and the doubled fluents are exactly the ones that are *true* in
   that state (packages **and** piles — anything `clear` ranges over).
2. **Full completed state 0** — all 54 fluents after completion, with `>>>`
   marking the contradictory pairs, e.g. both `(clear p1 - package)` **and**
   `(not (clear p1 - object))` present simultaneously. Note the type
   annotations of the two entries — that *is* the bug (see below).
3. **Raw parsed state 0** — the 15 fluents actually in the trajectory file:
   clean, one polarity each. The file is innocent.

Regenerate everything with:

```bash
python src/depot-polarity-test/repro.py
```

The script also prints the smoking-gun triple for `clear(p1)`:
`eq: True | same str: False | same hash: False | set membership: False`.

## The mechanism, in four steps

1. **Two derivations of the "same" fluent.** The `TrajectoryParser` stores
   `(clear p1)` with the parameter type *refined to the concrete object's
   type*: `(clear p1 - package)`. The CWA-completion enumerator
   (`get_all_possible_groundings`, `src/utils/pddl_state.py:248`) reuses the
   *lifted declared* signature: `(clear p1 - object)`.
2. **Broken eq/hash contract in pddl_plus_parser.** `GroundedPredicate.__eq__`
   is type-hierarchy-aware → the two objects compare **equal**. But
   `__hash__ = hash(str(self))`, and the strings differ (`- package` vs
   `- object`) → **different hashes**. Equal objects with unequal hashes
   violate Python's contract.
3. **Set membership silently fails.** `in` jumps to the hash bucket first and
   only calls `__eq__` there. The generated `(clear p1 - object)` lands in a
   different bucket than the stored `(clear p1 - package)`, so the
   hierarchy-aware `__eq__` is never consulted →
   `ground_all_predicates_in_state` (`pddl_state.py:279`) concludes the fluent
   is absent → adds `(not (clear p1))`.
4. **Contradiction → crash or silent bit-flips.** The MILP converter maps both
   polarities to the same proposition → duplicate solution-hint variables →
   OR-Tools rejects the model (loud failure). ROSAME's encoder writes a truth
   bit per proposition while iterating the state → whichever polarity comes
   last in set-iteration order wins (silent corruption). PI-SAM/CDPS receive
   the same contradictory states.

## Why only depot

The mismatch requires *declared parameter type ≠ concrete object type*, i.e. a
predicate parameter typed with a **non-leaf ancestor type**. Depot's
`clear(?x - object)` is the only such predicate across all benchmark domains —
blocksworld's `clear` is over `block`, gripper/hanoi/npuzzle use concrete
types everywhere. That is why 1,080 non-depot cells ran clean while depot
failed 270/270. Any future domain with type hierarchies (IPC depots,
transport, ...) would hit the same mine.

## The intended fix (not yet applied)

In `ground_all_predicates_in_state`: test presence with an explicit
**type-free, polarity-free key** (predicate name + ordered object names)
instead of set membership over objects with a broken hash. For leaf-typed
domains the function is extensionally identical (all three membership cases
produce byte-identical output), so existing results cannot change; only
ancestor-typed predicates — the pathological case — behave differently, which
is the repair. A defensive duplicate-proposition assert in the MILP converter
(`observation_to_trace`) makes any future variant of this bug fail loudly
with a named fluent.
