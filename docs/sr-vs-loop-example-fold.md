# SR vs LOOP: the two learned action models of one fold

Fold: `benchmark/running_results/blocksworld/simulation-final-run__mask=0.01__noise=0.2/testing/fold0_numtrajs3_gtrate0`

| | |
|---|---|
| Domain | blocksworld |
| Degradation | p_mask = 0.01, p_noise = 0.2 |
| Training traces | problem3 (5 steps), problem7 (8 steps), problem6 (2 steps) |
| Test problems | problem1, problem10 |
| SR learner input | all 3 traces, repaired jointly (91 flips) |
| LOOP learner input | round 2's subset {problem3, problem7} only, repaired on their own (83 flips) |
| GT-free V on the 3 originals | SR 91, LOOP 91, reference domain 91 |

Files: `pisam_milp_single_round/learned_domain_PISAM_MILP_SR.pddl` and `pisam_milp_loop/learned_domain_PISAM_MILP_LOOP.pddl` inside the fold.

## Legend

<span style="background:#ffd8a8;padding:0 4px">orange</span> = literal present only in the SR model &nbsp;&nbsp;
<span style="background:#d0ebff;padding:0 4px">blue</span> = literal present only in the LOOP model &nbsp;&nbsp;
plain = identical in both.
A `✗` after a highlighted literal marks the arm that is wrong against the reference domain; `✓` marks the arm that is right.

Preconditions and effects are listed sorted, so the two columns line up. `(not (= ?x ?y))` is emitted by PI-SAM in both arms and is omitted here.

## Side by side

<table>
<tr><th width="50%">SR (pisam_milp_single_round)</th><th width="50%">LOOP (pisam_milp_loop)</th></tr>

<tr><td colspan="2"><b>pick_up ?x</b></td></tr>
<tr>
<td><pre>:precondition (and
  (clear ?x)
  (handempty)
  (ontable ?x))
:effect (and
  (holding ?x)
  (not (clear ?x))
  (not (handempty))
  <span style="background:#ffd8a8">(not (ontable ?x))</span>  ✓
)</pre></td>
<td><pre>:precondition (and
  (clear ?x)
  (handempty)
  (ontable ?x))
:effect (and
  (holding ?x)
  (not (clear ?x))
  (not (handempty))
  <span style="background:#d0ebff">— missing (not (ontable ?x))</span>  ✗
)</pre></td>
</tr>

<tr><td colspan="2"><b>put_down ?x</b></td></tr>
<tr>
<td><pre>:precondition (and
  <span style="background:#ffd8a8">— no precondition at all</span>  ✗
)
:effect (and
  (clear ?x)
  (handempty)
  <span style="background:#ffd8a8">— missing (not (holding ?x))</span>  ✗
  (ontable ?x)
)</pre></td>
<td><pre>:precondition (and
  <span style="background:#d0ebff">(holding ?x)</span>  ✓
)
:effect (and
  (clear ?x)
  (handempty)
  <span style="background:#d0ebff">(not (holding ?x))</span>  ✓
  (ontable ?x)
)</pre></td>
</tr>

<tr><td colspan="2"><b>stack ?x ?y</b></td></tr>
<tr>
<td><pre>:precondition (and
  (clear ?y)
  <span style="background:#ffd8a8">— missing (holding ?x)</span>  ✗
)
:effect (and
  (clear ?x)
  (handempty)
  (not (clear ?y))
  (not (holding ?x))
  (on ?x ?y)
)</pre></td>
<td><pre>:precondition (and
  (clear ?y)
  <span style="background:#d0ebff">(holding ?x)</span>  ✓
)
:effect (and
  (clear ?x)
  (handempty)
  (not (clear ?y))
  (not (holding ?x))
  <span style="background:#d0ebff">(not (ontable ?x))</span>  ✗ (extra; a no-op in reachable states)
  (on ?x ?y)
)</pre></td>
</tr>

<tr><td colspan="2"><b>unstack ?x ?y</b></td></tr>
<tr>
<td><pre>:precondition (and
  (clear ?x)
  (handempty)
  (on ?x ?y))
:effect (and
  (clear ?y)
  <span style="background:#ffd8a8">— missing (holding ?x)</span>  ✗
  (not (clear ?x))
  (not (handempty))
  (not (on ?x ?y))
)</pre></td>
<td><pre>:precondition (and
  (clear ?x)
  (handempty)
  (on ?x ?y))
:effect (and
  (clear ?y)
  <span style="background:#d0ebff">(holding ?x)</span>  ✓
  (not (clear ?x))
  (not (handempty))
  (not (on ?x ?y))
)</pre></td>
</tr>
</table>

## Same content as a diff (SR → LOOP)

For renderers that strip inline colours. `-` lines exist only in SR, `+` lines only in LOOP.

```diff
 (:action pick_up
   :parameters (?x - block)
   :precondition (and (clear ?x) (handempty) (ontable ?x))
   :effect (and (holding ?x) (not (clear ?x)) (not (handempty))
-               (not (ontable ?x))
   ))

 (:action put_down
   :parameters (?x - block)
-  :precondition (and )
+  :precondition (and (holding ?x))
   :effect (and (clear ?x) (handempty) (ontable ?x)
+               (not (holding ?x))
   ))

 (:action stack
   :parameters (?x - block ?y - block)
-  :precondition (and (clear ?y) (not (= ?x ?y)))
+  :precondition (and (clear ?y) (holding ?x) (not (= ?x ?y)))
   :effect (and (clear ?x) (handempty) (not (clear ?y)) (not (holding ?x)) (on ?x ?y)
+               (not (ontable ?x))
   ))

 (:action unstack
   :parameters (?x - block ?y - block)
   :precondition (and (clear ?x) (handempty) (on ?x ?y) (not (= ?x ?y)))
   :effect (and (clear ?y) (not (clear ?x)) (not (handempty)) (not (on ?x ?y))
+               (holding ?x)
   ))
```

## Scorecard against the reference domain

| Action | SR errors | LOOP errors |
|---|---|---|
| pick_up | none | missing effect `(not (ontable ?x))` |
| put_down | missing precondition `(holding ?x)`; missing effect `(not (holding ?x))` | none |
| stack | missing precondition `(holding ?x)` | extra effect `(not (ontable ?x))` |
| unstack | missing effect `(holding ?x)` | none |

| Metric on the test problems | SR | LOOP |
|---|---|---|
| solving ratio | 0.5 | 1.0 |
| predictive applicability precision | 0.57 | 1.0 |
| predictive effects recall | 0.94 | 0.88 |
| precision / recall (overall) | 0.93 / 0.83 | 0.92 / 0.96 |

SR's errors are all *missing* preconditions and effects, which makes its model too permissive: it plans with `stack` and `put_down` while holding nothing, and those plans are invalid under the reference domain (false plans). LOOP's errors are one missing delete effect and one redundant one, neither of which the test problems can expose.

## Why the two arms diverge here

- The fold contains 91 injected noise flips (27 in problem3, 57 in problem7, 7 in problem6). SR's optimal joint repair also costs 91, and the reference model scores V = 91 on the originals. The truth-consistent repair therefore costs the same as the one CP-SAT returned: the min-flip optimum is a tie, and the solver picked a member whose T′ still carries 4 wrong fluents, placed where PI-SAM drops the `holding` literals.
- LOOP's winning round solved only {problem3, problem7}. On their own those two traces have a cheaper optimum (83 versus the 84 the joint solve spent on them), so a different tied T′ came out, with 3 residual wrong fluents in different places. PI-SAM on that input keeps `holding` and loses `(not (ontable ?x))` instead.
- V could not have chosen between the two models: both score 91, the same as the reference domain. LOOP's better test result in this fold is where the residual errors happened to land, not a selection effect.
