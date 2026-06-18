# Related Work Intersection Map

Living comparison of this thesis (conflict-search for safe action models from noisy,
partially-observable visual traces) against the four reference papers. Refine as we go.

## The one-line positioning

The open frontier is **safe / conflict-free lifted action models learned from noisy,
partially-observable, action-unlabeled visual traces**. None of the four competitors
occupy that corner:

- SAM / PI-SAM: strong (proven) safety, but clean *symbolic* input.
- ROSAME / ICAPS-26: ingest hard *visual* input, but give *no* safety guarantee.

This thesis = the top-right quadrant (hard input **and** safety).

## Axis-by-axis comparison

| Axis | SAM (KR-21) | PI-SAM (AAAI-24) | ROSAME (ICAPS-24) | ICAPS-26 MILP | This thesis |
|---|---|---|---|---|---|
| Input | symbolic state-action triplets | symbolic, some fluents masked | image seq + actions | image seq, no actions | images -> LLM-extracted noisy PDDL triplets |
| Actions observed | yes | yes | yes | no (predicted) | no (predicted/extracted) |
| State observability | full | partial (masked) | full (image -> state) | full (image -> state) | partial (LLM "unknown" -> masked) |
| Noise in observations | none | none (mask noiseless) | none | perception only | perception + label noise |
| Core mechanism | logical inference rules | inference on observed literals | differentiable PAM + gradient | NN + MILP consistency oracle | conflict detection + best-first patch search over SAM/PI-SAM |
| Safety guarantee | yes, proven | yes, proven | none (soft/empirical) | none (consistency only) | conflict-free (builds on SAM safety) |
| Output | lifted PDDL | lifted PDDL (+ conformant in E-PI-SAM) | lifted human-readable PDDL | lifted human-readable PDDL | lifted PDDL |
| Vocabulary given | yes | yes | yes (preds/actions/signatures) | yes | yes |
| Planning demonstrated | yes (safe model-free planning) | yes | no | no | solving-ratio / false-plan-ratio eval |
| Sample complexity | proven linear in lifted size | proven (bounded concealment) | empirical | empirical | empirical (search cost) |
| Main bottleneck | none (poly, 1-2 traces) | none (poly) | CV training | MILP solve time | conflict-search branching |

## Closest competitor: the ICAPS-26 MILP method

Deepest overlap is **mechanism, not setting**. Both take noisy/inconsistent predicted
triplets and repair them toward logical consistency via *minimal edits*, then learn from
the repaired data.

- Their MILP "fixer" and our `ConflictDrivenPatchSearch` do structurally the same job.
- Our two branch types map onto their two MILP move types:
  - data fix (flip a fluent in a state) <-> their `hol[p,t]` edits
  - model fix (add FORBID/REQUIRE constraint) <-> their `pre/add/del` edits
- Frame axioms are the core consistency constraint in both
  (their Eqs. 36-37; our `_collect_frame_axiom_conflicts`).

So "we add consistency repair" is NOT the novelty. Differentiation is sharper on:

1. **Safety vs. consistency.** They produce a logically consistent explanation that
   maximizes agreement with noisy NN predictions; no soundness guarantee, empirical
   recovery only. We inherit SAM's *proven* safety and target conflict-free models =
   a guarantee.
2. **Noise type.** They handle only perception error on clean, valid, fully-observable
   simulated traces. We handle perception noise *and* LLM label noise/masking under real
   partial observability.
3. **No learner lock-in + cheaper repair.** Their MILP is an NP-hard solve (their stated
   scalability bottleneck). Our conflict search is heuristic best-first/branch-and-bound
   over a SAM/PI-SAM learner used as a black box -- trades optimality for tractability,
   and is built on the *safe* learner family.

## Honest watch-outs (for the writeup)

- SAM/PI-SAM already own "safe + partial observability" -> PO alone is not the novelty.
- ROSAME/ICAPS-26 already own "lifted + interpretable + visual" -> that is not the novelty.
- Defensible core = the *combination*: safe, conflict-free, lifted, from noisy +
  partially-observable + action-unlabeled visual traces.

## Conflict-search internals (for grounding claims)

- Conflict types (`src/pi_sam/noisy_pisam/typings.py`): FORBID_EFFECT_VS_MUST,
  REQUIRE_EFFECT_VS_CANNOT, FORBID_PRECOND_VS_IS, FRAME_AXIOM.
- Detection: `NoisyLearnerMixin.handle_effects()` and `_collect_frame_axiom_conflicts()`
  in `src/pi_sam/noisy_pisam/noisy_learner_mixin.py`.
- Search: `ConflictDrivenPatchSearchBase.run()` in
  `src/pi_sam/plan_denoising/conflict_search.py` (group conflicts -> branch into
  data-fix vs model-fix -> best-first with cost = weighted #fluent_patches +
  #model_constraints; ANYTIME_DFS / UCS; state-encoding for pruning).
- Built on: `NoisyPisamLearner(NoisyLearnerMixin, PISAMLearner)` reusing SAM-learning
  effect extraction; `ConflictDrivenPatchSearchPISAM` / `...SAM` variants.
- Pipeline: images -> LLM object detection + fluent classification
  (`src/trajectory_handlers/llm_blocks_trajectory_handler.py`) -> `.trajectory` +
  `.masking_info` -> ground + mask -> conflict search -> conflict-free lifted model ->
  solve test problems.

## Open questions to resolve next

- Does conflict-free imply safe in the SAM sense here, or only consistent? (state the
  exact guarantee the search provides).
- Cost/optimality of the search vs. the MILP optimum -- any approximation bound?
- Demonstrate planning with the learned model (both ROSAME and ICAPS-26 skip this -- an
  opportunity).
- Sample-complexity / robustness story under increasing noise and masking rates.
