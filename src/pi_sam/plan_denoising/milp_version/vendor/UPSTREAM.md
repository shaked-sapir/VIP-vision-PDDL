# Vendored ROSAME+MILP solver code

Source repository: https://github.com/xikaioliver/ROSAME
Branch: `ROSAME+MILP`
Commit: `95c733f1fecd9ddbe9634c8c54ebc85b27ebc076` (2026-06-20)

Vendored packages (top-level names preserved; loaded via a sys.path insertion
in `src/pi_sam/plan_denoising/milp_version/__init__.py` so upstream's
absolute imports keep working):

- `planning_structs/` — `domain.py`, `instance.py`, `traces.py` (verbatim)
- `constraint_opt/` — `factory.py`, `util.py`, `cp_sat.py`, `mip_gurobi.py`
  (verbatim), `__init__.py` (MODIFIED: gurobipy import guarded so the CP-SAT
  path needs no Gurobi license)

Our observed-actions encoder variant lives OUTSIDE the vendor tree
(`src/pi_sam/plan_denoising/milp_version/encoder.py`) and is registered in
the same factory under `"cp-sat-observed"`. It is shared by the
`rosame_milp*` baselines and by our own `cdps_milp_*` learners.

## Reference hyperparameters (paper Sec. 7 "Training", confirmed in code)

From `train_common.py` on the vendored commit:

| Parameter | Code value | Paper value | Note |
|---|---|---|---|
| epochs | 5000 | 5000 | our runners default to the VIP regime (100) |
| batch_size | 128 | 128 | |
| optimizer / lr | Adam / 1e-4 | Adam β=(0.9,0.999) / 1e-4 | |
| `lambda` (prior bias) | **0.2** | **0.4** | paper/code discrepancy; AMLGym's vendored ROSAME hard-codes 0.2 in the train step |
| `gamma` (final-step weight) | 10 | 10 | |
| `pre_mip_epoch` (warmup) | 50 | 50 | |
| `mip_interval` | 1 (every epoch) | every epoch | |
| `mip_traces` (subset size) | 3 | 3–4 random | our default: whole fold (3–8 traces ≈ their subset anyway); paper behavior available via `mip_traces` |
| `mip_time_limit` | 60 s | "a time limit" | |
| `pseudo_weight_decay` (ψ) | 0.99 | 0.99 | **applies only to per-trace (state/action) labels** — see below |
| `cp_type` | `mip-gurobi` | Gurobi 12.0.1 | we default to `cp-sat-observed` (no license); Gurobi pluggable |

## Documentation notes: paper/code discrepancies

1. **Constraints in code but absent from the paper.** The paper specifies the
   MILP entirely via equations 11–37 (Sec. 6). The released code adds two
   constraint families mentioned nowhere in the paper (verified by full-text
   search):
   - *Non-empty schemas* (`PreIsNotEmpty`/`AddIsNotEmpty`;
     `mip_gurobi.py:102–103`, mirrored in `cp_sat.py`): every action schema
     must select ≥1 precondition and ≥1 add effect.
   - *No redundant adds* (`StepAddPre`; `mip_gurobi.py:177`, mirrored in
     `cp_sat.py`): `stepadd[p,t] + hol[p,t] <= 1`. With observed actions this
     can make the ground-truth model infeasible in domains with legal
     redundant adds (e.g. depot's `drop` adding `(at ?p ?d)` after `lift`,
     which does not delete it).
   Both are kept in our encoder for fidelity, each behind a flag
   (`enforce_nonempty_schemas`, `forbid_redundant_adds`, default True =
   upstream behavior).
2. **ψ-decay does not touch the model channel.** The paper describes
   pseudo-label influence "annealed ... using ψ = 0.99". In the code
   (`dl/model.py:274–306`), ψ multiplies only the per-trace state/action label
   weights; the *action-model* pseudo-label CE (`loss_pseudo_m`) is unweighted
   and its labels are simply overwritten at each solve. Since the model
   channel is the only one that survives in our simulation setting (states and
   actions are data, not network outputs), our loop faithfully applies
   fresh, undecayed model-CE per round.
3. **λ mismatch** — paper Sec. 7 states λ=0.4; the code default is 0.2 (their
   ICAPS-24 README uses 0.2 for most domains, 0.4 for 8-puzzle).
4. **Grounding excludes repeated-object tuples.** `Instance._build_actions` and
   `Instance._build_propositions` (`instance.py:96,104`) ground with
   `itertools.permutations`, so there is no `(on a a)` proposition and no
   `stack(a,a)` action. Consistent for ROSAME — its network shares that
   vocabulary — but wrong for `cdps_milp_*`, where (a) our observations are
   CWA-completed with `itertools.product` and therefore *do* contain `(on a a)`,
   which would then reach PI-SAM with no MILP variable behind it (a hole in
   design §4.1), and (b) a step whose action repeats an object has no grounded
   Action, so `observation_to_trace` drops the entire trace.
   **The vendor file is left verbatim.** `converter.RepeatedArgsInstance`
   subclasses it with `product`-based grounding and is selected by
   `build_ps_instance*(..., include_repeated_args=True)` — on for `cdps_milp_*`,
   off (upstream) for `rosame_milp*`. Cost: n!/(n−k)! → n^k tuples (~8% more on
   a 10-block instance).

   The *lifted* binding enumeration (`domain.py:164`, also `permutations`) needs
   no change: PI-SAM's `PredicatesMatcher` likewise enumerates distinct action
   *parameter slots*, so both vocabularies contain `(on ?x ?y)`/`(on ?y ?x)` but
   never `(on ?x ?x)`. Verified equivalence — for `stack(a,a)`, PI-SAM matches
   `(on a a)` to both `(on ?x ?y)` and `(on ?y ?x)`, and our encoder returns
   bindings `[(1,2), (2,1)]`, which it ORs together (`encoder._bindings` +
   `cp.any`).
