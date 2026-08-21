# Vendored ROSAME+MILP solver code

Source repository: https://github.com/xikaioliver/ROSAME
Branch: `ROSAME+MILP`
Commit: `95c733f1fecd9ddbe9634c8c54ebc85b27ebc076` (2026-06-20)

Vendored packages (top-level names preserved; loaded via a sys.path insertion
in `src/milp/__init__.py` so upstream's absolute imports keep working):

- `planning_structs/` — `domain.py`, `instance.py`, `traces.py` (verbatim),
  `util.py` (verbatim)
- `constraint_opt/` — `factory.py`, `util.py`, `cp_sat.py`, `mip_gurobi.py`
  (verbatim), `__init__.py` (MODIFIED: gurobipy import guarded so the CP-SAT
  path needs no Gurobi license)
- `dl/` — the ICAPS-26 network: `model.py`, `mixins/`, `util/dataset.py`,
  `util/layers.py`, `util/plot.py`, `util/util.py`, `util/ROSAME/rosame.py`,
  `main/normalization.py` (verbatim); `network.py`, `util/tuning.py`,
  `__init__.py`, `main/__init__.py` (MODIFIED, see below)
- `convertor/` — `convertor.py`, `translator.py`, `selector.py`,
  `pseudo_label.py` (all verbatim)
- `util/` — `model_perm.py`, `pddl_parsing.py` (MODIFIED, see below)

Not vendored, deliberately:

- `dl/main/common.py`, `dl/main/rosame_full.py` — upstream's argparse CLI layer.
  Our adapter replaces it.
- `dl/util/ROSAME/models/domains/` — upstream's checked-in grounding assets.
  Ours are generated per run; see "Domain assets" below.

Our observed-actions encoder variant lives OUTSIDE the vendor tree
(`src/milp/encoder.py`) and is registered in
the same factory under `"cp-sat-observed"`. It is shared by the
`rosame_milp*` baselines and by our own `pisam_milp_*` learners.

## Vendor modifications

Every one is marked in-file with a `VENDOR MODIFICATION` comment naming this
document. Nine in total.

| file | what | why |
|---|---|---|
| `constraint_opt/__init__.py:7` | gurobipy import guarded | the CP-SAT path needs no Gurobi license |
| `dl/network.py:8` | tensorboard import guarded | not a project dependency; upstream's `SummaryWriter` logging is inert for us |
| `dl/util/tuning.py` | replaced by a stub exporting `parameters: Dict` | upstream is 655 lines of grid/genetic search that spawns subprocesses and re-runs whole experiments — none of which may happen inside a fold. `dl/main/normalization` imports the dict as an image mean/std cache and is the only reachable use. **Upstream's cache is unkeyed and carries the image shape**, so one process touching two domains reuses a wrong-shaped array; our adapter owns a keyed on-disk cache and writes through this dict. |
| `dl/__init__.py:1` | drops `from . import main` | pulls in the un-vendored argparse CLI layer |
| `dl/main/__init__.py:1` | drops the CLI re-exports | same; the package survives only to host `normalization.py` |
| `util/pddl_parsing.py:6` | `lifted_pddl` import guarded | not a project dependency, and this module is on the import path of *every* consumer of `dl/network.py` (via `convertor.convertor`), so an unguarded import makes the whole vendored DL tree unimportable. `Parser` is reachable only from `parse_pddl_domain`, which raises. |
| `util/pddl_parsing.py:76` | f-string quote rewrite | upstream reuses the outer double quotes inside the f-string, which only parses under PEP 701 (Python ≥ 3.12); we are on 3.11. Produced string is identical. |
| `dl/util/ROSAME/rosame.py:375` | `get_domain_model(domain, root=None)` | see "Domain assets" |
| `dl/mixins/action_model.py:10` | `domain_assets_root` threaded through `_build_around` | see "Domain assets" |

## Domain assets

Three files are resolved by domain name, in two families with different
lifetimes. `src/milp/domain_assets.py` generates all of them from **our**
`src/domains/*.pddl`, so every vocabulary derives from the file the rest of the
pipeline parses rather than being maintained in parallel.

**Static, checked in** — read by `Convertor.__init__` relative to the vendor root.
Regenerate after editing any domain PDDL with `python -m src.milp.domain_assets`:

    planning_structs/specs/<domain>/domain.json   the CP domain spec
    pddl/<domain>/domain.pddl                     the `gt_am` reference

**Run-scoped, generated** — read by `ROSAMEMixin._build_around` via
`get_domain_model`, from a root the caller passes:

    <domain_assets_root>/<domain>/domain_model.json   the DL head's vocabulary
    <domain_assets_root>/<domain>/objects.json        its grounding universe

The second family cannot be checked in. Upstream's `objects.json` is a constant
because its corpora are object-homogeneous; ours is the per-run object union
over the whole `data_dir`. Hence the two modifications above.
`get_domain_model(root=None)` preserves upstream's path exactly, but no code of
ours takes that branch.

Directories are keyed by **benchmark domain key** (`blocksworld`, `npuzzle`, ...),
which is what the runners already pass around, so `Convertor`'s hardcoded
`"blocks" -> "blocksworld"` alias (`convertor.py:41`) stays a no-op and that file
needs no edit.

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
   vocabulary — but wrong for `pisam_milp_*`, where (a) our observations are
   CWA-completed with `itertools.product` and therefore *do* contain `(on a a)`,
   which would then reach PI-SAM with no MILP variable behind it (a hole in
   design §4.1), and (b) a step whose action repeats an object has no grounded
   Action, so `observation_to_trace` drops the entire trace.
   **The vendor file is left verbatim.** `converter.RepeatedArgsInstance`
   subclasses it with `product`-based grounding and is selected by
   `build_ps_instance*(..., include_repeated_args=True)` — on for `pisam_milp_*`,
   off (upstream) for `rosame_milp*`. Cost: n!/(n−k)! → n^k tuples (~8% more on
   a 10-block instance).

   The *lifted* binding enumeration (`domain.py:164`, also `permutations`) needs
   no change: PI-SAM's `PredicatesMatcher` likewise enumerates distinct action
   *parameter slots*, so both vocabularies contain `(on ?x ?y)`/`(on ?y ?x)` but
   never `(on ?x ?x)`. Verified equivalence — for `stack(a,a)`, PI-SAM matches
   `(on a a)` to both `(on ?x ?y)` and `(on ?y ?x)`, and our encoder returns
   bindings `[(1,2), (2,1)]`, which it ORs together (`encoder._bindings` +
   `cp.any`).
5. **The DL head grounds arguments in sorted-type order, the PDDL does not.**
   `dl/util/ROSAME/rosame.py` stores a signature as a `{Type: count}` map, and
   `Predicate.__init__` does `sorted(params.keys(), key=lambda x: x.name)`.
   `Predicate.ground` then emits one `itertools.permutations` group per key in
   that order, so **a PDDL signature that is not already grouped and name-sorted
   is grounded as a permutation of itself** and the head is not positionally
   comparable to the CP proposition list.

   Measured on our five domains — predicates plus schemas whose order differs:
   blocksworld **0**, depot 10, gripper 2, hanoi 2, npuzzle 2. blocksworld being
   zero is a trap, since it is the domain most gates here use.

   This is a divergence between the two upstreams. AMLGym's ICAPS-24 fork
   carries the same line commented out (`self.params_types = list(params.keys())`)
   and so recovers PDDL order by accident of dict insertion order. The 24 arms
   therefore never see the reordering and the 26 arms always do;
   `benchmark/algorithm_adapters/test_check_predicate.py` is green for the 24
   fork only and must not be read as evidence otherwise.

   **The vendor file is left verbatim.** `domain_assets.rosame_argument_permutation`
   reproduces the mapping — a *stable* argsort by type name, so same-typed
   arguments keep their PDDL order. It is a bijection and lossless even for a
   signature repeating a type at non-adjacent positions (hanoi's
   `move_peg_disc(disc, peg, disc)` → `[0, 2, 1]`), because `ground` permutes
   rather than combines. Any adapter must map CP index → head index through it;
   aligning by predicate name alone is silently wrong on four of five domains,
   and `translator.trans_full_state` zips **positionally**, so a permuted vector
   of the right width raises nothing and simply means different propositions.
   Pinned by `src/milp/test_rosame_argument_order.py`.
