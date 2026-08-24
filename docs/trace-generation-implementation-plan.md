# Implementation Plan: Generic Trace Generation

> **Audience**: an implementing agent with access to this repo
> (`VIP-vision-PDDL`).
>
> **Goal**: make trajectory generation source-agnostic and configurable, so a
> corpus of arbitrary size and length distribution can be produced from **either**
> a problem file (walked with a planner, a random walk, or a mix) **or** an
> existing trajectory file — for **any** PDDL domain, not only the ones PDDLGym
> ships. Symbolic output only; images remain the existing pipeline's job.

> **Status (2026-08-15, branch `trace-generation-implementation`)**: §8 steps 1–4
> are built and the step-4 acceptance gate passed (§8.1). Steps 5–6 and
> multi-input corpora are not built. This document has been edited in place to
> match what exists; where a statement was superseded it is struck through and
> the replacement given, rather than deleted, so the reasoning stays auditable.
>
> **Revision (2026-08-20, branch `revising-trace-generation`)**: a review of the
> built code against this document produced a list of corrections, all applied.
> The three that change the *contract*, rather than only the code, are §0.2
> items 20–22 below and are the reason §4, §5 and §7 have been rewritten:
> `buckets` is **deleted**, trace mode reads **nothing** from `config.yaml`, and
> a domain no longer has to be registered at all. The rest — a closed-loop
> window rejection, an fd leak in `_plan_from`, an untested
> `preserve_solvability`, silent truncation, an uncached `domain_name` — were
> code-only and left this document's claims standing.
>
> ```
> python benchmark/data_generator.py --gen-mode trace \
>     --domain blocksworld --source-kind problem \
>     --problem-file src/domains/blocks/problems/problem1/problem1.pddl \
>     --num-problems 10 --seed 17 --cut-seed 5
> ```
>
> and, for a domain this repo has never heard of:
>
> ```
> python benchmark/data_generator.py --gen-mode trace \
>     --domain-file /elsewhere/elevator.pddl --source-kind problem \
>     --problem-file /elsewhere/elevator-p1.pddl \
>     --num-problems 10 --p-rnd 0.4 --seed 17
> ```

---

## 0. Settled design decisions (do not revisit)

1. **Two input paths, one downstream pipeline.**
   - *From problem*: walk it (planner / random / mixed), then optionally cut.
   - *From trajectory*: take it whole, or cut it.
   Cutting is identical in both cases and must live in one place.
2. **The default walk backend is `unified_planning`, not PDDLGym.** A problem may
   belong to a domain PDDLGym has never heard of. PDDLGym becomes the
   *renderer-only* backend, used solely when images are wanted.
3. **Symbolic output only for the new paths.** No rendering, no LLM inference.
   Imaged generation stays exactly as it is today.
4. ~~**Length modes**: `none` | `uniform(min, max)` | `buckets([...])`.
   Buckets emit into a **single mixed pool**, with each window's length recorded
   in metadata — *not* one directory per length.~~
   **Superseded by item 20 — the modes are `none` | `uniform(min, max)`.**
5. **Input trajectories are ground truth.** Windows land in `gt_trajectories/`;
   degradation stays the experiment's job (`SimulatedDataSource` injects masking
   and noise per fold). `TrajectorySource` **trusts its caller** on this: it does
   not attempt to detect or reject a noisy input. A `.trajectory` carries no
   marker distinguishing GT from degraded, so any such check would be a guess.
6. **A trajectory input requires its problem `.pddl`.** Two independent reasons:
   a `.trajectory` carries object *names* but not their *types*, so a valid
   `:objects` block cannot be written from it alone; and
   `TrajectoryParser(domain, problem).parse_trajectory(...)` takes a `problem`
   argument, so the file cannot even be read without one. Fail loudly.
7. **Option B for the existing code**: share the cutter and emitter with the
   PDDLGym path; keep the sources separate. Nothing is deleted.

### 0.1 Settled at implementation time (2026-08-15)

Items 8–13 correct §0–§10 where the code disagreed with them; 14–19 fill gaps the
plan left open. Where these conflict with the sections below, these win.

8. **A corpus is two trees, not one.** §5 of this plan implies windows land in
   `gt_trajectories/` alone. They do not — a runnable corpus needs both:

   ```
   <corpus>/training/trajectories/problemN/problemN.pddl
   <corpus>/training/trajectories/problemN/plan.txt
   <corpus>/training/trajectories/problemN/problemN_trajectory.json
   <corpus>/gt_trajectories/problemN/problemN.trajectory
   <corpus>/gt_trajectories/problemN/problemN_trajectory.json
   ```

   `SimulatedDataSource.prepare` reads the problem `.pddl` from the *training-pool*
   dir (`data_source.py:224`), the fold copies the pool `_trajectory.json`
   (`run_fold.py:482-522`), and `benchmark_runner._discover_gt_trajectories` globs
   `gt_trajectories/*/*.trajectory` — the `.trajectory`, not the JSON.
9. **The emitter calls `gt_builder.export_gt_trajectory`; it does not port
   `_write_gt_trajectory`.** §5 says port it. That method writes only the JSON,
   so the `.trajectory` half of item 8 would be missing. `export_gt_trajectory`
   already writes both.
10. **"Any PDDL domain" holds for *generation*, not for *running*.** A corpus can
    be generated for a domain absent from `config.yaml`; running an experiment on
    it still needs a `domains.<key>` entry, because `_resolve_domain_config`
    raises without one (`experiment_runner.py:92`).
11. ~~**A planner walk stops at the goal.** The plan never says what happens when
    the walk terminates before the step budget is spent. It stops — no forced
    continuation, no re-reset. The consequence is explicit: at `p_rnd=0` corpus
    size is bounded by plan length, and `p_rnd` is therefore both a
    randomness dial *and* a corpus-size dial.~~
    **Superseded by the code, which is right and was left alone.** Reaching the
    goal is a *choice*, `stop_at_goal`, and it defaults **off**
    (`sources.py:_guided_walk`). Off, the walk continues from the goal state as a
    random walk for the rest of its budget, because replanning from a goal state
    returns the empty plan forever and the walk would otherwise spin. So
    `p_rnd=0` does **not** bound corpus size at plan length by default; it means
    "planner until the goal, then random". `stop_at_goal=True` restores the
    behaviour this item described, and only then is `p_rnd` also a corpus-size
    dial. Producing fewer windows than requested **warns and continues**; it is
    not an error — see item 24 for where the warning surfaces.
12. **Planner use follows `p_rnd`.** `p_rnd=1` never invokes the planner, so a
    pure random walk needs no planner engine installed. `preserve_solvability`
    (the AMLGym replan-after-random-action step, §10.2 — named
    `solvability_preserving` in this document's first draft) defaults **off**.
13. ~~**`benchmark/evaluation/test_states_generator.py` is not touched.**~~
    **Superseded — it was rewired onto the shared walk** (§3.1 and §9 win over
    §11). Leaving it alone would have meant two copies of the AMLGym
    plan/substitute/replan loop drifting apart, and the S_test copy is the one
    with the private-API coupling worth having in a single place. It keeps its
    own S_test post-processing; only the walk moved. Its module-global `random`
    seeding becomes a per-instance `random.Random`, which changes which states
    a given seed produces — accepted, since S_test is regenerated per run and
    no stored S_test is reproduced from a recorded seed.
14. ~~**Many inputs, one corpus.**~~ **Not built.** `build_source` takes one
    `problem_file` *or* one `trajectory_file`, and `generate_corpus` takes one
    source. Multi-input would also void gotcha §10.5's uniform object universe,
    which ROSAME-I requires, so it is not a free extension. Adding it later
    means a loop over sources in `generate_corpus`, a global window counter,
    and threading item 15's signatures — no change to the cutter or emitter.
15. **Deduplication is global across inputs.** `cut()` takes
    `exclude_signatures`; the caller threads the accumulated signatures through
    successive calls. The set is copied, never mutated, so `cut()` stays pure
    (§4). Built and tested, but with item 14 unbuilt it has no caller yet.
16. **`cut_mode=none` means one window per input**, i.e. the trace as-is,
    uncut. With a single input that yields a one-problem corpus, which
    `experiment_runner.py:301` rejects (it needs ≥ 2 problem dirs) — so `none` is
    for multi-input corpora, or for corpora consumed by something other than a
    fold.
17. ~~**The entry point is keyed by the `config.yaml` domain key**, not by raw
    domain/problem paths — consistent with every other `data_generator` mode.
    Generation settings live under `domains.<key>.generation`, CLI-overridable.~~
    **Superseded by items 21 and 22.** Consistency with the other modes was the
    only argument for it, and it cost the headline goal: keying on a registry
    entry means a domain must be in the registry, which contradicts "for **any**
    PDDL domain" at the top of this document.
18. **Semantics kept verbatim from the gym generator**: goal = the window's full
    final state; a source problem's own goal is discarded; closed-world positive
    fluents only; window lengths count *actions*; and two independent
    `random.Random` instances (walk RNG, cutter RNG), per §10.4.
19. **Scope of this implementation: §8 steps 1–4.** Steps 5 (retrofitting
    `PDDLGymProblemGenerator` onto the shared cutter and emitter) and 6
    (arbitrary-problem injection into the pddlgym backend) are both excluded, so
    `src/trajectory_handlers/pddlgym_problem_generator.py` is untouched and the
    imaged path still runs its own private walk/cut/emit. Step 5's stated
    verification — regenerate an existing npuzzle corpus at the same seed and
    diff — is **unmeetable**: the corpus on disk records no seed, and its
    `_trajectory.json` files were overwritten in place by the inference pass.
    Substituted check, to be run when step 5 is: two fresh `--no-inference` runs
    at one fixed seed, one before and one after the retrofit, must produce
    byte-identical trees.

### 0.2 Settled at revision time (2026-08-20)

Items 20–22 change the contract and supersede §0 item 4, §0.1 item 17, §4, §5
and §7. Items 23–25 record behaviour the plan never specified. Where these
conflict with anything above, these win.

20. **`buckets` is deleted, not deferred.** It was built, tested and never used:
    no caller ever passed it, and the reporting layer it was for (§5, "grouped
    by length at report time") does not exist and is not planned. The intended
    workflow is one corpus at a time at one length range, split by hand
    afterwards if a length comparison is ever wanted, which `uniform` already
    serves. `CutMode` is now `NONE | UNIFORM`, `cut()` has no `buckets`
    parameter, and `generation_info.json` has no `buckets` key. Reinstating it
    is a `CutMode` member and a length stream, not a redesign.
21. **Trace mode reads nothing from `config.yaml` except a domain file path.**
    Every setting is a `generate_trajectories_via_trace` argument with a real
    default; the CLI flag is the only way to change it. `config.yaml` is
    consulted for exactly one thing — looking up a registered `--domain`'s
    `domain_file` — and not at all when `--domain-file` is passed. This replaces
    §7.1's table of config keys with a table of flags, and removes the
    override-precedence rules that table needed. The reason is that a corpus is
    a *run*, not a property of a domain: two corpora from the same domain
    normally differ in every one of these settings, so a per-domain config block
    is the wrong shape, and a tri-state `--render` exists only to override a
    config value that no longer exists.
22. **A domain does not have to be registered.** `--domain-file <path>` takes any
    PDDL domain, and `--domain` becomes optional in trace mode, where it now only
    names the output subdirectory. With neither, generation raises and names both
    flags. This is what item 10's "'any PDDL domain' holds for generation" was
    always supposed to mean; before this it did not, since the domain file could
    only be reached through `_DOMAIN_REGISTRY`. Item 10's second half still
    stands unchanged: *running* an experiment on the corpus still needs a
    `domains.<key>` entry.
23. **A window that ends where it began is dropped.** Deduplication was on
    `(init fluents, final fluents)`, which is exactly the signature that makes a
    closed loop indistinguishable from a no-op: emitted, it becomes a problem
    whose goal is its own initial state, solvable by the empty plan, teaching a
    learner nothing while counting against `num_problems`. `cutter.is_closed_loop`
    rejects it on the same path as a duplicate — steps consumed, window dropped.
    Separately, `cut()` now rejects `length_range` in `NONE` mode rather than
    ignoring it, so a caller who thinks they are cutting by length finds out.
24. **A short corpus says so where it will be read.** Item 11 settled that a
    shortfall warns and continues. The warning was a `logger` call, which lands
    on stderr while the banners go to stdout; in a real run it surfaces
    thousands of lines from the summary it qualifies, and the summary printed a
    bare `Problems: 2` with no mention of the 10 that were asked for. It is now
    reported three times over: the log line, `short_of_requested` plus the
    existing `num_problems` vs `num_windows` pair in `generation_info.json`, and
    `*** SHORT: asked for N ***` in the completion banner. The walk's own early
    exits — no applicable action, no plan found, a planned action that would not
    apply — are `WARNING`, not `DEBUG`, for the same reason.
25. **The manifest must be readable without knowing the run's cwd.** It exists
    only to be read later (it has no programmatic consumer by design), so a
    relative `"domain_file": "src/domains/blocks/blocks.pddl"` copied verbatim
    from the command line defeats its one purpose. Both sources `.resolve()`
    every path they record. `describe()` also gained `max_planning_time`,
    `max_replanning_time` and `max_random_trials`: all three can change the trace
    and none were recorded, so a `preserve_solvability` corpus was not
    reproducible from its own record.

---

## 1. What exists today (read this before changing anything)

### 1.1 `benchmark/data_generator.py`

- `generate_trajectories(...)` — **predefined mode**. Walks pre-authored problem
  dirs. PDDLGym domains: `handler.run_pipeline` renders + generates GT, then LLM
  inference. External domains (depot, gripper): copies supplied images + GT, then
  infers in place. **Untouched by this plan** — depot/gripper images cannot be
  reproduced by any simulator.
- `generate_trajectories_via_generation(...)` — **generate mode**. Drives
  `PDDLGymProblemGenerator`, then `_run_generation_inference` per folder
  (exports true GT to `gt_trajectories/` *before* the classifier rebuild
  overwrites the JSON — preserve this ordering).

### 1.2 `src/trajectory_handlers/pddlgym_problem_generator.py`

`PDDLGymProblemGenerator(gym_domain_name, problem_index)` does
`pddlgym.make(env_id)` then `fix_problem_index(...)`; the bundled problem fixes
the object set **and its goal is discarded**.

`generate(...)` → `_walk_and_write(...)`: reset, render frame 0, then loop until
`num_problems` folders exist or the walk reaches `cursor_steps_limit` —

1. sample `length ~ U(length_range)`;
2. `_extend_walk` steps with `_sample_state_changing_action` (random, no no-ops),
   rendering each state into a scratch dir;
3. take `walk[cursor : cursor + length]`, then `cursor += length + skip`;
4. dedupe on `_window_signature` = `(frozenset(init literals), frozenset(final
   literals))`; duplicates are dropped but their steps are kept;
5. `_write_problem` → `.pddl` (objects from `init_obs.objects`, init =
   `_fluent_literals(init_obs)`, **goal = the window's full final state**),
   `plan.txt`, `state_*.png`, GT `.trajectory` + `_trajectory.json`.

**Everything from `_window_signature` downward is already source-agnostic.** It
just does not know it yet. That is the whole refactor.

### 1.3 `benchmark/evaluation/test_states_generator.py` — the engine you want

Already builds trajectories from a problem with **no PDDLGym involvement**:
`unified_planning`'s `PDDLReader`, `SequentialSimulator`
(`get_initial_state` / `apply` / `_is_applicable`), `OneshotPlanner` on Fast
Downward (`_DOWNWARD_SEARCH_CFG`), plus `tarski`'s `LPGroundingStrategy` for
grounding. It follows the AMLGym procedure: plan, execute, substitute a random
applicable action with probability `p_rnd`, replan, collect states.

**`p_rnd` is the single dial that gives all three strategies:** `0` = planner,
`1` = random walk, in between = the AMLGym-faithful mixed walk.

Its one gap for our purposes: `_generate_trajectory` accumulates only `states`
and **discards `action_instance`** as it applies it. A trace needs the actions.

### 1.4 Consumers

`SimulatedDataSource` reads `gt_trajectories/` plus the problem `.pddl` from the
training-pool dir; `benchmark_runner._discover_gt_trajectories` locates them.
A symbolic corpus therefore needs exactly those two artifacts to be complete.

---

## 2. Target architecture

```
src/trace_generation/
  trace_step.py   TraceState / TraceStep — the one record both sources produce
  sources.py      TraceSource → (steps(), domain_name, describe())
                  ├─ ProblemWalkSource(problem_file, domain_file,
                  │                    backend={native|pddlgym}, p_rnd, seed)
                  └─ TrajectorySource(trajectory_file, problem_file, domain_file)
                  plus walk_problem(), the shared AMLGym loop test_states_generator
                  now calls
  cutter.py       cut(steps, mode, skip, num_problems, seed) → [Window]
  emitter.py      emit_window(window, ...) → problemN/ in both trees
                  build_generation_info / write_generation_info → the manifest
  corpus.py       build_source() + generate_corpus() — the one place the three
                  meet; returns a Corpus describing what landed
  up_bridge.py    UP State / ActionInstance   → the eval schema (native walk)
  pddl_bridge.py  pddl-plus-parser State / ActionCall → the eval schema (replay)
  eval_schema.py  the one definition of how a literal, an action and a typed
                  object are spelled, so the two bridges cannot drift
```

The two bridges exist because the two sources speak different libraries — the
native walk holds `unified_planning` objects, the replay holds
`pddl-plus-parser` ones — and neither library's rendering matches the eval
schema. `eval_schema.py` is what makes their outputs comparable.

### 2.1 The one type that keeps this compact

Both backends must produce the **same** step record, so the cutter and emitter
never branch on source. `PDDLGymProblemGenerator`'s `WalkStep` was already 90%
of it; `trace_step.py` is the remaining 10%.

As built, states and actions are **strings in the eval schema**, not library
objects — the bridge happens at the source, so nothing downstream of `TraceStep`
imports `unified_planning` or `pddl-plus-parser`:

```python
@dataclass(frozen=True)
class TraceState:
    literals: Tuple[str, ...]   # positive fluents, "on(a:block,b:block)"
    objects: Tuple[str, ...]    # the universe, "a:block"

@dataclass(frozen=True)
class TraceStep:
    prev_state: TraceState
    action: str                 # "stack(a:block, b:block)"
    next_state: TraceState
    frame_before: Optional[Path] = None   # None for symbolic sources
    frame_after:  Optional[Path] = None
```

Both are frozen with tuple fields, so a window can be hashed and deduplicated
without defensive copying. Note the schema's deliberate asymmetry: literal
arguments are joined with `","`, action arguments with `", "`. That is what the
eval code already expects; `eval_schema.py` is the single place it is spelled.

Get this right first. Every "how do I share X?" question downstream dissolves
into "produce a `TraceStep`".

---

## 3. Sources

### 3.1 `ProblemWalkSource(backend="native")` — the default

Lift the walk loop out of `test_states_generator._generate_trajectory` into
`sources.py`, with two changes:

- **record the action** alongside each state, emitting `TraceStep`s;
- make the length cap and the solvability-preserving replanning configurable —
  the corpus generator wants a plain walk; S_test wants the replanning.

`test_states_generator` then calls the shared walk and keeps its own S_test
post-processing. Do not merge their *purposes*; share only the walk.

Requires a planner engine ~~— already a hard dependency (`problem_solving` in every
experiment's evaluation uses one)~~ **whenever `p_rnd < 1`**, per item 12. At
`p_rnd=1` the planner is never invoked and the walk runs without one.

Two details the walk gets right and this section did not state. Its early exits
— no applicable action, no plan within `max_planning_time`, a planned action
that turns out inapplicable — log at `WARNING` with the step count reached, so a
truncated walk is visible rather than merely short (item 24). And a random
action that leaves the state unchanged is rejected on **both** random paths, the
plain one and the solvability-preserving one, so a no-op never becomes a trace
step; the gym generator's `_sample_state_changing_action` had that property and
losing it would have silently changed what a window's `length` means.

### 3.2 `ProblemWalkSource(backend="pddlgym")` — renderer only

`PDDLGymProblemGenerator` keeps `_extend_walk` + env setup and nothing else. To
accept an arbitrary problem rather than a bundled index, construct
`PDDLEnv(domain_file, tmpdir_containing_that_problem, render=...)` directly
instead of `pddlgym.make()` + `fix_problem_index()`; borrow the render function
from a registered env. **Only needed when images are wanted**, so it is not on
the critical path for this plan.

### 3.3 `TrajectorySource(trajectory_file, problem_file)`

Parse with `TrajectoryParser(domain, problem)`, emit `TraceStep`s with frames
`None`. Object universe and types come from the problem's `:objects` — **merged
with the domain's `:constants`**, because `ProblemParser` does not fold constants
into `problem.objects` (UP's `PDDLReader` does, which is why `up_bridge` needs no
equivalent). A domain declaring constants would otherwise emit windows whose
`:objects` block omits objects their own literals mention.

Frames were **built and left off by default**, not left unwired: `attach_frames`
on the source plus `render` on the corpus copies a window's contiguous
`state_*.png` slice out. Both default false, so the symbolic-only decision holds
unless a caller asks otherwise.

---

## 4. Cutter

```python
def cut(steps: Iterable[TraceStep], *, mode, length_range=None,
        skip=1, num_problems=None, seed=None,
        exclude_signatures=()) -> List[Window]
```

`steps` is an `Iterable`, pulled lazily, never a materialized list: a walk with
`max_steps=1000` cut into 5 windows must not walk 1000 steps first. `UNIFORM`
stops pulling once `num_problems` windows are accepted; `NONE` drains the stream
by definition.

- `mode="none"` — one window spanning every step.
- `mode="uniform"` — today's behaviour: `length ~ U(*length_range)`.
- ~~`mode="buckets"` — cycle the `buckets` list (or sample it) for each window's
  length. All windows go into **one pool**; the length is metadata.~~
  **Deleted, per item 20.**
- `skip` states discarded between consecutive windows.
- Dedupe on `(frozenset(init fluents), frozenset(final fluents))` — port
  `_window_signature` verbatim. Duplicates are dropped, their steps consumed.
- **Reject closed loops**, per item 23: a window whose signature has
  `init == final` is dropped on the same path as a duplicate. It is a problem
  the empty plan solves.
- Stop at `num_problems` windows or when the step stream is exhausted. When the
  stream runs dry early, warn with the count, as `_walk_and_write` does today.
  Reporting the shortfall is the *caller's* job, not the cutter's — `cut()`
  returns a short list and `generate_corpus` decides what to say about it
  (item 24).
- **Reject contradictory arguments** rather than silently ignoring them:
  `length_range` is required by `UNIFORM` and refused by `NONE`; `skip` and
  `num_problems` must be non-negative; `mode` must be a `CutMode`.

The cutter must be **pure** — a list of `TraceStep` in, a list of `Window` out.
No I/O, no env. That is what makes it testable and what stops it drifting.

---

## 5. Emitter

Reproduce `_write_problem`, `_write_problem_pddl`, `_write_plan`, `_write_images`
from `PDDLGymProblemGenerator` in `emitter.py`, generalized off `TraceStep`.
They are reimplemented rather than moved, because step 5 — which would delete the
originals — is out of scope (item 19), so for now the two copies coexist:

- `.pddl` — objects from the window's initial state, `init` = window's first
  state fluents, **`goal` = the window's full final state** (keep this
  semantics; it is what makes each window an independently solvable problem).
- `plan.txt` — the window's ground actions, one per line.
- GT `.trajectory` + `_trajectory.json`.
- `state_*.png` only when `render=True` **and** the source supplied frames.

Plus one **new** artifact, `generation_info.json` at corpus level. As built it is
the source's own `describe()` merged with the cut parameters, so a walk corpus
carries `backend` / `p_rnd` / `seed` / `max_steps` and a replay corpus carries
`trajectory_file` / `attach_frames` instead — the manifest describes whichever
source actually ran rather than a union with nulls in it:

```json
{
  "source_kind": "problem",
  "source_file": "/abs/path/problem1.pddl",
  "domain_file": "/abs/path/blocks.pddl",
  "backend": "native", "p_rnd": 1.0, "seed": 7, "max_steps": 60,
  "preserve_solvability": false, "stop_at_goal": false,
  "max_planning_time": 120, "max_replanning_time": 60, "max_random_trials": 3,

  "cut_mode": "uniform", "length_range": [4, 7],
  "skip": 1, "num_problems": 5, "cut_seed": 3, "render": false,
  "domain": "blocksworld",

  "uniform_object_universe": true,
  "object_signature": ["a:block", "b:block", "c:block", "d:block"],

  "num_windows": 5,
  "short_of_requested": false,
  "windows": [{"name": "problem0", "index": 0, "length": 6, "source": "..."}]
}
```

~~This is what lets a mixed-bucket pool be grouped by length at report time, and
it is what the experiment-reporting work will read to describe a corpus.~~
**Neither survived.** Buckets are gone (item 20), and the reporting consumer was
never written — the manifest is a **write-only provenance record**, read by a
human asking "what produced this corpus?" and by nothing else. That is a
decision, not an omission, and it is what item 25 follows from: a record nothing
parses has to be self-sufficient on its face, hence absolute paths and every
knob that can move the trace, including the three planner timeouts that were
initially left out.

Four fields carry more than they look like. `num_problems` is what was *asked
for* and `num_windows` what was *produced*, with `short_of_requested` stating
the comparison outright so item 11's warned shortfall stays legible after the
fact instead of a short corpus looking like the intended one.
`object_signature` is gotcha §10.5's uniform-universe claim written down, so it
can be checked rather than assumed once item 14 lands. And `seed` / `cut_seed`
are separate because the RNGs are (§10.4).

`domain` arrives via `extra_info` from the `data_generator` shell, not from the
source: it is the output subdirectory's name, which is either `--domain` or,
absent that, the domain name parsed out of the PDDL (item 22).

---

## 6. UP bridge

`up_bridge.py` converts `UPState` + `ActionInstance` into our formats. This is
the only genuinely new logic in the plan; everything else is a move or a rewire.
Get the object-name and predicate-arity conventions right against a domain that
already has a known-good corpus on disk, and diff.

**As built there are two bridges, not one**, because the replay source never
holds a UP object: `pddl_bridge.py` does the same job for `pddl-plus-parser`'s
`State` and `ActionCall`. Neither owns the conventions — `eval_schema.py` does,
and both call it, which is what stops the two paths emitting subtly different
spellings of the same state. Verification is a round trip rather than a diff
against disk: emitted `.pddl` and `.trajectory` files are re-read by the real
`ProblemParser` / `TrajectoryParser` in `test_emitter.py`, and §8.1's gripper
fold confirms the replay path against a real learner.

---

## 7. Entry point and config

The composition itself is `src/trace_generation/corpus.py:generate_corpus`, not a
`data_generator` function: it needs no config, so keeping it in the library lets
it be unit-tested directly. `data_generator.generate_trajectories_via_trace` is
the shell around it — ~~config resolution~~ **domain-file resolution** (item 21),
output-dir naming, banners — reached by `--gen-mode trace`.

`generate_trajectories_via_generation` was **not** made a preset over it; that
is §8 step 5, excluded (item 19). The two entry points are still independent.

### 7.1 ~~`config.yaml` keys~~ CLI flags

**Rewritten per item 21: trace mode has no config keys.** Every setting below is
a flag with a default declared once, in
`generate_trajectories_via_trace`'s signature. There is no config fallback and
therefore no override-precedence rule to remember, and nothing is "required in
config or on the CLI" — a required setting is a required *flag*.

| Flag | Values | Default |
|---|---|---|
| `--source-kind` | `problem` \| `trajectory` | **required** |
| `--problem-file` | path | **required** |
| `--trajectory-file` | path | required iff `--source-kind trajectory` |
| `--domain-file` | path | from `--domain` via the registry; one of the two is **required** |
| `--domain` | a `_DOMAIN_REGISTRY` key | none — names the output subdir only |
| `--cut-mode` | `none` \| `uniform` | `uniform` |
| `--num-problems` | int | 10 |
| `--length-min` / `--length-max` | int | 9 / 20 |
| `--skip` | int | 1 |
| `--problem-prefix` | str | `problem` |
| `--backend` | `native` \| `pddlgym` | `native` |
| `--p-rnd` | float in `[0, 1]` | 1.0 |
| `--max-steps` | int | 1000 |
| `--preserve-solvability` | flag | off |
| `--stop-at-goal` | flag | off |
| `--max-planning-time` | int (seconds) | 120 |
| `--max-replanning-time` | int (seconds) | 60 |
| `--max-random-trials` | int | 3 |
| `--render` / `--no-render` | bool | false |
| `--seed` / `--cut-seed` | int | none / `--seed` |

`--render` is no longer tri-state: with no config value to override, `default=None`
bought nothing, so it is a plain `False`.

Four of these flags — `--num-problems`, `--length-min`, `--length-max`, `--skip`
— are shared with `--gen-mode generate`, which *does* read config and needs
`None` to mean "fall back to it". They therefore keep `default=None` at the
argparse layer, and trace mode drops the `None`s before the call so the
signature's defaults apply. The defaults are still written down exactly once.

Trace mode consults `config.yaml` for one thing only: resolving a registered
`--domain` to its `domains.<key>.domain_file`. With `--domain-file` it does not
read config at all.

**What remains under `domains.<key>.generation` belongs to `--gen-mode generate`**,
which is unchanged: `num_problems`, `length_min`, `length_max`, `skip`, and
`from_pddlgym.default_problem_index`. That last one *does* move into the
`from_pddlgym:` sub-block as this section always said — `_resolve_problem_index`
has read that path since it lost its silent default (§9), but the shipped
`config.yaml` still had it flat under `generation:` for `blocksworld`, `hanoi`
and `npuzzle`, so `--gen-mode generate` without an explicit `--problem-index`
raised a `KeyError` for all three. Fixed in both files.

Two seeds, never one, per gotcha §10.4: `--seed` drives the walk and `--cut-seed`
the window lengths. `--cut-seed` defaults to `--seed`, so one flag still
reproduces a corpus, and `generation_info.json` records both as `seed` and
`cut_seed` so the manifest always shows which was used.

`config.yaml` is gitignored, so `config.example.yaml` is the tracked copy of the
generate-mode block; it carries it for `blocksworld` and `hanoi`. **A fix applied
only to `config.yaml` does not propagate through a branch** — the `KeyError`
above has to be reapplied by hand in every checkout.

---

## 8. Order of work — the risk lands last

1. `TraceStep` + `cutter.py` + `emitter.py`, with the cutter unit-tested in
   isolation (pure function, trivial to test).
2. `up_bridge.py` + `ProblemWalkSource(backend="native")`.
3. `TrajectorySource`.
4. **Prove it**: generate a symbolic corpus, run a simulated experiment on it
   end to end. This is the acceptance gate for the new path. **Passed** — see
   §8.1.
5. **Only then** retrofit `PDDLGymProblemGenerator` onto the shared cutter and
   emitter, and verify by regenerating
   `benchmark/data/npuzzle/npuzzle_generated_problem0__final-version_fixed`
   at the same seed and diffing against what is on disk.
6. Arbitrary-problem injection for the pddlgym backend (§3.2) — last, and only
   if imaged generation from foreign problems is actually wanted.

### 8.1 The acceptance gate, as run

Both sources were taken through generation *and* one simulated CDPS fold, since
a corpus that parses is not the same as a corpus a learner can consume:

| | walk source | replay source |
|---|---|---|
| domain | blocksworld | gripper |
| corpus | 10 problems, `uniform(9,20)`, `--seed 17 --cut-seed 5` | 5 problems, `uniform(2,3)`, `--cut-seed 5` |
| fold | `--n-folds 5 --folds 0 --num-trajectories 3`, mask 0.1 / noise 0.2 | same |
| precision / recall | 0.87 / 0.88 | 0.77 / 0.77 |
| search | 1457 nodes, 10 conflict-free models, hit its timeout | 1513 nodes, 3 models, **exhausted** |

The replay arm exhausting its search rather than timing out is the load-bearing
result: it means every observation grounded, so the eval-schema round trip
through `pddl_bridge` holds against a real learner and not only against tests.
Its `solving_ratio` of 0 follows from 8 transitions cut into 2–3 step windows —
a property of the only trajectory on disk long enough to cut, not of the path.

Two things the gate exposed, neither a defect in this work:

- `experiment_runner`'s `batch_timeout = n_jobs × learning_timeout × 2` can
  expire while a fold's *evaluation* is still running, after the fold itself has
  written `fold_result.json`. The run then reports `Total result rows: 0` while
  the results are on disk; `--resume` reloads them and reports 1. This is
  pre-existing and applies to any corpus.
- No `.trajectory` in `src/domains/` exceeds ~20 steps, so `cut_mode=uniform`
  with realistic lengths yields 2–3 windows from a replay source. Replay corpora
  large enough for a fold need either short windows or item 14's multi-input
  support.

---

## 9. Files touched

| File | Change |
|---|---|
| `src/trace_generation/{trace_step,sources,cutter,emitter,corpus,up_bridge,pddl_bridge,eval_schema}.py` | **new**, each with a colocated `test_*.py` |
| `src/utils/pddl_trajectory.py` | gains `export_gt_trajectory`, moved out of `gt_builder` (which re-exports it) so `src/` no longer needs `benchmark/` to write GT |
| `benchmark/experiment_running_helpers/gt_builder.py` | re-exports the moved function; no behaviour change |
| `benchmark/evaluation/test_states_generator.py` | walk loop moves to the native source; keeps S_test post-processing |
| `benchmark/data_generator.py` | `generate_trajectories_via_trace` + `--gen-mode trace` CLI; `_resolve_problem_index` loses its silent default. **Revision:** `_REQUIRED` and `_generation_setting` deleted, replaced by `_resolve_trace_domain_file`; `--domain` relaxed to optional in trace mode; `--domain-file` and the five walk knobs added |
| `config.yaml` / `config.example.yaml` | ~~generation keys above~~ **only the generate-mode block survives** (item 21); `default_problem_index` nested under `from_pddlgym` in both, which is a bug fix, not a tidy-up (§7.1) |

**Not touched:** `src/trajectory_handlers/pddlgym_problem_generator.py` (§8
step 5, excluded), anything under `src/milp/` or `src/plan_denoising/`.

`gt_builder.py` is the one file under `benchmark/experiment_running_helpers/`
that this work modifies. This table originally listed that whole directory as
untouched, on the grounds that it is where the concurrent ROSAME-I+MILP work
lives. The change is a re-export line: the function had to move so the emitter
could call it without `src/` importing `benchmark/`. No behaviour that branch
depends on is altered.

---

## 10. Gotchas

1. **`test_states_generator` uses `simulator._is_applicable`** — a private UP
   API. It already works; preserve the call rather than "improving" it, and note
   the version coupling in a comment.
2. **Solvability.** The AMLGym procedure replans to keep problems solvable after
   a random action. A pure random walk does not need that, and forcing it makes
   generation much slower. Make it a flag, defaulting off for corpus generation.
   **Built as `preserve_solvability`, and now reachable and tested.** It was
   plumbed through `walk_problem` but not through `build_source`, so no caller
   could switch it on and no test covered the branch it guards: on, a random
   action is kept only if the planner still finds a plan from the state it
   produces, retrying up to `max_random_trials` times. It is now a
   `build_source` argument and a `--preserve-solvability` flag, with the
   replanned plan reused rather than discarded. Still off by default, for the
   cost reason above.
3. **GT export ordering** in the imaged path: GT is exported *before* the
   classifier rebuild overwrites `_trajectory.json`. Any refactor of
   `_run_generation_inference` must preserve that ordering or `gt_trajectories/`
   silently fills with classifier output.
4. **Seeding.** Today two RNGs matter — `random.Random(seed)` for window lengths
   and the env's own seed for the walk. The native backend needs the equivalent
   split (walk RNG vs cutter RNG) or "same seed" will not mean the same corpus.
5. **Object universe.** A corpus cut from one source problem is automatically
   uniform, which is what ROSAME-I requires. If multi-problem corpora are ever
   added, that property is lost — record the object signature in
   `generation_info.json` so the constraint can be checked rather than assumed.

---

## 11. Deliberately out of scope

- Imaged output from the new paths.
- Predefined mode (`generate_trajectories`) and the external depot/gripper flow.
- Folding `test_states_generator`'s S_test post-processing into the shared
  pipeline — it has a different purpose and does not want the cutter. Only its
  *walk* moved (item 13); everything downstream of the walk stayed.
- §8 steps 5 and 6, per item 19.
- Multi-input corpora, per item 14.
- Bucketed length modes, per item 20.
- Any programmatic consumer of `generation_info.json`. It is a provenance
  record for a human to read, and §5's "the experiment-reporting work will read
  it" is withdrawn.
