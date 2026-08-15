# Implementation Plan: Generic Trace Generation

> **Audience**: an implementing agent with access to this repo
> (`VIP-vision-PDDL`).
>
> **Goal**: make trajectory generation source-agnostic and configurable, so a
> corpus of arbitrary size and length distribution can be produced from **either**
> a problem file (walked with a planner, a random walk, or a mix) **or** an
> existing trajectory file — for **any** PDDL domain, not only the ones PDDLGym
> ships. Symbolic output only; images remain the existing pipeline's job.

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
4. **Length modes**: `none` | `uniform(min, max)` | `buckets([...])`.
   Buckets emit into a **single mixed pool**, with each window's length recorded
   in metadata — *not* one directory per length.
5. **Input trajectories are ground truth.** Windows land in `gt_trajectories/`;
   degradation stays the experiment's job (`SimulatedDataSource` injects masking
   and noise per fold). A noisy input trajectory must be rejected or loudly
   flagged, never silently treated as GT.
6. **A trajectory input requires its problem `.pddl`.** Two independent reasons:
   a `.trajectory` carries object *names* but not their *types*, so a valid
   `:objects` block cannot be written from it alone; and
   `TrajectoryParser(domain, problem).parse_trajectory(...)` takes a `problem`
   argument, so the file cannot even be read without one. Fail loudly.
7. **Option B for the existing code**: share the cutter and emitter with the
   PDDLGym path; keep the sources separate. Nothing is deleted.

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
  sources.py    TraceSource → (steps, objects, domain)
                ├─ ProblemWalkSource(problem_file, domain_file,
                │                    backend={native|pddlgym}, p_rnd, seed)
                └─ TrajectorySource(trajectory_file, problem_file)
  cutter.py     cut(steps, mode, skip, num_problems, seed) → [Window]
  emitter.py    emit(window, out_dir, name, render) → problemN/
  up_bridge.py  UPState / ActionInstance  ↔  our .trajectory / .pddl formats
```

### 2.1 The one type that keeps this compact

Both backends must produce the **same** step record, so the cutter and emitter
never branch on source. Today's `WalkStep` is already 90% of it:

```python
@dataclass
class TraceStep:
    prev_state: State          # our state repr, not pddlgym's / UP's
    action: GroundedAction
    next_state: State
    frame_before: Optional[Path] = None   # None for symbolic sources
    frame_after:  Optional[Path] = None
```

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

Requires a planner engine — already a hard dependency (`problem_solving` in every
experiment's evaluation uses one).

### 3.2 `ProblemWalkSource(backend="pddlgym")` — renderer only

`PDDLGymProblemGenerator` keeps `_extend_walk` + env setup and nothing else. To
accept an arbitrary problem rather than a bundled index, construct
`PDDLEnv(domain_file, tmpdir_containing_that_problem, render=...)` directly
instead of `pddlgym.make()` + `fix_problem_index()`; borrow the render function
from a registered env. **Only needed when images are wanted**, so it is not on
the critical path for this plan.

### 3.3 `TrajectorySource(trajectory_file, problem_file)`

Parse with `TrajectoryParser(domain, problem)`, emit `TraceStep`s with frames
`None`. Object universe and types come from the problem's `:objects`. If the
source folder happens to carry `state_*.png`, slicing them per window is a
contiguous-slice one-liner — **wire the plumbing but leave it off**, per the
symbolic-only decision.

---

## 4. Cutter

```python
def cut(steps, *, mode, length_range=None, buckets=None,
        skip=1, num_problems=None, seed=None) -> List[Window]
```

- `mode="none"` — one window spanning every step.
- `mode="uniform"` — today's behaviour: `length ~ U(*length_range)`.
- `mode="buckets"` — cycle the `buckets` list (or sample it) for each window's
  length. All windows go into **one pool**; the length is metadata.
- `skip` states discarded between consecutive windows.
- Dedupe on `(frozenset(init fluents), frozenset(final fluents))` — port
  `_window_signature` verbatim. Duplicates are dropped, their steps consumed.
- Stop at `num_problems` windows or when the step stream is exhausted. When the
  stream runs dry early, warn with the count, as `_walk_and_write` does today.

The cutter must be **pure** — a list of `TraceStep` in, a list of `Window` out.
No I/O, no env. That is what makes it testable and what stops it drifting.

---

## 5. Emitter

Port `_write_problem`, `_write_problem_pddl`, `_write_plan`, `_write_images`,
`_write_gt_trajectory` from `PDDLGymProblemGenerator` into `emitter.py`,
generalized off `TraceStep`:

- `.pddl` — objects from the window's initial state, `init` = window's first
  state fluents, **`goal` = the window's full final state** (keep this
  semantics; it is what makes each window an independently solvable problem).
- `plan.txt` — the window's ground actions, one per line.
- GT `.trajectory` + `_trajectory.json`.
- `state_*.png` only when `render=True` **and** the source supplied frames.

Plus one **new** artifact, `generation_info.json` at corpus level:

```json
{
  "source_kind": "problem" | "trajectory",
  "source_file": "...",
  "backend": "native" | "pddlgym",
  "p_rnd": 0.0, "seed": 42,
  "cut_mode": "buckets", "buckets": [5, 10, 20, 40], "skip": 1,
  "windows": [{"name": "problem0", "length": 10, "index": 0}, ...]
}
```

This is what lets a mixed-bucket pool be grouped by length at report time, and
it is what the experiment-reporting work will read to describe a corpus.

---

## 6. UP bridge

`up_bridge.py` converts `UPState` + `ActionInstance` into our formats. This is
the only genuinely new logic in the plan; everything else is a move or a rewire.
Get the object-name and predicate-arity conventions right against a domain that
already has a known-good corpus on disk, and diff.

---

## 7. Entry point and config

One `data_generator` function composing source → cutter → emitter.
`generate_trajectories_via_generation` becomes a thin preset over it
(`backend="pddlgym"`, `render=True`, then LLM inference).

New keys in `config.yaml` under `domains.<d>.generation`, alongside the existing
`num_problems` / `length_min` / `length_max` / `skip` / `default_problem_index`:

| Key | Values |
|---|---|
| `source_kind` | `problem` \| `trajectory` |
| `problem_file` / `trajectory_file` | path |
| `backend` | `native` \| `pddlgym` |
| `p_rnd` | float in `[0, 1]` |
| `render` | bool |
| `cut_mode` | `none` \| `uniform` \| `buckets` |
| `buckets` | list of ints |

All CLI-overridable, matching the existing `--num-problems` / `--length-min`
style in `data_generator.py`'s argparse block.

---

## 8. Order of work — the risk lands last

1. `TraceStep` + `cutter.py` + `emitter.py`, with the cutter unit-tested in
   isolation (pure function, trivial to test).
2. `up_bridge.py` + `ProblemWalkSource(backend="native")`.
3. `TrajectorySource`.
4. **Prove it**: generate a symbolic corpus, run a simulated experiment on it
   end to end. This is the acceptance gate for the new path.
5. **Only then** retrofit `PDDLGymProblemGenerator` onto the shared cutter and
   emitter, and verify by regenerating
   `benchmark/data/npuzzle/npuzzle_generated_problem0__final-version_fixed`
   at the same seed and diffing against what is on disk.
6. Arbitrary-problem injection for the pddlgym backend (§3.2) — last, and only
   if imaged generation from foreign problems is actually wanted.

---

## 9. Files touched

| File | Change |
|---|---|
| `src/trace_generation/{sources,cutter,emitter,up_bridge}.py` | **new** |
| `src/trajectory_handlers/pddlgym_problem_generator.py` | keep the walk; move cut/emit out |
| `benchmark/evaluation/test_states_generator.py` | walk loop moves to the native source; keeps S_test post-processing |
| `benchmark/data_generator.py` | new composed entry point + CLI |
| `config.yaml` | generation keys above |

**Not touched:** anything under `benchmark/experiment_running_helpers/`,
`src/milp/`, `src/plan_denoising/`. Zero file overlap with the ROSAME-I+MILP work.

---

## 10. Gotchas

1. **`test_states_generator` uses `simulator._is_applicable`** — a private UP
   API. It already works; preserve the call rather than "improving" it, and note
   the version coupling in a comment.
2. **Solvability.** The AMLGym procedure replans to keep problems solvable after
   a random action. A pure random walk does not need that, and forcing it makes
   generation much slower. Make it a flag, defaulting off for corpus generation.
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
  pipeline — it has a different purpose and does not want the cutter.
