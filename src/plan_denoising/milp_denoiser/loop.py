"""The ``pisam_milp_loop`` driver: many small solves instead of one big one.

``single_round`` hands every trace to one solve. That is the strongest possible
joint repair, but it is also a single point of failure: on a big fold the solve
is slow, and whatever it returns is the only answer there will ever be.

This loop trades that for many cheap solves (plan §2)::

    M_best <- none; min_V <- inf
    repeat:
        S_r  <- sample(pool)                  # a subset of the traces
        T'_r <- MILP(S_r, prior=M_best)       # repair just those
        M_r  <- PISAM(learner input)          # a candidate model
        V_r  <- Evaluate(M_r, ORIGINAL obs)   # ground-truth-free score
        if V_r < min_V: M_best <- M_r

Three things make it work, and each is a decision rather than an accident:

1. **V is measured against the ORIGINAL noisy observations, always.** They are
   the only fixed reference available without ground truth. A model is judged by
   how well it re-explains what was actually seen, never by how well it explains
   the repairs it caused.
2. **The pool is frozen by default** (:class:`config.PoolPolicy`). Writing
   repairs back into the pool creates a ratchet — round r's repair becomes round
   r+1's evidence — so an early wrong repair can never be argued away by the
   data it overwrote.
3. **The incumbent is the prior for the next solve.** Without it, equally-cheap
   repairs make consecutive rounds wander between unrelated explanations; with
   it, a round has to *earn* a departure from the incumbent.

Because 2 keeps the input to a round fixed and ``encoder.solve`` pins CP-SAT's
seed and worker count, a round is fully identified by ``(subset, M_best)``. That
gives free dedup and a real fixpoint stop rule: once every subset has been
solved against the current incumbent, no further round can produce anything new.
Both are switched off automatically under ``pool_policy: replace``, where the
identity no longer holds.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from pddl_plus_parser.models import Domain, Observation
from sam_learning.core import LearnerDomain
from utilities import NegativePreconditionPolicy

from constraint_opt.factory import resolve as resolve_encoder
from planning_structs.traces import Traces

from src.plan_denoising.evaluator import (
    EvaluationResult,
    EvaluationWeights,
    observations_reconstruction_score,
)
from src.milp import encoder as _encoder_module  # noqa: F401  (factory registration)
from src.plan_denoising.milp_denoiser.config import (
    PisamMilpConfig,
    LearnerInput,
    PoolPolicy,
    Sampler,
)
from src.milp.converter import (
    build_ps_domain,
    build_ps_instance_from_objects,
    observation_to_trace,
    try_observation_to_trace,
)
from src.plan_denoising.milp_denoiser.model_prior import (
    learner_domain_to_observation_m,
)
from src.plan_denoising.milp_denoiser.single_round import _learn_with_pisam
from src.plan_denoising.milp_denoiser.trajectory_extraction import (
    extract_repaired_observations,
)

# Both channels: the loop always has a reference model from round 2 onward, and
# round 1 drops the model channel via ``encoding_config(has_prior=False)``.
_OBJECTIVES_WITH_PRIOR = {"state", "model"}
_OBJECTIVES_NO_PRIOR = {"state"}

# How many times to redraw before declaring the sampler unable to find a subset
# it has not already solved against the current incumbent. Only reached when the
# exact fixpoint test (all C(n, m) subsets seen) is impractical to hit by luck.
_MAX_RESAMPLE_ATTEMPTS = 50

NO_MODEL_HASH = "none"

# Where the per-round candidate models land, one ``round_{i}/model.pddl`` each.
# A sibling of ``milp_loop_rounds.json``: the log says what each round scored,
# this directory holds the model it scored, so a round can be re-evaluated
# offline against metrics the loop itself never computed.
ROUND_MODELS_DIR = "milp_loop_round_models"
#: Streamed one JSON row per round, flushed as the round ends. The
#: end-of-run ``milp_loop_rounds.json`` stays the artifact readers parse;
#: this is what makes V visible while the loop is still running.
ROUND_STREAM_NAME = "milp_loop_rounds.jsonl"


# ---------------------------------------------------------------- round record


@dataclass
class RoundLog:
    """One row of the loop's history — the raw material for the P5 anytime plot.

    Every field is either an input to the round or a measurement of it; nothing
    here is derived from ground truth.
    """

    round_index: int
    subset: List[int]
    prior_hash: str
    elapsed_seconds: float
    round_seconds: float
    solved: bool = False
    # Flips on THIS round's subset only — not comparable to the single round's
    # fold-wide cost, and under ``pool_policy: replace`` it is an increment over
    # the previous repair rather than a total. See ``LoopResult.as_report``.
    repair_cost: int = 0
    model_hash: str = NO_MODEL_HASH
    v_raw: Optional[float] = None
    v_per_transition: Optional[float] = None
    effect_mismatches: Optional[int] = None
    inapplicability_events: Optional[int] = None
    success_rate: Optional[float] = None
    improved: bool = False
    tied_with_best: bool = False
    mixed_set_conflict: bool = False
    pisam_conflicts: int = 0
    learner_input_size: int = 0
    solve_seconds: Optional[float] = None
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """JSON-ready row."""
        return {
            "round": self.round_index,
            "subset": list(self.subset),
            "prior_hash": self.prior_hash,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "round_seconds": round(self.round_seconds, 3),
            "solve_seconds": self.solve_seconds,
            "solved": self.solved,
            "repair_cost": self.repair_cost,
            "model_hash": self.model_hash,
            "v_raw": self.v_raw,
            "v_per_transition": self.v_per_transition,
            "effect_mismatches": self.effect_mismatches,
            "inapplicability_events": self.inapplicability_events,
            "success_rate": self.success_rate,
            "improved": self.improved,
            "tied_with_best": self.tied_with_best,
            "mixed_set_conflict": self.mixed_set_conflict,
            "pisam_conflicts": self.pisam_conflicts,
            "learner_input_size": self.learner_input_size,
            "note": self.note,
        }


@dataclass
class LoopResult:
    """What the loop returns: the incumbent, and the history that justifies it."""

    learned_domain: Optional[LearnerDomain]
    conflicts: List[Any] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    solved: bool = False
    best_round_repair_cost: int = 0
    best_round_subset_size: int = 0
    best_v: Optional[float] = None
    best_round: Optional[int] = None
    stop_reason: str = ""
    rounds: List[RoundLog] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_conflict_free(self) -> bool:
        """Whether the returned model explains its own learner input without conflict."""
        return self.solved and not self.conflicts

    def as_report(self) -> Dict[str, Any]:
        """Flat dict for ``fold_result.json``, shaped like ``SingleRoundResult``.

        The first five keys are deliberately the single round's, spelled the
        same way: ``run_fold._milp_specific`` reads one vocabulary, so the two
        arms land in one comparable schema instead of two near-identical ones.

        ``repair_cost``/``best_cost`` are deliberately ``None`` here. They exist
        to carry the design §7.1 check ``cost(MILP) <= cost(best CDPS CFM)``,
        which needs one solve that certified *every* trace jointly. The loop
        never performs one: each round solves a subset, so the winning round's
        cost is over ``best_round_subset_size`` traces, not the fold. Reporting
        it under the same key would make the two arms look comparable when they
        are not (a subset cost is smaller for free). The number itself is still
        useful, so it is kept — under a name that states its scope.
        """
        report = dict(self.stats)
        report.update({
            "algorithm": "pisam_milp_loop",
            "milp_solved": self.solved,
            "repair_cost": None,
            "best_cost": None,
            "best_round_repair_cost": self.best_round_repair_cost,
            "best_round_subset_size": self.best_round_subset_size,
            "conflict_free_model_count": 1 if self.is_conflict_free else 0,
            "pisam_conflicts_on_feasible": len(self.conflicts) if self.solved else None,
            "n_rounds": len(self.rounds),
            "n_rounds_solved": sum(1 for r in self.rounds if r.solved),
            "n_rounds_improved": sum(1 for r in self.rounds if r.improved),
            "n_rounds_tied": sum(1 for r in self.rounds if r.tied_with_best),
            "n_mixed_set_conflicts": sum(1 for r in self.rounds if r.mixed_set_conflict),
            "best_v": self.best_v,
            "best_round": self.best_round,
            "stop_reason": self.stop_reason,
        })
        return report


# ---------------------------------------------------------------- small helpers


def _canonical_action(action) -> str:
    """One action rendered in a fixed order, independent of set iteration order."""
    parameters = " ".join(
        f"{name}:{kind}" for name, kind in sorted(action.signature.items())
    )
    preconditions = sorted(
        f"{operator} {getattr(operand, 'untyped_representation', operand)}"
        for operator, operand in action.preconditions
    )
    effects = sorted(
        getattr(effect, "untyped_representation", str(effect))
        for effect in action.discrete_effects
    )
    return f"{action.name}({parameters})|{'&'.join(preconditions)}|{'&'.join(effects)}"


def model_hash(domain: Optional[LearnerDomain]) -> str:
    """Content hash of a learned model — the loop's identity for it.

    Deliberately **not** a hash of ``to_pddl()``. That text is not a stable
    function of the model: ``to_pddl`` builds an augmented ``:requirements``
    *set* on each call, so the same object can render two different strings and
    two structurally identical models can render differently. Dedup and the
    fixpoint stop rule both key on this hash, so an unstable identity would
    silently re-run solved rounds and never reach a fixpoint.

    Hashing the model structure in a canonical order instead means the hash
    changes exactly when the model does.
    """
    if domain is None:
        return NO_MODEL_HASH
    canonical = "\n".join(
        _canonical_action(action) for _name, action in sorted(domain.actions.items())
    )
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def _subset_gt(
    gt_states_by_obs: Optional[Dict[int, Set[int]]], indices: Sequence[int]
) -> Optional[Dict[int, Set[int]]]:
    """Re-key the GT map onto a subset's local 0..m-1 positions."""
    if not gt_states_by_obs:
        return None
    remapped = {
        local: gt_states_by_obs[original]
        for local, original in enumerate(indices)
        if original in gt_states_by_obs
    }
    return remapped or None


def _evaluate(
    domain: Optional[LearnerDomain],
    observations: Sequence[Observation],
    weights: EvaluationWeights,
) -> Optional[EvaluationResult]:
    """V of a model against the frozen originals; ``None`` when there is no model."""
    if domain is None:
        return None
    return observations_reconstruction_score(domain, observations, weights=weights)


# ---------------------------------------------------------------- trace cache


class _TraceCache:
    """Converts observations to ``planning_structs`` traces, once each.

    A fold's traces are converted once and reused by every round; only a
    ``pool_policy: replace`` write-back invalidates an entry. Without this the
    loop would pay the conversion for the same trace on every round it is
    sampled, which on a long run dominates the solve itself.
    """

    def __init__(
        self,
        ps_domain,
        partial_domain: Domain,
        gt_anchoring,
        object_types_by_obs: Optional[Mapping[int, Mapping[str, str]]] = None,
    ):
        self._ps_domain = ps_domain
        self._partial_domain = partial_domain
        self._gt_anchoring = gt_anchoring
        self._object_types_by_obs = object_types_by_obs or {}
        self._instances: Dict[int, Any] = {}
        self._traces: Dict[Tuple[int, str], Any] = {}

    def instance(self, index: int, observation: Observation):
        """The grounded instance for an observation (folds mix problems)."""
        if index not in self._instances:
            self._instances[index] = build_ps_instance_from_objects(
                self._ps_domain,
                self._partial_domain,
                observation.grounded_objects,
                include_repeated_args=True,
                object_types=self._object_types_by_obs.get(index),
            )
        return self._instances[index]

    def trace(
        self,
        index: int,
        observation: Observation,
        gt_state_indices: Optional[Set[int]],
        kind: str = "pool",
    ):
        """The trace for an observation.

        ``kind`` names the cache namespace and, with it, the failure policy —
        the two are the same distinction, so they are not separate arguments:

        - ``pool``: the encoded, objective-bearing trace of a *frozen* input
          observation. Unencodable means the fold's input is malformed, so it
          raises; silently shrinking the pool would change ``subset_size``,
          ``n_possible_subsets`` and the fixpoint rule without saying so.
        - ``hint``: a warm-start-only trace built from a *repair* the loop
          produced. ``None`` here costs nothing but a warm start, so it is
          tolerated and the caller drops the whole hint set.
        """
        key = (index, kind)
        if key not in self._traces:
            convert = (
                try_observation_to_trace if kind == "hint" else observation_to_trace
            )
            self._traces[key] = convert(
                self.instance(index, observation),
                observation,
                goal_fluents=None,
                gt_state_indices=gt_state_indices,
                gt_anchoring=self._gt_anchoring,
            )
        return self._traces[key]

    def invalidate(self, index: int, kind: str = "pool") -> None:
        """Drop a cached trace after its observation was replaced."""
        self._traces.pop((index, kind), None)


# ---------------------------------------------------------------- samplers


def _sample_random(
    pool: Sequence[int], size: int, rng: random.Random, blocked: Set[int]
) -> List[int]:
    """``size`` traces uniformly without replacement, avoiding ``blocked`` if possible."""
    eligible = [i for i in pool if i not in blocked]
    if len(eligible) < size:
        # The cooldown/co-sample guard is a preference, never a hard constraint:
        # with few traces it can starve the sampler, and a smaller subset would
        # silently change the experiment.
        eligible = list(pool)
    return sorted(rng.sample(eligible, size))


def _sample_hardest_first(
    pool: Sequence[int],
    size: int,
    per_trace_v: Dict[int, float],
    rng: random.Random,
    blocked: Set[int],
    forced: Sequence[int] = (),
) -> List[int]:
    """The worst-scoring traces under M_best, minus last round's picks.

    Ties are broken by trace index, not randomly, so a round is reproducible
    from its log. ``forced`` (the co-sampling heuristic) takes the first slots.
    """
    chosen = [i for i in dict.fromkeys(forced) if i in set(pool)][:size]
    if len(chosen) == size:
        return sorted(chosen)

    remaining = [i for i in pool if i not in chosen]
    eligible = [i for i in remaining if i not in blocked]
    if len(eligible) < size - len(chosen):
        eligible = remaining
    ranked = sorted(eligible, key=lambda i: (-per_trace_v.get(i, 0.0), i))
    chosen.extend(ranked[: size - len(chosen)])
    if len(chosen) < size:  # not enough ranked traces; top up at random
        spare = [i for i in pool if i not in chosen]
        chosen.extend(rng.sample(spare, min(size - len(chosen), len(spare))))
    return sorted(chosen)


def _per_trace_scores(
    evaluation: Optional[EvaluationResult], pool: Sequence[int]
) -> Dict[int, float]:
    """Map pool index -> that trace's length-normalised V under the incumbent.

    ``v_per_transition`` rather than ``v_raw``: ranking on the raw score would
    make ``hardest_first`` a proxy for "longest trace", which is not what the
    sampler is for. The pool filter is now a no-op -- ``run_loop`` requires every
    observation to be encodable, so the pool is all of them -- but it is kept
    because ``pool`` is the sampler's domain and this function's contract is to
    return something the sampler can index.
    """
    if evaluation is None:
        return {}
    in_pool = set(pool)
    return {
        trace.observation_index: trace.v_per_transition
        for trace in evaluation.per_trace
        if trace.observation_index in in_pool
    }


# ---------------------------------------------------------------- stop rules


#: Coefficient of the patience schedule. Fitted against the p99 improvement gap
#: measured on the small-data folds (N=5..8), where the required value was
#: 5.5-6.8 and drifting up with N; 7 sits above every observed requirement.
#: The fit rests on four points at N<=8 -- see
#: docs/large-corpora-experiments-plan.md section 3.
PATIENCE_COEFFICIENT = 7


def resolve_no_improvement_rounds(
    setting, n_traces: int, subset_size: int
) -> Optional[int]:
    """Patience in rounds: ``ceil(7 * ln C(n_traces, subset_size))`` for ``"auto"``.

    The subset space grows with the pool, so a patience that is right at 8
    traces is far too short at 2000. ``"auto"`` scales it with the space instead
    of pinning a constant per L, which is what lets one config serve a sweep.

    Args:
        setting: ``None`` (off), an int (used as-is), or ``"auto"``.
        n_traces: Size of the sampling pool.
        subset_size: Traces per subset.

    Returns:
        Rounds without improvement before stopping, or ``None`` when off.

    Raises:
        ValueError: on a string other than ``"auto"``.
    """
    if setting is None or isinstance(setting, int):
        return setting
    if str(setting).strip().lower() != "auto":
        raise ValueError(
            f"no_improvement_rounds must be an int, None or 'auto'; got {setting!r}"
        )
    total = math.comb(max(n_traces, subset_size), subset_size)
    return max(1, math.ceil(PATIENCE_COEFFICIENT * math.log(max(total, 2))))


def _stop_reason(
    stop_rules,
    rounds: Sequence[RoundLog],
    best_v: Optional[float],
    elapsed: float,
    rounds_without_improvement: int,
    exhausted: bool,
) -> Optional[str]:
    """The name of the first satisfied stop rule, or ``None`` to keep going."""
    budget = stop_rules.budget_seconds
    if budget is not None and elapsed >= budget:
        return "budget_seconds"
    if stop_rules.max_rounds is not None and len(rounds) >= stop_rules.max_rounds:
        return "max_rounds"
    if stop_rules.stop_on_perfect_fit and best_v is not None and best_v <= 0:
        return "perfect_fit"
    if (
        stop_rules.no_improvement_rounds is not None
        and rounds_without_improvement >= stop_rules.no_improvement_rounds
    ):
        return "no_improvement"
    if exhausted and stop_rules.stop_on_fixpoint:
        return "fixpoint"
    return None


def _remaining_budget(
    stop_budget: Optional[int], solve_limit: Optional[int], elapsed: float
) -> Optional[int]:
    """Per-solve cap: never let one solve overrun the whole loop's budget."""
    if stop_budget is None:
        return solve_limit
    left = max(1, int(stop_budget - elapsed))
    return left if solve_limit is None else min(solve_limit, left)


# ---------------------------------------------------------------- one round


def _solve_subset(
    ps_domain,
    traces_for_subset: List[Any],
    hint_traces: Optional[List[Any]],
    prior_observation_m,
    config: PisamMilpConfig,
    time_limit: Optional[int],
):
    """Encode and solve one subset. Returns ``(encoder, solved)``."""
    has_prior = prior_observation_m is not None
    encoder = resolve_encoder(config.solver_encoder_key())(
        ps_domain,
        Traces(instance=None, obs_m=prior_observation_m, obs_t=traces_for_subset),
        _OBJECTIVES_WITH_PRIOR if has_prior else _OBJECTIVES_NO_PRIOR,
        config=config.encoding_config(has_prior=has_prior),
    )
    encoder.state_hint_traces = hint_traces
    return encoder, encoder.solve(time_limit=time_limit)


def _learner_input(
    config: PisamMilpConfig,
    subset: Sequence[int],
    subset_repairs: Sequence[Observation],
    repaired: Dict[int, Observation],
) -> Tuple[List[Observation], List[int]]:
    """PI-SAM's input for this round, and the pool indices it corresponds to.

    ``subset_only`` is conflict-impossible (MILP feasibility means a single
    witness model explains the whole subset). ``accumulated`` is strictly more
    data but was never certified by one solve, so it may genuinely conflict —
    the caller falls back and records it.
    """
    if config.learner_input is LearnerInput.SUBSET_ONLY:
        return list(subset_repairs), list(subset)
    indices = sorted(repaired)
    return [repaired[i] for i in indices], indices


# ---------------------------------------------------------------- driver


def run_loop(
    partial_domain: Domain,
    observations: Sequence[Observation],
    config: PisamMilpConfig,
    cdps_budget_seconds: Optional[int] = None,
    gt_states_by_obs: Optional[Dict[int, Set[int]]] = None,
    negative_preconditions_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.hard,
    seed: Optional[int] = None,
    fold_work_dir: Optional[Path] = None,
    save_observations_fn=None,
    object_types_by_obs: Optional[Mapping[int, Mapping[str, str]]] = None,
) -> LoopResult:
    """Run the sample-repair-learn-score loop and return the best model found.

    Args:
        partial_domain: Domain template (predicates + action signatures).
        observations: The masked, noisy observations. Treated as immutable: they
            are the frozen reference V is measured against.
        config: A validated ``pisam_milp`` block with ``variant: loop``.
        cdps_budget_seconds: The fold's denoiser budget — what CDPS would have
            been given (Q7a). It is the *fallback* for two independent caps:
            ``config.stop.budget_seconds`` for the whole loop, and
            ``config.time_limit_seconds`` for one solve. Set either to override
            it; neither override affects the other.
        gt_states_by_obs: obs index -> known-GT state indices.
        seed: Overrides ``config.seed`` for the sampler (the benchmark passes the
            fold's seed). The solver's seed is pinned separately.
        fold_work_dir: Where to write the round history and the per-round
            candidate models. ``None`` runs the loop with no disk output at all.
        object_types_by_obs: obs index -> (object name -> declared type name),
            from the problems the observations came from. ``.trajectory`` files
            declare no ``:objects``, so without this the types are *inferred*
            and objects in untyped predicate slots come back as ``object`` —
            which makes their actions ungroundable. See
            :func:`converter.problem_object_types`.

    Returns:
        A :class:`LoopResult`. ``solved=False`` means no round ever produced a
        model — every solve was infeasible or ran out of time.

    Raises:
        ValueError: if any observation cannot be encoded. Dropping it instead
            would change the pool, and with it the subset size and the
            fixpoint rule, with nothing in the results saying so.
    """
    start = time.perf_counter()
    gt_states_by_obs = config.resolve_gt_states(gt_states_by_obs)
    stop_rules = config.effective_stop_rules()
    # One argument, two budgets, each with its own override. Deriving them
    # separately is what keeps ``time_limit_seconds: 30`` (a per-solve cap) from
    # silently also capping the whole loop at 30 seconds.
    loop_budget = stop_rules.resolve_budget(cdps_budget_seconds)
    solve_limit = config.resolve_time_limit(cdps_budget_seconds)
    weights = EvaluationWeights(
        effect_mismatch=config.eval.effect_mismatch_weight,
        inapplicability=config.eval.inapplicability_weight,
    )
    rng = random.Random(config.seed if seed is None else seed)

    ps_domain = build_ps_domain(partial_domain)
    cache = _TraceCache(
        ps_domain, partial_domain, config.gt_anchoring, object_types_by_obs
    )

    # The pool is every observation: each must be encodable, and `trace` raises
    # if one is not (see _TraceCache.trace). Encoding them all up front means
    # that failure surfaces before any solve time is spent.
    pool: List[int] = list(range(len(observations)))
    for index, observation in enumerate(observations):
        try:
            cache.trace(index, observation, (gt_states_by_obs or {}).get(index))
        except ValueError as error:
            raise ValueError(
                f"Observation {index} cannot be encoded for the MILP loop: {error}"
            ) from error
    if not pool:
        raise ValueError("No usable traces for the MILP loop")

    subset_size = config.subset_size.resolve(len(pool))
    total_subsets = math.comb(len(pool), subset_size)
    stop_rules = replace(
        stop_rules,
        no_improvement_rounds=resolve_no_improvement_rounds(
            stop_rules.no_improvement_rounds, len(pool), subset_size
        ),
    )

    # A rerun into the same fold dir must not append onto the previous run's
    # rows: the stream is a record of THIS loop, not of every loop that ever
    # wrote here.
    if fold_work_dir is not None:
        stale_stream = Path(fold_work_dir) / ROUND_STREAM_NAME
        if stale_stream.exists():
            stale_stream.unlink()

    state = _LoopState(
        pool_observations={i: observations[i] for i in pool},
        repaired={},
        rng=rng,
        round_models_dir=(
            None if fold_work_dir is None
            else Path(fold_work_dir) / ROUND_MODELS_DIR
        ),
    )
    result = LoopResult(learned_domain=None, stats={
        "config": config.as_stats(),
        "n_traces": len(pool),
        # Kept for schema stability with rows written before the pool became
        # mandatory. Always 0 now: an unencodable observation aborts the arm.
        "n_traces_dropped": 0,
        "subset_size": subset_size,
        "n_possible_subsets": total_subsets,
        "loop_budget_seconds": loop_budget,
        "no_improvement_rounds": stop_rules.no_improvement_rounds,
        "solve_time_limit_seconds": solve_limit,
    })

    while True:
        elapsed = time.perf_counter() - start
        exhausted = state.subsets_seen_for(model_hash(result.learned_domain)) >= total_subsets
        reason = _stop_reason(
            stop_rules, result.rounds, result.best_v, elapsed,
            state.rounds_without_improvement, exhausted,
        )
        if reason:
            result.stop_reason = reason
            break

        subset = state.draw(
            config, pool, subset_size, model_hash(result.learned_domain), total_subsets
        )
        if subset is None:
            result.stop_reason = "sampler_exhausted"
            break

        _run_round(
            round_index=len(result.rounds) + 1,
            subset=subset,
            state=state,
            result=result,
            config=config,
            cache=cache,
            ps_domain=ps_domain,
            partial_domain=partial_domain,
            observations=observations,
            gt_states_by_obs=gt_states_by_obs,
            negative_preconditions_policy=negative_preconditions_policy,
            weights=weights,
            pool=pool,
            solve_limit=_remaining_budget(loop_budget, solve_limit, elapsed),
            started_at=start,
        )
        # Every exit path of _run_round has appended its row by now, including
        # the early ones, so the stream sees the round whatever happened in it.
        if result.rounds:
            append_round_stream(result.rounds[-1], fold_work_dir)

    result.stats["total_time_seconds"] = round(time.perf_counter() - start, 3)
    result.stats["rounds"] = [r.as_dict() for r in result.rounds]
    if fold_work_dir is not None:
        save_round_log(result, Path(fold_work_dir))
    return result


@dataclass
class _LoopState:
    """The loop's mutable bookkeeping, kept out of :func:`run_loop`'s body."""

    pool_observations: Dict[int, Observation]
    repaired: Dict[int, Observation]
    rng: random.Random
    # Where per-round models go, or ``None`` to keep the loop purely in-memory
    # (the default for unit tests and for callers that pass no work dir).
    round_models_dir: Optional[Path] = None
    solved_pairs: Set[Tuple[Tuple[int, ...], str]] = field(default_factory=set)
    per_trace_v: Dict[int, float] = field(default_factory=dict)
    cooldown: Set[int] = field(default_factory=set)
    forced_next: List[int] = field(default_factory=list)
    rounds_without_improvement: int = 0

    def subsets_seen_for(self, prior_hash: str) -> int:
        """How many distinct subsets have been solved against this incumbent."""
        return sum(1 for _subset, h in self.solved_pairs if h == prior_hash)

    def draw(
        self,
        config: PisamMilpConfig,
        pool: Sequence[int],
        size: int,
        prior_hash: str,
        total_subsets: int,
    ) -> Optional[List[int]]:
        """A subset not yet solved against ``prior_hash``, or ``None`` if none exists.

        Dedup is only applied when the pool is frozen: with ``replace`` the same
        subset denotes different data each round, so skipping it would drop real
        work rather than duplicated work.

        After the configured sampler's first miss the redraws go uniform.
        ``hardest_first`` is a deterministic function of the incumbent, so while
        the incumbent stands it re-proposes the same subset forever; retrying it
        would spin. Falling back to uniform keeps exploration alive without
        pretending the round was the sampler's choice.
        """
        if config.dedup_rounds and self.subsets_seen_for(prior_hash) >= total_subsets:
            return None  # exact fixpoint: no unseen subset exists for this prior
        for attempt in range(_MAX_RESAMPLE_ATTEMPTS):
            subset = (
                self._draw_once(config, pool, size)
                if attempt == 0
                else _sample_random(pool, size, self.rng, blocked=set())
            )
            if not config.dedup_rounds:
                return subset
            if (tuple(subset), prior_hash) not in self.solved_pairs:
                return subset
        return None

    def _draw_once(
        self, config: PisamMilpConfig, pool: Sequence[int], size: int
    ) -> List[int]:
        blocked = set(self.cooldown)
        if config.sampler is Sampler.HARDEST_FIRST and self.per_trace_v:
            return _sample_hardest_first(
                pool, size, self.per_trace_v, self.rng, blocked, self.forced_next
            )
        return _sample_random(pool, size, self.rng, blocked)


def _run_round(
    *,
    round_index: int,
    subset: List[int],
    state: _LoopState,
    result: LoopResult,
    config: PisamMilpConfig,
    cache: _TraceCache,
    ps_domain,
    partial_domain: Domain,
    observations: Sequence[Observation],
    gt_states_by_obs: Optional[Dict[int, Set[int]]],
    negative_preconditions_policy: NegativePreconditionPolicy,
    weights: EvaluationWeights,
    pool: Sequence[int],
    solve_limit: Optional[int],
    started_at: float,
) -> None:
    """Solve, learn, score and (maybe) accept one round; appends its log row."""
    round_start = time.perf_counter()
    prior_hash = model_hash(result.learned_domain)
    log = RoundLog(
        round_index=round_index,
        subset=list(subset),
        prior_hash=prior_hash,
        elapsed_seconds=round_start - started_at,
        round_seconds=0.0,
    )
    result.rounds.append(log)
    state.solved_pairs.add((tuple(subset), prior_hash))
    state.cooldown = set(subset)  # ineligible next round (plan §2.2)
    state.forced_next = []

    prior = None
    if result.learned_domain is not None:
        prior = learner_domain_to_observation_m(
            result.learned_domain, ps_domain
        ).observation_m

    subset_observations = [state.pool_observations[i] for i in subset]
    traces = [
        cache.trace(i, state.pool_observations[i], (gt_states_by_obs or {}).get(i))
        for i in subset
    ]
    hints = _hint_traces(config, cache, state, subset, gt_states_by_obs)

    encoder, solved = _solve_subset(
        ps_domain, traces, hints, prior, config, solve_limit
    )
    log.solve_seconds = encoder.solve_stats.get("solve_time_seconds")
    if not solved:
        log.note = "solver returned nothing usable"
        log.round_seconds = time.perf_counter() - round_start
        state.rounds_without_improvement += 1
        return

    extraction = extract_repaired_observations(encoder, subset_observations)
    log.solved = True
    log.repair_cost = extraction.repair_cost
    for position, index in enumerate(subset):
        state.repaired[index] = extraction.observations[position]
        if config.pool_policy is PoolPolicy.REPLACE:
            state.pool_observations[index] = extraction.observations[position]
            cache.invalidate(index)

    _learn_and_score(
        log=log, state=state, result=result, config=config,
        subset=subset, subset_repairs=extraction.observations,
        partial_domain=partial_domain, observations=observations,
        gt_states_by_obs=gt_states_by_obs,
        negative_preconditions_policy=negative_preconditions_policy,
        weights=weights, pool=pool,
    )
    log.round_seconds = time.perf_counter() - round_start


def _hint_traces(
    config: PisamMilpConfig,
    cache: _TraceCache,
    state: _LoopState,
    subset: Sequence[int],
    gt_states_by_obs: Optional[Dict[int, Set[int]]],
) -> Optional[List[Any]]:
    """Warm-start traces built from each trace's previous repair, or ``None``.

    Only ``frozen_with_hints`` uses them, and only for traces that already have
    a repair; a trace seen for the first time falls back to being hinted by its
    own observation, which is what the encoder does anyway.
    """
    if config.pool_policy is not PoolPolicy.FROZEN_WITH_HINTS:
        return None
    if not any(i in state.repaired for i in subset):
        return None
    hints = []
    for i in subset:
        source = state.repaired.get(i, state.pool_observations[i])
        kind = "hint" if i in state.repaired else "pool"
        if kind == "hint":
            cache.invalidate(i, "hint")  # the repair changes every round
        hints.append(
            cache.trace(i, source, (gt_states_by_obs or {}).get(i), kind=kind)
        )
    return hints if all(h is not None for h in hints) else None


def _learn_and_score(
    *,
    log: RoundLog,
    state: _LoopState,
    result: LoopResult,
    config: PisamMilpConfig,
    subset: Sequence[int],
    subset_repairs: Sequence[Observation],
    partial_domain: Domain,
    observations: Sequence[Observation],
    gt_states_by_obs: Optional[Dict[int, Set[int]]],
    negative_preconditions_policy: NegativePreconditionPolicy,
    weights: EvaluationWeights,
    pool: Sequence[int],
) -> None:
    """PI-SAM on the round's input, V against the originals, then greedy update."""
    learner_observations, indices = _learner_input(
        config, subset, subset_repairs, state.repaired
    )
    domain, conflicts, _report = _learn_with_pisam(
        partial_domain, learner_observations,
        _subset_gt(gt_states_by_obs, indices),
        negative_preconditions_policy, config.seed,
    )

    if conflicts and config.learner_input is LearnerInput.ACCUMULATED:
        # Repairs certified in different rounds need not admit a common model.
        # Fall back to this round's subset, which one solve did certify (§2.3).
        log.mixed_set_conflict = True
        conflicting = [i for i in indices if i not in subset]
        if config.co_sample_conflicts:
            state.forced_next = (list(subset) + conflicting)[: len(subset)]
        domain, conflicts, _report = _learn_with_pisam(
            partial_domain, list(subset_repairs),
            _subset_gt(gt_states_by_obs, subset),
            negative_preconditions_policy, config.seed,
        )
        learner_observations = list(subset_repairs)

    log.pisam_conflicts = len(conflicts)
    log.learner_input_size = len(learner_observations)
    log.model_hash = model_hash(domain)
    if state.round_models_dir is not None:
        save_round_model(domain, state.round_models_dir, log.round_index)

    evaluation = _evaluate(domain, observations, weights)
    if evaluation is None:
        log.note = "PI-SAM returned no model"
        state.rounds_without_improvement += 1
        return

    log.v_raw = evaluation.v_raw
    log.v_per_transition = evaluation.v_per_transition
    log.effect_mismatches = evaluation.effect_mismatches
    log.inapplicability_events = evaluation.inapplicability_events
    log.success_rate = evaluation.success_rate

    # Greedy, strict (Q8a). A tie leaves the incumbent in place (Q8c: rule C);
    # ties are logged because no GT-free rule was found that beats an arbitrary
    # pick, and that finding should stay falsifiable from run data.
    if result.best_v is not None and abs(evaluation.v_raw - result.best_v) < 1e-9:
        log.tied_with_best = True
    if result.best_v is None or evaluation.v_raw < result.best_v:
        log.improved = True
        result.best_v = evaluation.v_raw
        result.best_round = log.round_index
        result.learned_domain = domain
        result.conflicts = list(conflicts)
        result.observations = list(learner_observations)
        result.best_round_repair_cost = log.repair_cost
        result.best_round_subset_size = len(subset)
        result.solved = True
        state.rounds_without_improvement = 0
        state.per_trace_v = _per_trace_scores(evaluation, pool)
    else:
        state.rounds_without_improvement += 1


# ---------------------------------------------------------------- artifacts


def save_round_model(
    domain: Optional[LearnerDomain], round_models_dir: Path, round_index: int
) -> Optional[Path]:
    """Write one round's candidate model; return its path, or ``None`` if unwritten.

    Every round that produced a model gets a file, including rounds that lost to
    the incumbent. The loop only ever keeps the winner, so without this the
    losers are unrecoverable — and an anytime curve is exactly a statement about
    what was on the table at each point in time, not only about what survived.

    Callers must not infer model identity from this text. ``to_pddl`` rebuilds
    its ``:requirements`` set per call, so two rounds sharing a ``model_hash``
    can render differently (see :func:`model_hash`). The hash in the round log
    is the identity; this file is the artifact.
    """
    if domain is None:
        return None
    target = Path(round_models_dir) / f"round_{round_index}"
    target.mkdir(parents=True, exist_ok=True)
    path = target / "model.pddl"
    path.write_text(domain.to_pddl())
    return path


def append_round_stream(log: "RoundLog", fold_work_dir: Optional[Path]) -> None:
    """Append one finished round to the JSONL stream.

    A best-effort tap: a failure here must not abort a round that already did
    its work, so OS errors are reported and swallowed.
    """
    if fold_work_dir is None:
        return
    path = Path(fold_work_dir) / ROUND_STREAM_NAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(log.as_dict()) + "\n")
    except OSError as error:
        print(f"  [pisam_milp_loop] could not append {path}: {error}")


def save_round_log(result: LoopResult, fold_work_dir: Path) -> None:
    """Write the round history next to the fold's other MILP diagnostics."""
    fold_work_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_v": result.best_v,
        "best_round": result.best_round,
        "stop_reason": result.stop_reason,
        "rounds": [r.as_dict() for r in result.rounds],
    }
    (fold_work_dir / "milp_loop_rounds.json").write_text(json.dumps(payload, indent=2))
