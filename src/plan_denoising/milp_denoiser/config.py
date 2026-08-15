"""Configuration surface of the ``pisam_milp_*`` algorithms.

Mirrors the ``pisam_milp:`` block of ``benchmark/run_config.yaml`` (documented in
``docs/pisam-milp-loop-plan.md`` §5). Every enum-valued key is validated here and
a bad value raises immediately, listing the allowed options — configuration
mistakes must not surface as a silent behavior change three hours into a run.

Both variants' keys are accepted. The loop-only keys (``sampler``,
``subset_size``, ``learner_input``, ``pool_policy``, ``co_sample_conflicts``,
``stop``, ``eval``) are parsed and validated whatever the variant, so a config
mistake is caught even in a ``single_round`` run that ignores them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import product
from typing import Any, List, Mapping, Optional, Type, TypeVar

from src.milp.converter import GtAnchoring
from src.milp.encoding_config import (
    MilpEncodingConfig,
    PriorWeightMode,
)

_E = TypeVar("_E", bound=Enum)


class MilpVariant(Enum):
    """Which ``pisam_milp_*`` algorithm to run."""

    SINGLE_ROUND = "single_round"
    LOOP = "loop"


class MilpSolver(Enum):
    """Backend for the constraint program. ``GUROBI`` is a stub (needs a license)."""

    CPSAT = "cpsat"
    GUROBI = "gurobi"


class ObsWeighting(Enum):
    """Per-fluent-slot objective weights. Dormant hook: image-mode VLM
    confidences may add a mode here later (plan decision 11)."""

    UNIFORM = "uniform"


class Sampler(Enum):
    """How each round picks its subset S_r (plan §2.2)."""

    RANDOM = "random"
    HARDEST_FIRST = "hardest_first"


class LearnerInput(Enum):
    """What PI-SAM sees in round r (plan §2.3)."""

    SUBSET_ONLY = "subset_only"
    ACCUMULATED = "accumulated"


class PoolPolicy(Enum):
    """What a round's repairs do to the pool the *next* round samples from.

    Decided 2026-08-11 (Q6). ``FROZEN`` is the default because feeding repairs
    back creates a ratchet: round r's repair becomes round r+1's evidence, so an
    early wrong repair can never be argued away by the data it overwrote, and V
    is then scored against a pool that has drifted from the observations.

    Attributes:
        FROZEN: The pool is always the ORIGINAL noisy observations. Repairs are
            used only to learn from.
        REPLACE: A repaired trace replaces its noisy sibling in the pool. Since
            the pool then changes under the loop, ``(subset, M_best)`` no longer
            identifies a round: dedup and the fixpoint stop rule are disabled
            automatically (they would be unsound, not merely useless).
        FROZEN_WITH_HINTS: Pool frozen, but the previous repair of a trace is
            handed to the solver as a warm start. Same search space, faster.
    """

    FROZEN = "frozen"
    REPLACE = "replace"
    FROZEN_WITH_HINTS = "frozen_with_hints"


class SubsetSizeKind(Enum):
    """The named subset-size policies (Q9b: named policies, not an expression)."""

    HALF = "half"
    ALL = "all"
    FIXED = "fixed"


@dataclass(frozen=True)
class SubsetSize:
    """Subset size ``m``, resolved once per fold against the trace count.

    The plan's §5 wrote an expression string; we take named policies instead
    (decided 2026-08-11, Q9b) so no expression evaluator has to be trusted.
    """

    kind: SubsetSizeKind = SubsetSizeKind.HALF
    value: Optional[int] = None

    @classmethod
    def parse(cls, raw: Any) -> "SubsetSize":
        """From ``half``/``all``/a positive int (or its string form)."""
        if isinstance(raw, SubsetSize):
            return raw
        if isinstance(raw, bool):  # bool is an int subclass; never a size
            raise ValueError(f"pisam_milp.subset_size: invalid value {raw!r}")
        if isinstance(raw, int):
            return cls._fixed(raw)
        text = str(raw).strip().lower()
        if text == "half":
            return cls(SubsetSizeKind.HALF)
        if text == "all":
            return cls(SubsetSizeKind.ALL)
        try:
            return cls._fixed(int(text))
        except ValueError:
            raise ValueError(
                f"pisam_milp.subset_size: invalid value {raw!r}. "
                f"Allowed: half, all, or a positive integer"
            ) from None

    @classmethod
    def _fixed(cls, number: int) -> "SubsetSize":
        if number < 1:
            raise ValueError(
                f"pisam_milp.subset_size: {number} is not positive"
            )
        return cls(SubsetSizeKind.FIXED, number)

    def resolve(self, n_traces: int) -> int:
        """The concrete ``m`` for a fold with ``n_traces`` traces.

        Always clamped to ``[1, n_traces]``: a subset larger than the pool is a
        config mistake the loop can absorb, and asking for 0 traces is never
        meaningful. ``half`` is ``max(2, ceil(n/2))`` (Q9a) — the 2 floor keeps
        a round able to expose a *disagreement between* traces, which is the
        only thing a joint solve can exploit that a per-trace solve cannot.
        """
        if n_traces <= 0:
            return 0
        if self.kind is SubsetSizeKind.ALL:
            return n_traces
        if self.kind is SubsetSizeKind.HALF:
            wanted = max(2, math.ceil(n_traces / 2))
        else:
            wanted = int(self.value or 1)
        return max(1, min(wanted, n_traces))

    def as_stat(self) -> str:
        """The value as written in config, for reporting."""
        return str(self.value) if self.kind is SubsetSizeKind.FIXED else self.kind.value


@dataclass(frozen=True)
class StopRules:
    """Loop termination, any-of semantics: the first satisfied rule stops it.

    Attributes:
        budget_seconds: Wall-clock cap for the whole loop. ``None`` = inherit
            the fold's CDPS search budget, which is what makes the anytime
            comparison against CDPS fair (Q7a).
        no_improvement_rounds: Stop after this many consecutive rounds without
            V improving. ``None`` = off.
        stop_on_perfect_fit: Stop when V(M_best) == 0.
        max_rounds: Hard cap on rounds. ``None`` = off.
        stop_on_fixpoint: Stop when every admissible ``(subset, M_best)`` pair
            has already been solved, so no further round can produce anything
            new. Only sound with a frozen pool and a deterministic solver, and
            :meth:`PisamMilpConfig.effective_stop_rules` switches it off when the
            pool is not frozen.
    """

    budget_seconds: Optional[int] = None
    no_improvement_rounds: Optional[int] = 5
    stop_on_perfect_fit: bool = True
    max_rounds: Optional[int] = None
    stop_on_fixpoint: bool = True

    _KEYS = frozenset({
        "budget_seconds", "no_improvement_rounds", "stop_on_perfect_fit",
        "max_rounds", "stop_on_fixpoint",
    })

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "StopRules":
        """Build (and validate) from the nested ``stop:`` mapping."""
        if not raw:
            return cls()
        unknown = sorted(set(raw) - cls._KEYS)
        if unknown:
            raise ValueError(
                f"pisam_milp.stop: unknown key(s) {unknown}. Allowed: "
                f"{', '.join(sorted(cls._KEYS))}"
            )
        return cls(
            budget_seconds=_parse_optional_int(raw.get("budget_seconds"), "stop.budget_seconds"),
            no_improvement_rounds=_parse_optional_int(
                raw.get("no_improvement_rounds", 5), "stop.no_improvement_rounds"
            ),
            stop_on_perfect_fit=_parse_bool(
                raw.get("stop_on_perfect_fit", True), "stop.stop_on_perfect_fit"
            ),
            max_rounds=_parse_optional_int(raw.get("max_rounds"), "stop.max_rounds"),
            stop_on_fixpoint=_parse_bool(
                raw.get("stop_on_fixpoint", True), "stop.stop_on_fixpoint"
            ),
        )

    def resolve_budget(self, cdps_budget_seconds: Optional[int]) -> Optional[int]:
        """Loop budget: the explicit setting, else the fold's CDPS budget (Q7a)."""
        return self.budget_seconds if self.budget_seconds is not None else cdps_budget_seconds

    def as_stats(self) -> dict:
        """Flat dict for ``fold_result.json`` reporting."""
        return {
            "stop_budget_seconds": self.budget_seconds,
            "stop_no_improvement_rounds": self.no_improvement_rounds,
            "stop_on_perfect_fit": self.stop_on_perfect_fit,
            "stop_max_rounds": self.max_rounds,
            "stop_on_fixpoint": self.stop_on_fixpoint,
        }


@dataclass(frozen=True)
class EvalWeights:
    """Weights of V's two components. Both 1.0 until evidence says otherwise."""

    effect_mismatch_weight: float = 1.0
    inapplicability_weight: float = 1.0

    _KEYS = frozenset({"effect_mismatch_weight", "inapplicability_weight"})

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "EvalWeights":
        """Build (and validate) from the nested ``eval:`` mapping."""
        if not raw:
            return cls()
        unknown = sorted(set(raw) - cls._KEYS)
        if unknown:
            raise ValueError(
                f"pisam_milp.eval: unknown key(s) {unknown}. Allowed: "
                f"{', '.join(sorted(cls._KEYS))}"
            )
        return cls(
            effect_mismatch_weight=float(raw.get("effect_mismatch_weight", 1.0)),
            inapplicability_weight=float(raw.get("inapplicability_weight", 1.0)),
        )

    def as_stats(self) -> dict:
        """Flat dict for ``fold_result.json`` reporting."""
        return {
            "eval_effect_mismatch_weight": self.effect_mismatch_weight,
            "eval_inapplicability_weight": self.inapplicability_weight,
        }


def _parse_optional_int(value: Any, key: str) -> Optional[int]:
    """``None``/``null`` stays None; anything else must be a non-negative int."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"null", "none", ""}:
        return None
    try:
        number = int(text)
    except ValueError:
        raise ValueError(
            f"pisam_milp.{key}: invalid value {value!r}. Allowed: a non-negative integer or null"
        ) from None
    if number < 0:
        raise ValueError(f"pisam_milp.{key}: {number} is negative")
    return number


def _parse_enum(enum_cls: Type[_E], value: Any, key: str) -> _E:
    """Enum member from a YAML scalar, with an error naming the allowed options."""
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value).strip().lower())
    except ValueError:
        allowed = ", ".join(m.value for m in enum_cls)
        raise ValueError(
            f"pisam_milp.{key}: invalid value {value!r}. Allowed: {allowed}"
        ) from None


def _parse_bool(value: Any, key: str) -> bool:
    """YAML-ish boolean. ``on``/``off`` are accepted because §5 writes ``eq16: on``."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"on", "true", "yes", "1"}:
        return True
    if text in {"off", "false", "no", "0"}:
        return False
    raise ValueError(f"pisam_milp.{key}: invalid value {value!r}. Allowed: on, off")


@dataclass(frozen=True)
class PisamMilpConfig:
    """Validated ``pisam_milp`` block.

    Attributes:
        variant: Which algorithm to run.
        eq16: ICAPS-26 eq. 16 precondition bias. NOTE: changes the repaired
            trajectories T', not only the witness model, and invalidates the
            ``cost(MILP) <= cost(best CDPS CFM)`` lower-bound check.
        lambda_pre: Coefficient of that bias; used only when ``eq16``.
        w_prior: Weighting of the reference-model (``M_best``) objective channel.
            Ignored by ``single_round``, which has no reference model.
        solver: Constraint-program backend.
        obs_weights: Per-fluent-slot weighting scheme.
        gt_anchoring: Which known-GT states become hard constraints.
        time_limit_seconds: Solver budget for ONE solve. ``None`` = inherit the
            fold's CDPS search budget, which is what makes the head-to-head
            comparison fair. Distinct from ``stop.budget_seconds``, which caps
            the loop as a whole.
        sampler: How each loop round picks its subset (loop only).
        subset_size: Subset-size policy (loop only).
        learner_input: What PI-SAM sees each round (loop only).
        pool_policy: What repairs do to the sampling pool (loop only).
        co_sample_conflicts: After an ``accumulated`` mixed-set conflict,
            co-sample the conflicting traces next round (loop only).
        stop: Loop termination rules (loop only).
        eval: Weights of V's components (loop only).
        seed: Drives the loop's sampler. The solver's own seed is pinned
            separately in ``encoder.DEFAULT_RANDOM_SEED``.
    """

    variant: MilpVariant = MilpVariant.SINGLE_ROUND
    eq16: bool = False
    lambda_pre: float = 0.4
    w_prior: PriorWeightMode = PriorWeightMode.TIEBREAK
    solver: MilpSolver = MilpSolver.CPSAT
    obs_weights: ObsWeighting = ObsWeighting.UNIFORM
    gt_anchoring: GtAnchoring = GtAnchoring.INIT_ONLY
    time_limit_seconds: Optional[int] = None
    sampler: Sampler = Sampler.RANDOM
    subset_size: SubsetSize = field(default_factory=SubsetSize)
    learner_input: LearnerInput = LearnerInput.SUBSET_ONLY
    pool_policy: PoolPolicy = PoolPolicy.FROZEN
    co_sample_conflicts: bool = False
    stop: StopRules = field(default_factory=StopRules)
    eval: EvalWeights = field(default_factory=EvalWeights)
    seed: int = 42

    _KEYS = frozenset({
        "variant", "eq16", "lambda_pre", "w_prior", "solver", "obs_weights",
        "gt_anchoring", "time_limit_seconds",
        "sampler", "subset_size", "learner_input", "pool_policy",
        "co_sample_conflicts", "stop", "eval", "seed",
    })

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "PisamMilpConfig":
        """Build (and validate) from the raw YAML mapping; ``None`` = all defaults."""
        if not raw:
            return cls()

        unknown = sorted(set(raw) - cls._KEYS)
        if unknown:
            raise ValueError(
                f"pisam_milp: unknown key(s) {unknown}. Allowed: "
                f"{', '.join(sorted(cls._KEYS))}"
            )

        limit = raw.get("time_limit_seconds")
        return cls(
            variant=_parse_enum(MilpVariant, raw.get("variant", "single_round"), "variant"),
            eq16=_parse_bool(raw.get("eq16", False), "eq16"),
            lambda_pre=float(raw.get("lambda_pre", 0.4)),
            w_prior=_parse_enum(PriorWeightMode, raw.get("w_prior", "tiebreak"), "w_prior"),
            solver=_parse_enum(MilpSolver, raw.get("solver", "cpsat"), "solver"),
            obs_weights=_parse_enum(
                ObsWeighting, raw.get("obs_weights", "uniform"), "obs_weights"
            ),
            gt_anchoring=_parse_enum(
                GtAnchoring, raw.get("gt_anchoring", "init_only"), "gt_anchoring"
            ),
            time_limit_seconds=None if limit is None else int(limit),
            sampler=_parse_enum(Sampler, raw.get("sampler", "random"), "sampler"),
            subset_size=SubsetSize.parse(raw.get("subset_size", "half")),
            learner_input=_parse_enum(
                LearnerInput, raw.get("learner_input", "subset_only"), "learner_input"
            ),
            pool_policy=_parse_enum(
                PoolPolicy, raw.get("pool_policy", "frozen"), "pool_policy"
            ),
            co_sample_conflicts=_parse_bool(
                raw.get("co_sample_conflicts", False), "co_sample_conflicts"
            ),
            stop=StopRules.from_dict(raw.get("stop")),
            eval=EvalWeights.from_dict(raw.get("eval")),
            seed=int(raw.get("seed", 42)),
        )

    @property
    def pool_is_frozen(self) -> bool:
        """Whether the sampling pool stays the ORIGINAL observations.

        ``frozen_with_hints`` only warm-starts the solver, so its pool is frozen
        too; the round identity ``(subset, M_best)`` therefore still holds.
        """
        return self.pool_policy is not PoolPolicy.REPLACE

    @property
    def dedup_rounds(self) -> bool:
        """Whether a ``(subset, M_best)`` pair may be skipped as already solved.

        Sound only when the pool is frozen (the same subset means the same input
        traces) AND the solver is deterministic (the same input gives the same
        answer — pinned in ``encoder.solve``, and measured: without pinning one
        blocksworld fold returned two different models for one input).
        """
        return self.pool_is_frozen

    def effective_stop_rules(self) -> StopRules:
        """``stop`` with the rules a non-frozen pool invalidates switched off.

        Silently keeping ``stop_on_fixpoint`` under ``pool_policy: replace``
        would stop the loop on a condition that no longer means "nothing new can
        happen", so it is disabled rather than trusted.
        """
        if self.pool_is_frozen:
            return self.stop
        return replace(self.stop, stop_on_fixpoint=False)

    def encoding_config(self, has_prior: bool = False) -> MilpEncodingConfig:
        """The encoder rule set implied by this config.

        Args:
            has_prior: Whether a reference model is actually available. False for
                ``single_round`` and for round 1 of the loop, where the model
                objective channel must be absent rather than merely zero-weighted.
        """
        return MilpEncodingConfig.cdps_dialect(
            eq16=self.eq16,
            lambda_pre=self.lambda_pre,
            prior_weighting=self.w_prior if has_prior else PriorWeightMode.NONE,
        )

    def solver_encoder_key(self) -> str:
        """The ``constraint_opt`` factory key for this solver backend.

        Gurobi is a declared-but-unbuilt option (no license, see plan §6.8): the
        seam exists so a backend can be added without touching the encoding, but
        asking for it today must fail loudly rather than silently run CP-SAT.
        """
        if self.solver is MilpSolver.CPSAT:
            return "cp-sat-observed"
        raise NotImplementedError(
            f"pisam_milp.solver={self.solver.value!r} has no backend yet. "
            f"Only {MilpSolver.CPSAT.value!r} is implemented."
        )

    def resolve_time_limit(self, cdps_budget_seconds: Optional[int]) -> Optional[int]:
        """Solver budget: the explicit setting, else the fold's CDPS budget."""
        return self.time_limit_seconds if self.time_limit_seconds is not None \
            else cdps_budget_seconds

    def as_stats(self) -> dict:
        """Flat dict for ``fold_result.json`` reporting."""
        return {
            "variant": self.variant.value,
            "eq16": self.eq16,
            "lambda_pre": self.lambda_pre if self.eq16 else 0.0,
            "w_prior": self.w_prior.value,
            "solver": self.solver.value,
            "obs_weights": self.obs_weights.value,
            "gt_anchoring": self.gt_anchoring.value,
            "time_limit_seconds": self.time_limit_seconds,
            "sampler": self.sampler.value,
            "subset_size": self.subset_size.as_stat(),
            "learner_input": self.learner_input.value,
            "pool_policy": self.pool_policy.value,
            "co_sample_conflicts": self.co_sample_conflicts,
            "seed": self.seed,
            **self.effective_stop_rules().as_stats(),
            **self.eval.as_stats(),
        }

    def arm_identity(self) -> tuple:
        """Hashable identity over the settings that change *this* variant's output.

        Two configs with equal identities produce the same model from the same
        input, so running both is pure duplicated compute. Equality of the
        dataclass itself is too strict for that question in two places:

        - the loop-only keys are inert under ``single_round``, so ablating (say)
          ``pool_policy`` in a run that also selects ``single_round`` must not
          make that arm solve the same problem twice;
        - ``lambda_pre`` is read only when ``eq16`` is on, so it is normalised to
          0.0 when off — the same rule ``as_stats`` already applies.
        """
        shared = (
            self.variant, self.eq16, self.lambda_pre if self.eq16 else 0.0,
            self.solver, self.obs_weights, self.gt_anchoring,
            self.time_limit_seconds,
        )
        if self.variant is not MilpVariant.LOOP:
            return shared
        return shared + (
            self.w_prior, self.sampler, self.subset_size, self.learner_input,
            self.pool_policy, self.co_sample_conflicts, self.stop, self.eval,
            self.seed,
        )


# ---------------------------------------------------------------------------
# The YAML key this block lives under
# ---------------------------------------------------------------------------

CONFIG_KEY = "pisam_milp"

# Pre-rename spelling, still used by configs archived under finished_run_configs/.
LEGACY_CONFIG_KEY = "cdps_milp"

CONFIG_KEYS = frozenset({CONFIG_KEY, LEGACY_CONFIG_KEY})


def select_milp_block(mapping: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """The MILP block out of a ``shared:``-shaped mapping, new key or legacy.

    Args:
        mapping: A mapping that may carry the block under either key.

    Returns:
        The block, or ``None`` if neither key is present.

    Raises:
        ValueError: If both keys are present.
    """
    present = [key for key in (CONFIG_KEY, LEGACY_CONFIG_KEY) if key in mapping]
    if len(present) == 2:
        raise ValueError(
            f"Both `{CONFIG_KEY}:` and the legacy `{LEGACY_CONFIG_KEY}:` are "
            f"present; keep `{CONFIG_KEY}:` and delete `{LEGACY_CONFIG_KEY}:`."
        )
    return mapping.get(present[0]) if present else None


# ---------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------

# Nested blocks are not ablatable: neither enters the results label (see
# benchmark.algorithms.pisam_milp_algorithm_name), so listing values here would
# produce several arms that all land in one results row.
_UNABLATABLE_KEYS = frozenset({"stop", "eval", "variant"})


def expand_pisam_milp_ablations(
    raw: Optional[Mapping[str, Any]]
) -> List[PisamMilpConfig]:
    """The configs a ``pisam_milp:`` block asks for — one per ablation combination.

    Without an ``ablations:`` sub-block this returns exactly one config, which is
    the block parsed as it always was. With one, every listed knob is crossed
    with every other, in the manner of ``simulation.grid``::

        pisam_milp:
          eq16: off
          pool_policy: frozen
          ablations:
            eq16: [off, on]
            pool_policy: [frozen, replace]   # -> 4 configs

    A knob named under ``ablations:`` **overrides** its value in the surrounding
    block: the list is the complete set of values for that knob, so the ``eq16:
    off`` above contributes nothing once ``eq16`` is listed. The alternative —
    treating the list as an addition to the block's own value — would silently
    run a configuration nobody named. The surrounding block keeps its job of
    supplying every knob the ablation block does *not* mention.

    Each combination goes through :meth:`PisamMilpConfig.from_dict` unchanged, so
    an invalid ablation value fails exactly as an invalid plain value does, and
    before any solving starts.

    Args:
        raw: The ``pisam_milp:`` mapping, ``ablations:`` included. ``None`` or
            empty gives a single all-defaults config.

    Returns:
        One config per combination, in cross-product order with the first listed
        knob varying slowest. Never empty.

    Raises:
        ValueError: If an ablation names an unknown or unablatable key, or gives
            it something other than a non-empty list.
    """
    if not raw:
        return [PisamMilpConfig()]

    base = {k: v for k, v in raw.items() if k != "ablations"}
    ablations = raw.get("ablations")
    if not ablations:
        return [PisamMilpConfig.from_dict(base)]

    _validate_ablations(ablations)
    keys = list(ablations)
    return [
        PisamMilpConfig.from_dict({**base, **dict(zip(keys, combination))})
        for combination in product(*(ablations[k] for k in keys))
    ]


def _validate_ablations(ablations: Any) -> None:
    """Reject an ``ablations:`` block that cannot mean what it appears to."""
    if not isinstance(ablations, Mapping):
        raise ValueError(
            f"pisam_milp.ablations: expected a mapping of knob -> list of values, "
            f"got {type(ablations).__name__}"
        )
    unknown = sorted(set(ablations) - PisamMilpConfig._KEYS)
    if unknown:
        raise ValueError(
            f"pisam_milp.ablations: unknown key(s) {unknown}. Allowed: "
            f"{', '.join(sorted(PisamMilpConfig._KEYS - _UNABLATABLE_KEYS))}"
        )
    blocked = sorted(set(ablations) & _UNABLATABLE_KEYS)
    if blocked:
        raise ValueError(
            f"pisam_milp.ablations: {blocked} cannot be ablated. `variant` is "
            f"chosen by the selected algorithm key, and `stop`/`eval` do not "
            f"enter the results label, so their arms would share one row."
        )
    for key, values in ablations.items():
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(
                f"pisam_milp.ablations.{key}: expected a non-empty list of values, "
                f"got {values!r}"
            )
