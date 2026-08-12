"""Configuration surface of the ``cdps_milp_*`` algorithms.

Mirrors the ``cdps_milp:`` block of ``benchmark/run_config.yaml`` (documented in
``docs/cdps-milp-loop-plan.md`` §5). Every enum-valued key is validated here and
a bad value raises immediately, listing the allowed options — configuration
mistakes must not surface as a silent behavior change three hours into a run.

Only the keys the ``single_round`` variant needs are accepted today; the loop
driver's keys (sampler, subset_size, learner_input, stop rules, eval weights)
land with the loop itself, and until then they are rejected as unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Type, TypeVar

from src.pi_sam.plan_denoising.milp_version.converter import GtAnchoring
from src.pi_sam.plan_denoising.milp_version.encoding_config import (
    MilpEncodingConfig,
    PriorWeightMode,
)

_E = TypeVar("_E", bound=Enum)


class MilpVariant(Enum):
    """Which ``cdps_milp_*`` algorithm to run."""

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


def _parse_enum(enum_cls: Type[_E], value: Any, key: str) -> _E:
    """Enum member from a YAML scalar, with an error naming the allowed options."""
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value).strip().lower())
    except ValueError:
        allowed = ", ".join(m.value for m in enum_cls)
        raise ValueError(
            f"cdps_milp.{key}: invalid value {value!r}. Allowed: {allowed}"
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
    raise ValueError(f"cdps_milp.{key}: invalid value {value!r}. Allowed: on, off")


@dataclass(frozen=True)
class CdpsMilpConfig:
    """Validated ``cdps_milp`` block.

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
        time_limit_seconds: Solver budget. ``None`` = inherit the fold's CDPS
            search budget, which is what makes the head-to-head comparison fair.
    """

    variant: MilpVariant = MilpVariant.SINGLE_ROUND
    eq16: bool = False
    lambda_pre: float = 0.4
    w_prior: PriorWeightMode = PriorWeightMode.TIEBREAK
    solver: MilpSolver = MilpSolver.CPSAT
    obs_weights: ObsWeighting = ObsWeighting.UNIFORM
    gt_anchoring: GtAnchoring = GtAnchoring.INIT_ONLY
    time_limit_seconds: Optional[int] = None

    _KEYS = frozenset({
        "variant", "eq16", "lambda_pre", "w_prior", "solver", "obs_weights",
        "gt_anchoring", "time_limit_seconds",
    })

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "CdpsMilpConfig":
        """Build (and validate) from the raw YAML mapping; ``None`` = all defaults."""
        if not raw:
            return cls()

        unknown = sorted(set(raw) - cls._KEYS)
        if unknown:
            raise ValueError(
                f"cdps_milp: unknown key(s) {unknown}. Allowed: "
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
        )

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
            f"cdps_milp.solver={self.solver.value!r} has no backend yet. "
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
        }
