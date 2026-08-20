"""Rule-set configuration for the MILP encoder (constraints + objective).

The MILP encoder (:class:`encoder.CPSATObservedActions`) is shared by two very
different callers:

- the ``rosame_milp*`` baselines — MILP as a *regularizer* of a neural learner,
  which want the upstream/paper rule set verbatim;
- the ``pisam_milp_*`` learners — MILP as a *denoiser* feeding PI-SAM, which need
  the rule set relaxed to whatever cannot exclude the ground-truth model.

Everything that differs between them is a named field here, so the encoder never
branches on "who is calling". Adding a new MILP variant = a new preset (+
optional enum member) and a thin runner/driver — no branching sprawl.

Two families of knobs live here:

1. **Constraint families** — ``schema_nonempty``, ``forbid_redundant_adds``,
   ``delete_implies_precondition``. The first two come from the released
   upstream code but are absent from the paper; the third *is* in the paper
   (eq. 18) but is still a modelling assumption that excludes legal PDDL
   action models. Each can make the GT model infeasible, so the CDPS dialect
   drops all three (see ``vendor/UPSTREAM.md`` and
   ``docs/pisam-milp-denoiser-design.md`` §3).
2. **Objective terms** — ``eq16``/``lambda_pre`` (ICAPS-26 precondition bias)
   and ``prior_weighting``/``tiebreak_mass`` (how strongly the reference model
   channel pulls the solution). See ``docs/pisam-milp-loop-plan.md`` §0.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SchemaNonemptyRule(Enum):
    """Per-schema "non-empty" constraint applied in ``build_domain_constraints``.

    - ``NONE``: no per-schema non-empty constraint at all.
    - ``PRE_AND_ADD``: upstream behavior — every schema must have >=1
      precondition AND >=1 add effect (``PreIsNotEmpty`` + ``AddIsNotEmpty``).
    - ``ADD``: every schema must have >=1 add effect only; the precondition
      requirement is dropped (the ``tag`` variant).
    """

    NONE = "none"
    PRE_AND_ADD = "pre_and_add"
    ADD = "add"


class PriorWeightMode(Enum):
    """How the reference-model ("model") objective channel is weighted.

    The channel prices agreement between the solved lifted ``pre/add/del`` bits
    and a reference model — a *soft* ROSAME prediction for the baselines, a
    *binary* ``M_best`` for the PI-SAM+MILP loop.

    - ``NONE``: no model term at all (rounds fully independent).
    - ``TIEBREAK``: the whole channel is normalized to a total mass of
      ``tiebreak_mass`` observed-fluent units, spread over the model bits, so
      the prior can never buy a single fluent flip — it only selects among
      repairs of equal cost. **Default for the PI-SAM+MILP loop.**
    - ``ROSAME``: upstream-faithful — one observed-fluent unit per model bit,
      scaled by the reference model's confidence (``2p - 1``). Required by the
      ``rosame_milp*`` baselines; an ablation arm for our loop.
    """

    NONE = "none"
    TIEBREAK = "tiebreak"
    ROSAME = "rosame"


@dataclass(frozen=True)
class MilpEncodingConfig:
    """Immutable bundle of the toggle-able MILP constraint families + objective
    terms.

    Attributes:
        schema_nonempty: Which per-schema non-empty rule to enforce.
        forbid_redundant_adds: If True, forbid ``stepadd & hol`` (adding an
            already-true fluent) — upstream ``StepAddPre``, not in the paper.
        delete_implies_precondition: If True, enforce ``del => pre`` (paper
            eq. 18). Legal PDDL allows deleting a non-precondition, so the CDPS
            dialect drops it.
        eq16: If True, add the ICAPS-26 eq. 16 precondition bias
            ``+ lambda_pre * pre[a, p, x]`` to the objective. Note this changes
            the repaired trajectories T', not just the witness model.
        lambda_pre: Coefficient of the eq. 16 bias, in observed-fluent units.
            Used only when ``eq16`` is True. Their published value is 0.4.
        prior_weighting: How to weight the reference-model objective channel.
        tiebreak_mass: Total mass (in observed-fluent units) of the model
            channel under ``PriorWeightMode.TIEBREAK``. Must stay < 1 so the
            prior can never outweigh a single fluent flip.
    """

    schema_nonempty: SchemaNonemptyRule = SchemaNonemptyRule.PRE_AND_ADD
    forbid_redundant_adds: bool = True
    delete_implies_precondition: bool = True
    eq16: bool = False
    lambda_pre: float = 0.4
    prior_weighting: PriorWeightMode = PriorWeightMode.ROSAME
    tiebreak_mass: float = 0.9

    def __post_init__(self) -> None:
        if self.lambda_pre < 0:
            raise ValueError(f"lambda_pre must be >= 0, got {self.lambda_pre}")
        if not 0 < self.tiebreak_mass < 1:
            raise ValueError(
                "tiebreak_mass must lie strictly in (0, 1) so the prior cannot "
                f"buy a fluent flip, got {self.tiebreak_mass}"
            )

    # ---------- presets ----------

    @classmethod
    def upstream(cls) -> "MilpEncodingConfig":
        """The released-code behavior (current default): >=1 pre & >=1 add,
        redundant adds forbidden, ``del => pre``, ROSAME-weighted model term."""
        return cls()

    @classmethod
    def tag(cls) -> "MilpEncodingConfig":
        """The ``rosame_milp_24_tag`` variant: >=1 add effect only, redundant adds
        allowed."""
        return cls(
            schema_nonempty=SchemaNonemptyRule.ADD,
            forbid_redundant_adds=False,
        )

    @classmethod
    def cdps_dialect(
        cls,
        eq16: bool = False,
        lambda_pre: float = 0.4,
        prior_weighting: PriorWeightMode = PriorWeightMode.NONE,
    ) -> "MilpEncodingConfig":
        """The ``pisam_milp_*`` dialect: every constraint family that can exclude
        a legal ground-truth model is dropped, so feasibility is guaranteed and
        the ``cost(MILP) <= cost(best CDPS CFM)`` lower bound holds.

        Args:
            eq16: Enable the eq. 16 precondition bias (invalidates the
                lower-bound check).
            lambda_pre: Bias coefficient, used only when ``eq16`` is True.
            prior_weighting: ``NONE`` for ``single_round`` (no reference model
                exists); ``TIEBREAK``/``ROSAME`` for the loop's later rounds.
        """
        return cls(
            schema_nonempty=SchemaNonemptyRule.NONE,
            forbid_redundant_adds=False,
            delete_implies_precondition=False,
            eq16=eq16,
            lambda_pre=lambda_pre,
            prior_weighting=prior_weighting,
        )

    # ---------- reporting ----------

    def as_stats(self) -> dict:
        """Flat dict for ``solve_stats`` / ``fold_result.json`` reporting.

        Keeps a derived ``enforce_nonempty_schemas`` boolean for continuity with
        the pre-config report columns.
        """
        return {
            "schema_nonempty": self.schema_nonempty.value,
            "forbid_redundant_adds": self.forbid_redundant_adds,
            "enforce_nonempty_schemas": self.schema_nonempty
            is SchemaNonemptyRule.PRE_AND_ADD,
            "delete_implies_precondition": self.delete_implies_precondition,
            "eq16": self.eq16,
            "lambda_pre": self.lambda_pre if self.eq16 else 0.0,
            "prior_weighting": self.prior_weighting.value,
        }
