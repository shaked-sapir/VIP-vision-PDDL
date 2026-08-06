"""Rule-set configuration for the ROSAME+MILP encoder.

The MILP encoder (:class:`encoder.CPSATObservedActions`) mixes paper-faithful
constraints (eqs. 17-18, 33-37) with a couple of constraint families that come
from the released upstream code but are absent from the paper. Those extra
families are the ones we want to toggle per experiment, so they live here in an
immutable config object with named presets (mirroring ``CDPSConfig`` in
``src/pi_sam/plan_denoising/conflict_search_config.py``).

Adding a new MILP variant = a new preset here (+ optional enum member) and a
thin runner subclass — no branching sprawl inside the encoder.
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


@dataclass(frozen=True)
class MilpEncodingConfig:
    """Immutable bundle of the toggle-able MILP constraint families.

    Attributes:
        schema_nonempty: Which per-schema non-empty rule to enforce.
        forbid_redundant_adds: If True, forbid ``stepadd & hol`` (adding an
            already-true fluent) — upstream ``StepAddPre``, not in the paper.
    """

    schema_nonempty: SchemaNonemptyRule = SchemaNonemptyRule.PRE_AND_ADD
    forbid_redundant_adds: bool = True

    @classmethod
    def upstream(cls) -> "MilpEncodingConfig":
        """The released-code behavior (current default): >=1 pre & >=1 add,
        redundant adds forbidden."""
        return cls()

    @classmethod
    def tag(cls) -> "MilpEncodingConfig":
        """The ``rosame_milp_tag`` variant: >=1 add effect only, redundant adds
        allowed."""
        return cls(
            schema_nonempty=SchemaNonemptyRule.ADD,
            forbid_redundant_adds=False,
        )

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
        }
