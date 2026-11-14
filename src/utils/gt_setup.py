"""Which endpoint states of a training trajectory are given as ground truth.

The four setups of the image experiments. A setup names the states that are
replaced by their ground truth in the data, and every arm that fixes states of
its own (a MILP, a CV anchor) fixes the same ones.

    none        no state is given; state 0 is classified like every other frame
    init        the initial state
    final       the final state
    init_final  both
"""

from __future__ import annotations

from enum import Enum
from typing import Set


class GtSetup(Enum):
    """The ground-truth endpoint states of a setup."""

    NONE = "none"
    INIT = "init"
    FINAL = "final"
    INIT_FINAL = "init_final"

    @property
    def gives_init(self) -> bool:
        """Whether the initial state is ground truth."""
        return self in (GtSetup.INIT, GtSetup.INIT_FINAL)

    @property
    def gives_final(self) -> bool:
        """Whether the final state is ground truth."""
        return self in (GtSetup.FINAL, GtSetup.INIT_FINAL)

    def state_indices(self, num_states: int) -> Set[int]:
        """The 0-based indices of the ground-truth states of a trajectory.

        Args:
            num_states: States in the trajectory (steps + 1).
        """
        indices: Set[int] = set()
        if num_states <= 0:
            return indices
        if self.gives_init:
            indices.add(0)
        if self.gives_final:
            indices.add(num_states - 1)
        return indices

    @property
    def name_suffix(self) -> str:
        """The experiment-name and corpus-name suffix; empty for ``init``."""
        return "" if self is GtSetup.INIT else f"__setup={self.value}"

    @classmethod
    def parse(cls, value: "str | GtSetup | None") -> "GtSetup":
        """A setup from its config value; ``None`` is ``init``.

        Raises:
            ValueError: on an unknown value.
        """
        if value is None:
            return cls.INIT
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError:
            raise ValueError(
                f"unknown gt_setup {value!r}; expected one of {[s.value for s in cls]}"
            ) from None
