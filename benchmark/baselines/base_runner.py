"""Abstract base class for pluggable baseline algorithm runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from benchmark.baselines.regime import DegradationRegime


class BaselineRunner(ABC):
    """Interface for a competitor algorithm that runs alongside CDPS."""

    # ------------------------------------------------------------------ #
    # Identity / display
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def name(self) -> str:
        """Short machine-readable algorithm name (e.g. ``'ROSAME'``)."""
        ...

    def row_name(self, domain_path: Path) -> str:
        """Label written to disk for this runner's result row.

        Defaults to :attr:`name`. Runners whose configuration varies per domain
        override this so two configurations never share one label.

        Args:
            domain_path: Reference PDDL domain file — the same value passed to
                :meth:`learn`, so the label and the run resolve from one input.
        """
        return self.name

    @property
    def display_name(self) -> str:
        """Human-readable name for plots and reports.  Defaults to :attr:`name`."""
        return self.name

    @property
    def color(self) -> str:
        """Hex colour used in plots.  Defaults to grey."""
        return "#888888"

    # ------------------------------------------------------------------ #
    # Factors
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def input_kind(self) -> str:
        """``'symbolic'`` or ``'imaged'`` — what the arm learns from."""
        ...

    @property
    @abstractmethod
    def paper(self) -> str:
        """Publication tag of the arm's source paper (``'24'``, ``'26'``, ``'icaps24'``, ...)."""
        ...

    @property
    @abstractmethod
    def uses_milp(self) -> bool:
        """Whether a MILP solve is part of this arm."""
        ...

    def factors(self) -> Dict[str, object]:
        """The arm's factors, for the result row."""
        return {
            "input_kind": self.input_kind,
            "paper": self.paper,
            "uses_milp": self.uses_milp,
        }

    def run_params(self) -> Dict[str, object]:
        """The training knobs this arm ran with, recorded in ``run_params.json``."""
        return {}

    # ------------------------------------------------------------------ #
    # Regime
    # ------------------------------------------------------------------ #

    def supports(self, regime: "DegradationRegime") -> Tuple[bool, str]:
        """Whether this arm is a valid competitor in ``regime``, and why not.

        Defaults to ``(True, "")``. An arm built for one degradation axis only
        (noise but no masking, or the reverse) returns ``(False, reason)`` for
        the cells outside it; the gate records the reason in the run manifest.
        """
        return True, ""

    def for_regime(self, regime: "DegradationRegime") -> "BaselineRunner":
        """The runner to use in ``regime``; ``self`` unless the arm reads the cell.

        An arm whose learning depends on the cell (a noise level to hand the
        learner, GT trajectories to measure it from) returns a bound copy so the
        run-wide runner stays cell-agnostic.
        """
        return self

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #

    @abstractmethod
    def learn(
        self,
        domain_path: Path,
        prepared_trajectories: List[Tuple[Path, Path, Path]],
        work_dir: Path,
        timeout_seconds: int = 60,
    ) -> Tuple[Optional[str], Dict]:
        """Run the baseline learning algorithm on the (degraded) trajectories.

        Args:
            domain_path: Path to the reference PDDL domain file.
            prepared_trajectories: List of ``(trajectory_path, masking_info_path,
                problem_pddl_path)`` tuples — same format used throughout the
                experiment pipeline.
            work_dir: Scratch directory for this baseline's temporary files.
            timeout_seconds: Wall-clock budget (seconds).

        Returns:
            ``(pddl_model_string | None, report_dict)`` where
            *report_dict* may be empty.
        """
        ...
