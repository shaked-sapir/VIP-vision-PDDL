"""Abstract base class for pluggable baseline algorithm runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class BaselineRunner(ABC):
    """Interface for a competitor algorithm that runs alongside the primary learner.

    Subclasses represent a single algorithm variant (e.g. fully-observable
    ROSAME *or* partially-observable PO-ROSAME, not both).  The experiment
    pipeline filters runners via :meth:`supports_mode` so only the
    appropriate variant executes in a given experiment.
    """

    # ------------------------------------------------------------------ #
    # Identity / display
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def name(self) -> str:
        """Short machine-readable algorithm name (e.g. ``'ROSAME'``)."""
        ...

    @property
    def display_name(self) -> str:
        """Human-readable name for plots and reports.  Defaults to :attr:`name`."""
        return self.name

    @property
    def color(self) -> str:
        """Hex colour used in plots.  Defaults to grey."""
        return "#888888"

    # ------------------------------------------------------------------ #
    # Mode support
    # ------------------------------------------------------------------ #

    @abstractmethod
    def supports_mode(self, mode: str) -> bool:
        """Return ``True`` if this runner should be used in *mode*.

        Args:
            mode: ``'masked'`` (partial observability) or ``'fullyobs'``.
        """
        ...

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #

    @abstractmethod
    def learn(
        self,
        mode: str,
        domain_path: Path,
        prepared_trajectories: List[Tuple[Path, Path, Path]],
        work_dir: Path,
        timeout_seconds: int = 60,
        profiler=None,
    ) -> Tuple[Optional[str], Dict]:
        """Run the baseline learning algorithm.

        Args:
            mode: ``'masked'`` or ``'fullyobs'``.
            domain_path: Path to the reference PDDL domain file.
            prepared_trajectories: List of ``(trajectory_path, masking_info_path,
                problem_pddl_path)`` tuples — same format used throughout the
                experiment pipeline.
            work_dir: Scratch directory for this baseline's temporary files.
            timeout_seconds: Wall-clock budget (seconds).
            profiler: Optional profiler object (pass-through).

        Returns:
            ``(pddl_model_string | None, report_dict)`` where
            *report_dict* may be empty.
        """
        ...
