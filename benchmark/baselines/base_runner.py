"""Abstract base class for pluggable baseline algorithm runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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

    @property
    def display_name(self) -> str:
        """Human-readable name for plots and reports.  Defaults to :attr:`name`."""
        return self.name

    @property
    def color(self) -> str:
        """Hex colour used in plots.  Defaults to grey."""
        return "#888888"

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
