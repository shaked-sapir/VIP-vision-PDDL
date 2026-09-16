"""The degradation regime of one experiment cell, and the gate that keeps a
baseline out of the cells it was not built for.

A cell's regime is what the data source did to the observations: nothing
beyond the classifier's own errors (``image``), or a synthetic masking rate and
flip rate (``simulated``). A :class:`BaselineRunner` declares which regimes it
is a valid competitor in through :meth:`BaselineRunner.supports`; the gate
applies that per cell and reports what it dropped, so a run's manifest says
which arms were absent on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from benchmark.baselines.base_runner import BaselineRunner

SIMULATED = "simulated"
IMAGE = "image"
_SOURCES = (SIMULATED, IMAGE)

# ``run_params.json`` records the data source by class name.
_SIMULATED_SOURCE_TYPE = "SimulatedDataSource"


@dataclass(frozen=True)
class DegradationRegime:
    """How one cell's observations were degraded.

    Attributes:
        source: ``"simulated"`` or ``"image"``.
        masking_p: The cell's nominal masking rate; ``None`` for image cells.
        noising_p: The cell's nominal flip rate; ``None`` for image cells.
        data_dir: The corpus root the cell was generated from, where
            ``gt_trajectories/`` lives. Optional; a runner that needs GT
            trajectories reads them from here.
    """

    source: str
    masking_p: Optional[float] = None
    noising_p: Optional[float] = None
    data_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.source not in _SOURCES:
            raise ValueError(
                f"regime source must be one of {_SOURCES}, got {self.source!r}"
            )
        if self.source == SIMULATED and (self.masking_p is None or self.noising_p is None):
            raise ValueError("a simulated regime needs both masking_p and noising_p")

    @classmethod
    def simulated(
        cls, masking_p: float, noising_p: float, data_dir: Optional[Path] = None
    ) -> "DegradationRegime":
        """A simulated cell with the given nominal rates."""
        return cls(SIMULATED, float(masking_p), float(noising_p), data_dir)

    @classmethod
    def image(cls, data_dir: Optional[Path] = None) -> "DegradationRegime":
        """An image cell: the classifier's own masking and errors, unquantified."""
        return cls(IMAGE, None, None, data_dir)

    @classmethod
    def from_run_params(
        cls, run_params: dict, data_dir: Optional[Path] = None
    ) -> "DegradationRegime":
        """The regime a finished experiment's ``run_params.json`` records.

        Raises:
            KeyError: If a simulated run lacks ``p_mask`` / ``p_noise``.
        """
        if run_params.get("data_source_type") == _SIMULATED_SOURCE_TYPE:
            return cls.simulated(run_params["p_mask"], run_params["p_noise"], data_dir)
        return cls.image(data_dir)

    @property
    def is_simulated(self) -> bool:
        return self.source == SIMULATED

    @property
    def mask_free(self) -> bool:
        """True when no fluent was hidden: simulated with ``masking_p == 0``."""
        return self.is_simulated and self.masking_p == 0.0

    @property
    def noise_free(self) -> bool:
        """True when no fluent was flipped: simulated with ``noising_p == 0``."""
        return self.is_simulated and self.noising_p == 0.0

    def describe(self) -> str:
        """One-line label, e.g. ``simulated(mask=0.1, noise=0.0)``."""
        if not self.is_simulated:
            return IMAGE
        return f"{SIMULATED}(mask={self.masking_p:g}, noise={self.noising_p:g})"


def gate_baselines(
    runners: Sequence["BaselineRunner"],
    regime: DegradationRegime,
    strict: bool = True,
) -> Tuple[List["BaselineRunner"], Dict[str, str]]:
    """Keep the runners valid in ``regime``, bound to it; name the rest.

    Args:
        runners: The run's resolved baseline runners.
        regime: The cell about to run.
        strict: ``False`` keeps every runner regardless of what it supports
            (a deliberate misapplication study); binding still happens.

    Returns:
        ``(kept, dropped)`` where ``kept`` holds each surviving runner's
        :meth:`BaselineRunner.for_regime` result, in the input order, and
        ``dropped`` maps a dropped runner's :attr:`BaselineRunner.name` to the
        reason it gave.
    """
    kept: List["BaselineRunner"] = []
    dropped: Dict[str, str] = {}
    for runner in runners:
        if strict:
            ok, reason = runner.supports(regime)
            if not ok:
                dropped[runner.name] = reason
                continue
        kept.append(runner.for_regime(regime))
    return kept, dropped
