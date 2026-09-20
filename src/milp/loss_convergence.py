"""Training-loss convergence: a windowed relative-plateau rule.

The per-epoch training loss is cut into consecutive ``window``-epoch blocks,
each summarised by its lowest loss. A block counts as progress when it beats
the running best by at least ``min_improvement`` (a fraction of that best);
``patience`` consecutive blocks without progress, after ``min_epochs``, is
convergence. Reads training loss only, never test data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Sequence

# Stop-reason vocabulary shared by every ROSAME arm.
CONVERGED = "converged"
EPOCHS_EXHAUSTED = "epochs_exhausted"
TIMEOUT = "timeout"
NO_USABLE_TRACES = "no_usable_traces"
AGREEMENT_REACHED = "agreement_reached"


def window_best(losses: Sequence[float], window: int) -> List[float]:
    """The best loss within each consecutive ``window``-epoch block.

    A trailing partial block is dropped: comparing a 20-epoch window against a
    3-epoch one would read as improvement whenever the short block happens to
    contain a good epoch.
    """
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    complete = len(losses) // window
    return [min(losses[i * window : (i + 1) * window]) for i in range(complete)]


def relative_improvements(bests: Sequence[float]) -> List[float]:
    """Fractional improvement of each window over the best window before it.

    Positive means a new best; zero or negative means the window did not beat
    what the run had already achieved. Normalised by the running best's
    magnitude, so the figure is comparable across domains whose losses differ
    in scale. A running best of exactly zero yields ``0.0``.
    """
    out: List[float] = []
    for index in range(1, len(bests)):
        best_before = min(bests[:index])
        scale = abs(best_before)
        out.append(0.0 if scale == 0.0 else (best_before - bests[index]) / scale)
    return out


def has_converged(
    losses: Sequence[float],
    *,
    window: int,
    min_improvement: float,
    patience: int,
    min_epochs: int,
) -> bool:
    """Whether the training loss has plateaued.

    Args:
        losses: Per-epoch training loss so far, in order.
        window: Epochs per window.
        min_improvement: Fractional improvement below which a window is a
            plateau. ``0.01`` is 1%.
        patience: Consecutive plateau windows required.
        min_epochs: Floor before convergence may trigger.

    Returns:
        ``True`` when training should stop.
    """
    if len(losses) < min_epochs:
        return False
    improvements = relative_improvements(window_best(losses, window))
    if len(improvements) < patience:
        return False
    return all(value < min_improvement for value in improvements[-patience:])


@dataclass(frozen=True)
class LossConvergenceRule:
    """The plateau rule's four knobs.

    Attributes:
        window: Epochs per block; the noise filter.
        min_improvement: Fraction of the running best a block must beat it by
            to count as progress.
        patience: Consecutive blocks without progress required to stop;
            ``window * patience`` is the number of epochs the loss may go
            without a meaningful improvement.
        min_epochs: Floor before the rule may fire. For an arm whose series
            restarts (the MILP arms, at the first solve) it counts from there.
    """

    window: int = 10
    min_improvement: float = 0.002
    patience: int = 3
    min_epochs: int = 50

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
        if self.min_improvement < 0:
            raise ValueError(f"min_improvement must be >= 0, got {self.min_improvement}")
        if self.min_epochs < 0:
            raise ValueError(f"min_epochs must be >= 0, got {self.min_epochs}")

    @classmethod
    def from_config(cls, block: Optional[Mapping[str, object]]) -> Optional["LossConvergenceRule"]:
        """The rule a config block describes; ``None`` for a missing block (rule off).

        Raises:
            ValueError: On a key the rule does not have.
        """
        if block is None:
            return None
        if isinstance(block, cls):
            return block
        unknown = set(block) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"unknown rosame_convergence key(s) {sorted(unknown)}; "
                f"known: {sorted(cls.__dataclass_fields__)}"
            )
        return cls(**dict(block))

    def converged(self, losses: Sequence[float]) -> bool:
        """Whether ``losses`` has plateaued under this rule."""
        return has_converged(
            losses,
            window=self.window,
            min_improvement=self.min_improvement,
            patience=self.patience,
            min_epochs=self.min_epochs,
        )

    def as_stats(self) -> Dict[str, object]:
        """The knobs, for a result row or ``run_params.json``."""
        return asdict(self)
