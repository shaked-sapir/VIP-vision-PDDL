"""Turn scored checkpoints into anytime performance profiles.

The curve is a step function: its value at time *t* is the best score any of that
arm's models had reached by *t*. That definition is what makes checkpoint density
harmless. An arm that snapshots ten times as often does not score higher for it;
it only resolves *when* its improvements happened more finely. Only a better
model sooner moves the curve up.

The consequence worth stating out loud is the single-round arm's shape: nothing
at all until its solve returns, then one jump to a flat line. That is not a
defect of the plot, it is what a one-shot solver looks like when you ask what it
had on the table over time.

Unparseable checkpoints never advance the best. They are still drawn in the
scatter, because an arm emitting garbage mid-run is a fact about the arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from benchmark.evaluation.anytime.score import ScoredCheckpoint

# Fixed per arm so the same arm keeps its colour across every figure. CDPS and
# ROSAME reuse the colours their existing dashboard rows already use.
ARM_STYLES: Dict[str, Dict[str, str]] = {
    "cdps": {"color": "#1a73e8", "label": "CDPS"},
    "cdps_anchored": {"color": "#7b1fa2", "label": "CDPS (anchored)"},
    "pisam_milp_single_round": {"color": "#188038", "label": "PI-SAM+MILP (single round)"},
    "pisam_milp_loop": {"color": "#c5221f", "label": "PI-SAM+MILP (loop)"},
    "ROSAME_24": {"color": "#e8710a", "label": "ROSAME (24)"},
}


@dataclass(frozen=True)
class CurvePoint:
    """One step of a running-best profile."""

    elapsed_seconds: float
    best_success_rate: float
    source_index: int


def running_best(scored: Sequence[ScoredCheckpoint]) -> List[CurvePoint]:
    """The step function's corners: the first checkpoint, then each improvement.

    Ties are dropped rather than recorded. A step function is fully described by
    its corners, so repeating the incumbent's value would add points that draw
    the identical line -- and would make the number of corners look like activity
    when it is only re-measurement.
    """
    points: List[CurvePoint] = []
    best: Optional[float] = None

    for item in sorted(scored, key=lambda s: s.checkpoint.elapsed_seconds):
        if item.success_rate is None:
            continue
        if best is None or item.success_rate > best:
            best = item.success_rate
            points.append(
                CurvePoint(item.checkpoint.elapsed_seconds, best, item.checkpoint.index)
            )
    return points


def step_series(
    points: Sequence[CurvePoint], end_seconds: Optional[float] = None
) -> tuple[List[float], List[float]]:
    """``(xs, ys)`` drawing ``points`` as a staircase, held flat to ``end_seconds``.

    Extending to a common ``end_seconds`` is what lets several arms be read
    against each other: without it, the arm that stopped improving first appears
    to stop *existing* first.
    """
    xs: List[float] = []
    ys: List[float] = []
    for index, point in enumerate(points):
        if index > 0:
            xs.append(point.elapsed_seconds)
            ys.append(points[index - 1].best_success_rate)
        xs.append(point.elapsed_seconds)
        ys.append(point.best_success_rate)

    if points and end_seconds is not None and end_seconds > xs[-1]:
        xs.append(end_seconds)
        ys.append(ys[-1])
    return xs, ys


def plot_fold(
    scored_by_arm: Dict[str, List[ScoredCheckpoint]],
    out_path: Path,
    title: str = "",
    scatter_arms: Sequence[str] = ("pisam_milp_loop",),
) -> Path:
    """Draw one figure: every arm's profile, plus a scatter for chosen arms.

    Args:
        scored_by_arm: Output of :func:`score_fold`.
        out_path: Where to write the PNG.
        title: Figure title, typically ``domain / cell / fold``.
        scatter_arms: Arms whose individual checkpoints are also drawn. Defaults
            to the loop, whose rejected rounds are the point of the exercise --
            the profile alone would show only the winners and so would hide the
            search the loop actually did.
    """
    import matplotlib
    matplotlib.use("Agg")  # No display on the cluster, and none needed here.
    import matplotlib.pyplot as plt

    horizon = max(
        (s.checkpoint.elapsed_seconds for scored in scored_by_arm.values() for s in scored),
        default=0.0,
    )

    figure, axes = plt.subplots(figsize=(9, 5))
    for arm, scored in sorted(scored_by_arm.items()):
        style = ARM_STYLES.get(arm, {"color": "#5f6368", "label": arm})
        points = running_best(scored)
        if not points:
            continue

        xs, ys = step_series(points, end_seconds=horizon)
        axes.plot(xs, ys, color=style["color"], label=style["label"], linewidth=2)

        if arm in scatter_arms:
            axes.scatter(
                [s.checkpoint.elapsed_seconds for s in scored if s.success_rate is not None],
                [s.success_rate for s in scored if s.success_rate is not None],
                color=style["color"], alpha=0.35, s=18, zorder=1,
            )

    axes.set_xlabel("elapsed seconds")
    axes.set_ylabel("best-so-far success rate")
    axes.set_ylim(-0.02, 1.02)
    axes.grid(alpha=0.3)
    if title:
        axes.set_title(title)
    axes.legend(loc="lower right", fontsize=9)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path
