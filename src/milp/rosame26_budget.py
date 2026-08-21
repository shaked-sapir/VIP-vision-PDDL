"""The pre-flight budget check the ICAPS-26 defaults require (plan §1.2).

``epoch: 5000`` with ``pre_mip_epoch: 50`` and ``mip_interval: 1`` schedules
**4950 CP-SAT solves per cell**. Read off ``mip_time_limit: 60`` that is ~82
hours; measured — 630 samples, median solve 0.318 s, the cap never binding — it
is ~70 minutes. Both are far past the 600 s every other arm in the grid gets,
and the grid's point is a like-for-like comparison, so the arm keeps that budget
and this module decides whether a configuration fits it.

The projection is the plan's::

    projected_solves  = (epochs - pre_mip_epoch) / mip_interval
    projected_seconds = n_seeds * (projected_solves * per_solve
                                   + epochs * per_epoch_dl)

and :func:`check_budget` refuses to start when it exceeds the timeout, naming
the epoch count that would fit. Refusing before epoch 0 rather than discovering
it at epoch 4000 is the whole point: the operator is told what to override
before burning a grid, not after.

§1.2 also asks for the estimate to be **re-projected once against a measurement
of the run itself**, and :func:`reproject` is that. The seeded constants are one
domain's, and the domains differ by roughly 2x — measured at 131 epochs x 3
seeds against one 480 s budget, blocksworld took 185 s, depot 231 s and hanoi
360 s, tracking grounding width (36/50, 49/122, 55/120 propositions/actions).
So the seeded projection is slack by 1.3x-2.6x, which is the right direction for
a refuse-to-start guard and the wrong one for an epoch budget: it costs real
epochs on the cheap domains. Timing a short probe and re-projecting recovers
them without needing a per-domain table nobody would maintain.

``per_solve`` is **measured**, not ``mip_time_limit``. Projecting from the cap
refuses ``epochs > 60`` on a configuration that in fact finishes in a tenth of
the budget, which would make the check unusable — see :data:`PER_SOLVE_SECONDS`.

WHAT THIS MODULE DOES NOT DO. It does not ration ``mip_time_limit`` per solve.
Upstream passes the constant (``network.py:303``) and has no per-solve scheduler
at all; ours (``milp_loop_i.py``'s ``_solve_time_limit``) floors each solve at
5 s, which against a 600 s cell would force ``mip_interval`` wide and trip the
``mip_interval_used == mip_interval`` assertion on every cell. The budget is
owned here instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

#: Median CP-SAT solve, seconds, over 630 samples of the real encoder (§1.2).
#: The nominal ``mip_time_limit`` of 60 s is a cap the measurements never reach.
PER_SOLVE_SECONDS: float = 0.318

#: One DL epoch over a fold, seconds. Measured on CPU over a 3-trace
#: blocksworld fold at resize 64, 60 epochs: 72.9 s, so 1.216 s/epoch. A fold is
#: 3-9 traces and one batch, so this is dominated by the ResNet-18 forward and
#: backward rather than by the fold size, and it is the term that dominates the
#: projection -- at this cost the DL is ~3.8x the MILP even at mip_interval 1.
#: A cuda run is far cheaper; the check is then simply slack, which is the
#: direction a refuse-to-start guard should err in.
PER_EPOCH_DL_SECONDS: float = 1.216

#: Fraction of the cell timeout the projection may claim. The remainder covers
#: what the projection does not model: adapter, grounding, decode and eval.
BUDGET_HEADROOM: float = 0.8


class BudgetExceededError(RuntimeError):
    """A configuration whose projected cost does not fit the cell timeout."""


@dataclass(frozen=True)
class Projection:
    """What a configuration is projected to cost, and what would fit.

    Attributes:
        solves: CP-SAT solves the schedule asks for, across all seeds.
        seconds: Projected wall clock, across all seeds.
        budget: The share of the timeout the projection may claim.
        fits: Whether ``seconds`` is within ``budget``.
        max_epochs: The largest epoch count that would fit, everything else
            held fixed. Zero means nothing fits.
    """

    solves: int
    seconds: float
    budget: float
    fits: bool
    max_epochs: int


def project(
    *,
    epochs: int,
    pre_mip_epoch: int,
    mip_interval: int,
    timeout_seconds: float,
    n_seeds: int = 1,
    per_solve: float = PER_SOLVE_SECONDS,
    per_epoch_dl: float = PER_EPOCH_DL_SECONDS,
    headroom: float = BUDGET_HEADROOM,
) -> Projection:
    """Project one cell's cost, and the epoch count that would fit it.

    ``n_seeds`` multiplies the whole projection: our arms run several seeds and
    keep the lowest final loss, so a check that ignores it passes while the grid
    overruns by ``n_seeds``\\ ×.

    Raises:
        ValueError: on a non-positive ``mip_interval``, ``n_seeds`` or timeout.
    """
    if mip_interval < 1:
        raise ValueError(f"mip_interval must be at least 1, got {mip_interval}")
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be at least 1, got {n_seeds}")
    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")

    solving_epochs = max(0, epochs - max(0, pre_mip_epoch))
    solves = solving_epochs // mip_interval + (1 if solving_epochs % mip_interval else 0)
    per_seed = solves * per_solve + epochs * per_epoch_dl
    budget = timeout_seconds * headroom

    return Projection(
        solves=solves * n_seeds,
        seconds=per_seed * n_seeds,
        budget=budget,
        fits=per_seed * n_seeds <= budget,
        max_epochs=_max_epochs(
            pre_mip_epoch=pre_mip_epoch,
            mip_interval=mip_interval,
            budget=budget,
            n_seeds=n_seeds,
            per_solve=per_solve,
            per_epoch_dl=per_epoch_dl,
        ),
    )


def check_budget(
    *,
    epochs: int,
    pre_mip_epoch: int,
    mip_interval: int,
    timeout_seconds: float,
    n_seeds: int = 1,
    per_solve: float = PER_SOLVE_SECONDS,
    per_epoch_dl: float = PER_EPOCH_DL_SECONDS,
    headroom: float = BUDGET_HEADROOM,
) -> Projection:
    """:func:`project`, raising when the configuration does not fit.

    Returns:
        The projection, so a caller can record it whether or not it bound.

    Raises:
        BudgetExceededError: with the epoch count that would fit.
    """
    projection = project(
        epochs=epochs,
        pre_mip_epoch=pre_mip_epoch,
        mip_interval=mip_interval,
        timeout_seconds=timeout_seconds,
        n_seeds=n_seeds,
        per_solve=per_solve,
        per_epoch_dl=per_epoch_dl,
        headroom=headroom,
    )
    if projection.fits:
        return projection

    raise BudgetExceededError(
        f"{epochs} epochs x {n_seeds} seed(s) project to "
        f"{projection.seconds:.0f} s ({projection.solves} CP-SAT solves), past "
        f"the {projection.budget:.0f} s this cell's {timeout_seconds:.0f} s "
        f"budget allows. Set epochs <= {projection.max_epochs}"
        + (
            f", or widen mip_interval past {mip_interval}"
            if projection.solves
            else ""
        )
        + ", or run this cell outside the timeout as a control."
    )


def reproject(
    *,
    epochs: int,
    measured_seconds: float,
    measured_epochs: int,
    pre_mip_epoch: int,
    mip_interval: int,
    timeout_seconds: float,
    elapsed_seconds: float = 0.0,
    n_seeds: int = 1,
    per_solve: float = PER_SOLVE_SECONDS,
    headroom: float = BUDGET_HEADROOM,
) -> Projection:
    """:func:`project` again, with the DL cost measured on this run (§1.2).

    Args:
        epochs: The count currently planned.
        measured_seconds: Wall clock the probe took.
        measured_epochs: Epochs the probe ran. Must be positive.
        elapsed_seconds: What the run has already spent, the probe included;
            deducted from the budget so the re-projection is against what is
            left rather than against the whole cell.

    Raises:
        ValueError: if ``measured_epochs`` is not positive.
    """
    if measured_epochs < 1:
        raise ValueError(
            f"a probe must run at least one epoch, got {measured_epochs}"
        )
    remaining = max(timeout_seconds * headroom - elapsed_seconds, 0.0)
    # `project` applies the headroom itself, so hand it a timeout that survives
    # the multiplication as the budget already left.
    return project(
        epochs=epochs,
        pre_mip_epoch=pre_mip_epoch,
        mip_interval=mip_interval,
        timeout_seconds=max(remaining / headroom, 1e-9),
        n_seeds=n_seeds,
        per_solve=per_solve,
        per_epoch_dl=measured_seconds / measured_epochs,
        headroom=headroom,
    )


def projection_for(
    parameters: Mapping[str, Any],
    timeout_seconds: float,
    n_seeds: int = 1,
    **overrides: Any,
) -> Projection:
    """:func:`project` read off a :func:`~src.milp.rosame26_training.default_parameters`."""
    return project(
        epochs=parameters["epoch"],
        pre_mip_epoch=parameters["pre_mip_epoch"],
        mip_interval=parameters["mip_interval"],
        timeout_seconds=timeout_seconds,
        n_seeds=n_seeds,
        **overrides,
    )


def _max_epochs(
    *,
    pre_mip_epoch: int,
    mip_interval: int,
    budget: float,
    n_seeds: int,
    per_solve: float,
    per_epoch_dl: float,
) -> int:
    """The largest epoch count fitting ``budget``, everything else held fixed.

    Solved by bisection rather than in closed form: the solve count is a
    ``ceil`` over the epochs past ``pre_mip_epoch``, so the cost is a step
    function of the epoch count and the closed form would be off by one either
    way at every step.
    """

    def cost(epochs: int) -> float:
        solving = max(0, epochs - max(0, pre_mip_epoch))
        solves = solving // mip_interval + (1 if solving % mip_interval else 0)
        return n_seeds * (solves * per_solve + epochs * per_epoch_dl)

    if cost(1) > budget:
        return 0

    low, high = 1, 2
    while cost(high) <= budget:
        low, high = high, high * 2
    while low + 1 < high:
        middle = (low + high) // 2
        if cost(middle) <= budget:
            low = middle
        else:
            high = middle
    return low
