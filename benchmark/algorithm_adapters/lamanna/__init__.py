"""Shared glue for the Lamanna et al. learners (OffLAM, NOLAM).

Both packages parse one trace grammar (explicit positive and negative
literals, absent means unknown), take no time budget, call ``exit()`` on a
parse error and write relative to the cwd. This package supplies the three
things every runner over them needs: the trace writer (:mod:`traces`), the
isolated learner call (:mod:`subprocess_learn`) and the measured corruption
rates of a fold (:mod:`realised_rates`).
"""

from benchmark.algorithm_adapters.lamanna.realised_rates import (
    RealisedRates,
    gt_trajectory_lookup,
    realised_rates,
)
from benchmark.algorithm_adapters.lamanna.subprocess_learn import (
    COMPLETED,
    ERROR,
    TIMEOUT,
    LearnOutcome,
    run_learner,
)
from benchmark.algorithm_adapters.lamanna.traces import (
    format_trajectory,
    state_literals,
    write_partial_trace,
)

__all__ = [
    "COMPLETED",
    "ERROR",
    "TIMEOUT",
    "LearnOutcome",
    "RealisedRates",
    "format_trajectory",
    "gt_trajectory_lookup",
    "realised_rates",
    "run_learner",
    "state_literals",
    "write_partial_trace",
]
