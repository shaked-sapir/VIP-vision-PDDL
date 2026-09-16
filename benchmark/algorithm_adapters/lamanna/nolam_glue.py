"""The in-child call into the ``nolam`` package.

Runs inside :func:`run_learner`'s child process, so the module-global
configuration it sets stays there.
"""

from __future__ import annotations

from typing import List


def _is_empty(operator) -> bool:
    return not (operator.precs_pos or operator.precs_neg or operator.eff_pos or operator.eff_neg)


def learn_nolam(
    domain_path: str,
    trace_paths: List[str],
    e: float,
    allow_neg_precs: bool,
    seed: int,
) -> str:
    """Learn a model with NOLAM's MAP criterion and return its PDDL text.

    An operator NOLAM leaves with no preconditions and no effects (one never
    observed, or one whose every atom got the "neither" hypothesis) is left
    out of the text: PI-SAM emits only observed operators, AMLGym's syntactic
    metrics score a missing operator and an empty one alike, and an empty
    ``(and)`` makes Fast Downward fail with an internal error that the solving
    metric cannot classify.

    Args:
        domain_path: The reference domain (predicates, types, operator signatures).
        trace_paths: Trace files in the grammar :mod:`traces` writes.
        e: The per-atom flip probability NOLAM conditions on.
        allow_neg_precs: ``True`` is the paper's ``MAP``, ``False`` its ``MAP_pre+``.
        seed: NumPy seed for MAP tie-breaking.
    """
    import numpy as np
    from nolam.algorithm import Configuration
    from nolam.algorithm.Learner import Learner

    Configuration.ALLOW_PREC_NEG = bool(allow_neg_precs)
    Configuration.SAMPLING = False
    np.random.seed(seed)
    model = Learner().learn(domain_path, list(trace_paths), e)

    empty = [op.operator_name for op in model.operators if _is_empty(op)]
    if empty:
        print(f"[nolam_glue] dropping empty operators: {', '.join(empty)}")
        model.operators = [op for op in model.operators if not _is_empty(op)]
    return str(model)
