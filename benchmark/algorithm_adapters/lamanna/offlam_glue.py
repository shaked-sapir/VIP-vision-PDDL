"""The in-child call into the ``offlam`` package.

Runs inside :func:`run_learner`'s child process: ``offlam.algorithm.learn``
writes ``PDDL/`` relative to the cwd and removes it on the way out.
"""

from __future__ import annotations

from typing import List

# The package emits an empty requirements block; AMLGym's adapter patches it too.
_EMPTY_REQUIREMENTS = "(:requirements)"
_TYPED_REQUIREMENTS = "(:requirements :typing)"


def learn_offlam(domain_path: str, trace_paths: List[str]) -> str:
    """Learn OffLAM's cautious model and return its PDDL text.

    Args:
        domain_path: The reference domain (predicates, types, operator signatures).
        trace_paths: Trace files in the grammar :mod:`traces` writes.
    """
    from offlam.algorithm import learn

    model = learn(domain_path, list(trace_paths))
    return model.replace(_EMPTY_REQUIREMENTS, _TYPED_REQUIREMENTS)
