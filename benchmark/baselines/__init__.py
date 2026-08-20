"""Pluggable baseline algorithm runners for experiment benchmarking.

Usage::

    from benchmark.baselines import get_baselines

    # From CLI arg like ``--baselines rosame``
    baselines = get_baselines(["rosame"])
"""

from __future__ import annotations

import inspect
from typing import Dict, List, Type

from benchmark.baselines.base_runner import BaselineRunner
from benchmark.baselines.rosame_runner import RosameBaselineRunner
from benchmark.baselines.rosame_i_runner import (
    ResizeSpec,
    RosameIBaselineRunner,
    _RESIZE_FROM_TABLE as RESIZE_FROM_TABLE,  # sentinel: use the per-domain table
)
from benchmark.baselines.rosame_i_milp_runner import RosameIMilpRunner
from benchmark.baselines.rosame_milp_runner import (
    RosameMilpBaseRunner,
    RosameMilpRunner,
    RosameMilpTagRunner,
)

# Maps a short CLI name to the concrete runner classes it activates.
BASELINE_REGISTRY: Dict[str, List[Type[BaselineRunner]]] = {
    "rosame": [RosameBaselineRunner],
    "rosame_i": [RosameIBaselineRunner],
    "rosame_i_milp": [RosameIMilpRunner],
    "rosame_milp": [RosameMilpRunner],
    "rosame_milp_tag": [RosameMilpTagRunner],
    "rosame_milp_base": [RosameMilpBaseRunner],
}


def _instantiate(cls: Type[BaselineRunner], **kwargs) -> BaselineRunner:
    """Instantiate ``cls`` forwarding only the kwargs its ``__init__`` accepts.

    Lets us thread runner-specific options (e.g. ``train_per_trajectory`` for
    the symbolic ROSAME runners) without leaking them into runners that don't
    take them — ROSAME-I among them, since it trains pooled unconditionally.
    """
    params = inspect.signature(cls.__init__).parameters
    accepted = {k: v for k, v in kwargs.items() if k in params}
    return cls(**accepted)


def get_baselines(names: List[str], **runner_kwargs) -> List[BaselineRunner]:
    """Instantiate baseline runners from CLI-provided names.

    Args:
        names: List of short names (keys of ``BASELINE_REGISTRY``).
            An empty list returns an empty list (no baselines).
        **runner_kwargs: Optional per-runner options forwarded only to the
            runners whose ``__init__`` accepts them (e.g.
            ``train_per_trajectory`` for the symbolic ROSAME runners).

    Returns:
        Flat list of instantiated ``BaselineRunner`` objects.

    Raises:
        ValueError: If a name is not found in the registry.
    """
    runners: List[BaselineRunner] = []
    for name in names:
        key = name.strip().lower()
        if key not in BASELINE_REGISTRY:
            available = ", ".join(sorted(BASELINE_REGISTRY.keys()))
            raise ValueError(
                f"Unknown baseline '{key}'. Available: {available}"
            )
        for cls in BASELINE_REGISTRY[key]:
            runners.append(_instantiate(cls, **runner_kwargs))
    return runners


def resolve_baselines(names: List[str], **runner_kwargs) -> List[BaselineRunner]:
    """Resolve baseline names to runners, treating ``"none"`` as "skip".

    A single ``"none"`` (case-insensitive) yields an empty list. ``"none"`` may
    not be mixed with real baseline names. Any other list is delegated to
    :func:`get_baselines`.

    Args:
        names: List of short baseline names, or a single ``["none"]``.
        **runner_kwargs: Optional per-runner options (see :func:`get_baselines`).

    Returns:
        Instantiated ``BaselineRunner`` objects (empty for ``["none"]``).

    Raises:
        ValueError: If ``"none"`` is combined with other names, or a name is
            not found in the registry.
    """
    lowered = [name.strip().lower() for name in names]
    if "none" in lowered:
        if len(lowered) > 1:
            raise ValueError("Cannot combine 'none' with other baseline names")
        return []
    return get_baselines(names, **runner_kwargs)
