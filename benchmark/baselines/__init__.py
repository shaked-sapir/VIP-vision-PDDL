"""Pluggable baseline algorithm runners for experiment benchmarking.

Usage::

    from benchmark.baselines import get_baselines

    # From CLI arg like ``--baselines rosame``
    baselines = get_baselines(["rosame"])
"""

from __future__ import annotations

from typing import Dict, List, Type

from benchmark.baselines.base_runner import BaselineRunner
from benchmark.baselines.rosame_runner import RosameBaselineRunner

# Maps a short CLI name to the concrete runner classes it activates.
BASELINE_REGISTRY: Dict[str, List[Type[BaselineRunner]]] = {
    "rosame": [RosameBaselineRunner],
}


def get_baselines(names: List[str]) -> List[BaselineRunner]:
    """Instantiate baseline runners from CLI-provided names.

    Args:
        names: List of short names (keys of ``BASELINE_REGISTRY``).
            An empty list returns an empty list (no baselines).

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
            runners.append(cls())
    return runners


def resolve_baselines(names: List[str]) -> List[BaselineRunner]:
    """Resolve baseline names to runners, treating ``"none"`` as "skip".

    A single ``"none"`` (case-insensitive) yields an empty list. ``"none"`` may
    not be mixed with real baseline names. Any other list is delegated to
    :func:`get_baselines`.

    Args:
        names: List of short baseline names, or a single ``["none"]``.

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
    return get_baselines(names)
