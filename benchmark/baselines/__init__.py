"""Pluggable baseline algorithm runners for experiment benchmarking.

Usage::

    from benchmark.baselines import get_baselines

    # From CLI arg like ``--baselines rosame``
    baselines = get_baselines(["rosame"])
"""

from __future__ import annotations

from typing import Dict, List, Type

from benchmark.baselines.base_runner import BaselineRunner
from benchmark.baselines.rosame_runner import (
    PORosameBaselineRunner,
    RosameBaselineRunner,
)

# Maps a short CLI name to the concrete runner classes it activates.
# Multiple classes per key are common when an algorithm has separate
# fully-observable and partially-observable variants.
BASELINE_REGISTRY: Dict[str, List[Type[BaselineRunner]]] = {
    "rosame": [RosameBaselineRunner, PORosameBaselineRunner],
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
