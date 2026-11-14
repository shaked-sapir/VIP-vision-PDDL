"""Endpoint anchors of one ROSAME-I training trace, beyond its final-state atoms.

A prepared problem is ``(problem, image_paths, action_strings,
final_state_predicates)``; an optional fifth element of this type says which
final-state atoms are unknown and whether the initial state is given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class EndpointAnchors:
    """The parts of a trace's endpoint anchors its final-state atoms do not carry.

    Attributes:
        final_unknown: Atoms of the final-state anchor whose value is unknown;
            the final-state loss term skips them.
        init_predicates: Positive atoms of the ground-truth initial state, or
            ``None`` when the initial state is not anchored.
    """

    final_unknown: Tuple[str, ...] = ()
    init_predicates: Optional[Tuple[str, ...]] = None


def anchors_of(prepared_problem: Sequence[object]) -> Optional[EndpointAnchors]:
    """The fifth element of a prepared problem, or ``None`` for a four-element one."""
    return prepared_problem[4] if len(prepared_problem) > 4 else None
