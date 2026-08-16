"""The one record every trace source produces and the cutter/emitter consume.

States and actions are carried in the *typed eval schema* — the same strings the
``_trajectory.json`` artifacts use:

    literal  ``"on(a:block,b:block)"``   (nullary: ``"handempty()"``)
    object   ``"a:block"``
    action   ``"pick_up(a:block)"``

``src.utils.pddl_gym.parse_gym_to_pddl_literal`` /
``parse_gym_to_pddl_ground_action`` turn these into the untyped ``.trajectory``
and ``.pddl`` forms, so a source that produces typed strings gets every
downstream artifact for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, Optional, Tuple


@dataclass(frozen=True)
class TraceState:
    """One fully observed state: its positive fluents and its object universe."""

    literals: Tuple[str, ...]
    objects: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.literals, tuple) or not isinstance(self.objects, tuple):
            raise TypeError(
                "TraceState.literals and .objects must be tuples "
                f"(got {type(self.literals).__name__} / {type(self.objects).__name__})."
            )

    def fluent_set(self) -> FrozenSet[str]:
        """The state's positive fluents as a set, for order-insensitive comparison."""
        return frozenset(self.literals)


@dataclass(frozen=True)
class TraceStep:
    """One (state, action, state) transition of a trace.

    Frames are populated only by rendering sources; symbolic sources leave them
    ``None`` and the emitter then writes no images.
    """

    prev_state: TraceState
    action: str
    next_state: TraceState
    frame_before: Optional[Path] = None
    frame_after: Optional[Path] = None

    @property
    def has_frames(self) -> bool:
        """True when both frames are present, so the step can be rendered."""
        return self.frame_before is not None and self.frame_after is not None
