"""Render eval-schema states with PDDLGym's own domain renderers.

PDDLGym renders from a set of ground ``Literal`` objects in its *gym* schema,
which differs from the eval schema the trace pipeline carries (hanoi splits
``on`` into ``on-disc`` / ``on-peg``; npuzzle merges ``x`` / ``y`` coordinates
into ``p_X_Y`` positions). A :class:`GymSchema` spells an eval-schema literal
back into the gym predicate and arguments the renderer reads.

The literal set alone decides the image, so the cache key is the state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pddlgym.structs import Literal, Predicate, Type

from src.trace_generation.frame_renderer import state_key
from src.trace_generation.trace_step import TraceState

GymLiteral = Tuple[str, List[Tuple[str, str]]]
"""``(predicate_name, [(object_name, object_type), ...])`` in the gym schema."""

_LITERAL_RE = re.compile(r"([A-Za-z][\w-]*)\((.*)\)")


def parse_eval_literal(literal: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Split ``on(a:block,b:block)`` into ``("on", [("a", "block"), ("b", "block")])``."""
    match = _LITERAL_RE.fullmatch(literal)
    if match is None:
        raise ValueError(f"Not an eval-schema literal: {literal!r}")
    name, body = match.groups()
    args = []
    for arg in filter(None, (a.strip() for a in body.split(","))):
        obj, _, typ = arg.partition(":")
        args.append((obj, typ))
    return name, args


@dataclass(frozen=True)
class GymSchema:
    """How a domain's eval-schema literals are spelled in PDDLGym.

    Attributes:
        name: The schema's name, recorded in ``generation_info.json``.
        to_gym: Maps one parsed eval literal to zero or one gym literal.
            Returning ``None`` drops the literal (static facts the renderer
            does not read).
    """

    name: str
    to_gym: Callable[[str, List[Tuple[str, str]]], Optional[GymLiteral]]

    def gym_literals(self, state: TraceState) -> List[Literal]:
        """The state's literals as gym ``Literal`` objects."""
        literals: List[Literal] = []
        for eval_literal in state.literals:
            converted = self.to_gym(*parse_eval_literal(eval_literal))
            if converted is None:
                continue
            name, args = converted
            predicate = Predicate(name, len(args), [Type(t) for _, t in args])
            literals.append(predicate(*[Type(t)(obj) for obj, t in args]))
        return literals


def _identity(name: str, args: List[Tuple[str, str]]) -> GymLiteral:
    return name, args


def _hanoi_to_gym(name: str, args: List[Tuple[str, str]]) -> GymLiteral:
    """``on-disc`` / ``on-peg`` -> ``on``; every object is untyped ``default``."""
    base = name.split("-", 1)[0]
    return base, [(obj, "default") for obj, _ in args]


_POSITION_RE = re.compile(r"p_(\d+)_(\d+)")


def _npuzzle_to_gym(name: str, args: List[Tuple[str, str]]) -> Optional[GymLiteral]:
    """``at(t_T, p_X_Y)`` -> ``at(tT, xX, yY)``; ``empty(p)`` -> ``blank(x, y)``."""
    if name == "at":
        tile, position = args[0][0], args[1][0]
        x, y = _POSITION_RE.fullmatch(position).groups()
        return "at", [(f"t{tile.split('_', 1)[1]}", "default"),
                      (f"x{x}", "default"), (f"y{y}", "default")]
    if name == "empty":
        x, y = _POSITION_RE.fullmatch(args[0][0]).groups()
        return "blank", [(f"x{x}", "default"), (f"y{y}", "default")]
    return None


BLOCKS_SCHEMA = GymSchema("blocks", _identity)
HANOI_SCHEMA = GymSchema("hanoi", _hanoi_to_gym)
NPUZZLE_SCHEMA = GymSchema("slidetile", _npuzzle_to_gym)
IDENTITY_SCHEMA = GymSchema("identity", _identity)


def to_uint8_rgba(frame: np.ndarray) -> np.ndarray:
    """PDDLGym's float or uint8 array as uint8, as the gym handler saved it."""
    if frame.dtype != np.uint8:
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
    return frame


class PDDLGymRenderer:
    """A :class:`FrameRenderer` over one of PDDLGym's ``render`` functions."""

    def __init__(
        self,
        render_fn: Callable[..., np.ndarray],
        schema: GymSchema,
        *,
        render_kwargs: Optional[Dict[str, Any]] = None,
        prime_colors: Optional[Callable[[Iterable[str]], None]] = None,
        name: str = "pddlgym",
    ) -> None:
        """Bind a gym render function to an eval->gym schema.

        Args:
            render_fn: ``pddlgym.rendering.<domain>.render``.
            schema: The domain's :class:`GymSchema`.
            render_kwargs: Extra keyword arguments for ``render_fn``.
            prime_colors: Called once with the sorted object names before the
                first render, for renderers that assign per-object colours in
                order of first sight.
            name: The renderer's name, recorded in ``generation_info.json``.
        """
        self._render_fn = render_fn
        self._schema = schema
        self._render_kwargs = render_kwargs if render_kwargs is not None else {"mode": "rgb_array"}
        self._prime_colors = prime_colors
        self._name = name
        self._primed = False

    def render(self, state: TraceState, *, step: int, action: Optional[str]) -> Image.Image:
        """Render the state's literals; ``step`` and ``action`` are ignored."""
        if not self._primed and self._prime_colors is not None:
            self._prime_colors(sorted(obj.split(":", 1)[0] for obj in state.objects))
            self._primed = True
        frame = self._render_fn(self._schema.gym_literals(state), **self._render_kwargs)
        plt.close("all")
        return Image.fromarray(to_uint8_rgba(frame))

    def frame_key(self, state: TraceState, *, step: int, action: Optional[str]) -> Hashable:
        """The state alone decides the image."""
        return state_key(state)

    def describe(self) -> Dict[str, Any]:
        """Renderer name and schema."""
        return {"name": self._name, "gym_schema": self._schema.name,
                "render_kwargs": dict(self._render_kwargs)}


def prime_block_colors(block_names: Sequence[str]) -> None:
    """Assign PDDLGym's block colours in a fixed order, independent of first sight."""
    from pddlgym.rendering.blocks import block_name_to_color

    for name in block_names:
        block_name_to_color(name)
