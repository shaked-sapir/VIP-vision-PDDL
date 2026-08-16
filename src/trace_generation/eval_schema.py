"""How eval-schema strings are spelled, shared by every bridge that writes them.

    literal  ``"on(a:block,b:block)"``      nullary: ``"handempty()"``
    object   ``"a:block"``
    action   ``"stack(a:block, b:block)"``  note the separator differs

``src.utils.pddl_gym.parse_gym_to_pddl_literal`` and
``parse_gym_to_pddl_ground_action`` read these back into the untyped PDDL forms.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

LITERAL_ARG_SEPARATOR = ","
ACTION_ARG_SEPARATOR = ", "


def typed_object(object_name: str, types: Dict[str, str]) -> str:
    """Attach an object's declared type, failing loudly on an unknown object."""
    if object_name not in types:
        raise KeyError(
            f"Object {object_name!r} is not declared in the problem's :objects "
            "or the domain's :constants; its type cannot be written into the "
            "eval schema."
        )
    return f"{object_name}:{types[object_name]}"


def typed_universe(types: Dict[str, str]) -> Tuple[str, ...]:
    """The object universe as sorted ``name:type`` strings."""
    return tuple(sorted(f"{name}:{type_name}" for name, type_name in types.items()))


def render_literal(name: str, arguments: Sequence[str],
                   types: Dict[str, str]) -> str:
    """Render a ground predicate as an eval-schema literal."""
    return _render(name, arguments, types, LITERAL_ARG_SEPARATOR)


def render_action(name: str, arguments: Sequence[str],
                  types: Dict[str, str]) -> str:
    """Render a ground action as an eval-schema action string."""
    return _render(name, arguments, types, ACTION_ARG_SEPARATOR)


def _render(name: str, arguments: Sequence[str], types: Dict[str, str],
            separator: str) -> str:
    """Join typed arguments onto a name; nullary renders as empty parens."""
    if not arguments:
        return f"{name}()"
    return f"{name}({separator.join(typed_object(a, types) for a in arguments)})"
