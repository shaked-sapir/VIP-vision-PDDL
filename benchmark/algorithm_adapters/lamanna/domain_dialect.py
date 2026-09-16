"""The domain spelling the Lamanna learners' regex parsers read correctly.

Their ``ActionModel.read_predicates`` takes a predicate's arity from its
typed parameters, so an untyped parameter such as depot's ``(clear ?x)``
becomes a zero-arity ``(clear)`` in the learned model and every plan fails.
The copy handed to the learners types every untyped predicate and operator
parameter as ``object`` and declares the flat type list under ``object``, so
the parsers see the root the way PDDL defines it. Evaluation still runs
against the untouched reference domain.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

ROOT_TYPE = "object"

_PREDICATES_OPEN = "(:predicates"
_DECLARATION = re.compile(r"\(([\w-]+)((?:\s+\?[\w-]+(?:\s+-\s+[\w-]+)?)*)\s*\)")
_PARAMETERS = re.compile(r"(:parameters\s*\()([^)]*)(\))")
_TYPES_BLOCK = re.compile(r"\(:types\s+([^)]*)\)", re.S)


def _type_parameter_list(params: str) -> str:
    """``params`` with a trailing run of untyped variables typed ``object``.

    PDDL lets several variables share one type (``?from ?to - position``), so
    a variable is untyped only when no ``- type`` follows its run.
    """
    tokens = params.split()
    if not tokens or tokens[-1] == "-" or "-" in tokens and tokens[-2] == "-":
        return params
    if not tokens[-1].startswith("?"):
        return params
    return f"{params.rstrip()} - {ROOT_TYPE}"


def _balanced_span(text: str, start: int) -> Tuple[int, int]:
    """``(start, end)`` of the parenthesised block opening at ``text[start]``."""
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return start, index + 1
    raise ValueError("unbalanced parentheses in domain text")


def _type_predicates(text: str) -> str:
    start = text.find(_PREDICATES_OPEN)
    if start < 0:
        return text
    start, end = _balanced_span(text, start)
    block = _DECLARATION.sub(
        lambda d: f"({d.group(1)}{_type_parameter_list(d.group(2))})", text[start:end]
    )
    return text[:start] + block + text[end:]


def _root_types_block(block: re.Match) -> str:
    """A flat ``(:types a b c)`` becomes ``(:types a b c - object)``; a hierarchy is kept."""
    body = block.group(1)
    if " - " in body:
        return block.group(0)
    return f"(:types {' '.join(body.split())} - {ROOT_TYPE})"


def typed_domain_text(text: str) -> str:
    """``text`` with untyped parameters typed as ``object`` and ``object`` declared as root.

    A domain with no untyped parameter is returned unchanged.
    """
    typed = _type_predicates(text)
    typed = _PARAMETERS.sub(
        lambda m: m.group(1) + _type_parameter_list(m.group(2)) + m.group(3), typed
    )
    if typed == text:
        return text
    return _TYPES_BLOCK.sub(_root_types_block, typed)


def write_typed_domain(domain_path: Path, out_path: Path) -> Path:
    """Write the typed copy of ``domain_path`` at ``out_path`` and return it."""
    out_path = Path(out_path)
    out_path.write_text(typed_domain_text(Path(domain_path).read_text()))
    return out_path
