"""Reconciliation of the two spellings the same PDDL symbol reaches us in.

``benchmark/experiment_running_helpers/normalize.py`` rewrites hyphens to
underscores in a corpus's identifiers, so a domain that ran through it spells
``at-truck`` as ``at_truck`` while ``src/domains`` keeps the hyphen. The rewrite
is per-corpus and not always total: hanoi's reference domain carries underscored
action names beside hyphenated predicate names, so no single transform maps one
dialect onto the other.

A symbol is therefore matched on :func:`canonical` and rewritten to whatever the
target domain declares.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping


def canonical(name: str) -> str:
    """A name in the underscored dialect with its whitespace normalised."""
    return " ".join(name.replace("-", "_").split())


def spelling_map(names: Iterable[str]) -> Mapping[str, str]:
    """``{canonical form: declared spelling}`` for ``names``.

    Raises:
        ValueError: if two names share a canonical form, which makes the
            rewrite ambiguous.
    """
    mapping: dict[str, str] = {}
    for name in names:
        key = canonical(name)
        if key in mapping and mapping[key] != name:
            raise ValueError(
                f"'{mapping[key]}' and '{name}' share the canonical form "
                f"'{key}', so a spelling cannot be chosen between them"
            )
        mapping[key] = name
    return mapping


def rewrite_symbols(text: str, spellings: Mapping[str, str]) -> str:
    """``text`` with every known symbol written as ``spellings`` declares it.

    Only tokens in head position — the identifier after a ``(`` — are rewritten,
    which leaves ``:requirements`` keywords and ``?`` variables untouched. A
    token absent from ``spellings`` is left alone.
    """

    def _rewrite(match: re.Match) -> str:
        prefix, name = match.group(1), match.group(2)
        return prefix + spellings.get(canonical(name), name)

    return re.sub(r"(\(\s*)([A-Za-z][A-Za-z0-9_-]*)", _rewrite, text)
