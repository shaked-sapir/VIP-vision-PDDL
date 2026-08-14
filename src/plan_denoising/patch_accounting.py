"""How many edits a set of fluent patches actually makes.

``FluentLevelPatch`` is frozen on the *raw* fluent string, so ``(on a b)`` and
``(not (on a b))`` at the same ``(observation_index, component_index,
state_type)`` are two distinct members of a ``set``. They are not two distinct
edits. :func:`src.utils.pddl_state.flip_fluent_in_state` matches a patch against
``{fluent, negate(fluent)}`` and *toggles* whatever it finds, so the polarity
written in the patch string never reaches the state — both records issue the
identical instruction "toggle on-a-b here". Two toggles of one predicate are the
identity.

So the realized edit count is the number of keys touched an **odd** number of
times, not the number of records and not the number of distinct keys (which
would charge a cancelling pair 1 instead of 0).

``ConflictDrivenPatchSearchBase._dedup_patches`` already applies exactly this
reasoning to the *other* way one state can be named twice — ``next`` of
component *i* and ``prev`` of component *i+1* are one object, so a patch on each
cancels — and it is deliberately polarity-blind when it does. This module is
that same rule for the case where the two records share a key outright.

This is reporting only. ``_compute_cost`` still counts raw set members, because
that count is the search's ``g`` and changing it would reorder the frontier and
invalidate every result already on disk. The functions here exist so that a
number *compared against another algorithm* — above all the design §7.1 bound
``cost(MILP) <= cost(best CDPS CFM)`` — can be like-for-like.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Tuple

from src.plan_denoising.typings import FluentLevelPatch

# (observation_index, component_index, state_type, normalized_fluent).
# Indices are stringified so that patches and their serialized records — which
# survive a JSON round trip as either ints or strings — land on the same key.
PatchKey = Tuple[str, str, str, str]


def _normalize_fluent(fluent: str) -> str:
    """Canonical positive form: strips a ``(not ...)`` wrapper if present."""
    if fluent.startswith("(not ") and fluent.endswith(")"):
        return fluent[5:-1]
    return fluent


def patch_key(patch: FluentLevelPatch) -> PatchKey:
    """Identity of the edit a patch performs, ignoring the polarity it names."""
    return (
        str(patch.observation_index),
        str(patch.component_index),
        patch.state_type,
        patch.normalized_fluent,
    )


def record_key(record: Mapping[str, Any]) -> PatchKey:
    """:func:`patch_key` for a serialized patch from a solutions log."""
    return (
        str(record["observation_index"]),
        str(record["component_index"]),
        str(record["state_type"]),
        _normalize_fluent(str(record["fluent"])),
    )


def _odd_parity_count(keys: Iterable[PatchKey]) -> int:
    """Number of keys appearing an odd number of times."""
    return sum(1 for count in Counter(keys).values() if count % 2 == 1)


def net_patch_count(patches: Iterable[FluentLevelPatch]) -> int:
    """Edits a patch set actually performs, after self-cancelling pairs drop out."""
    return _odd_parity_count(patch_key(p) for p in patches)


def net_patch_count_from_records(records: Iterable[Mapping[str, Any]]) -> int:
    """:func:`net_patch_count` for serialized patches, e.g. from an existing log."""
    return _odd_parity_count(record_key(r) for r in records)
