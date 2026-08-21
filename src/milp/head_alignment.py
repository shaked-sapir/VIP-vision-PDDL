"""Index alignment between the CP grounding and the ICAPS-26 DL head.

Two independent groundings of the same domain over the same objects, in two
different argument orders:

``planning_structs.Instance`` walks ``permutations(objects, arity)`` per
predicate and keeps the type-matching tuples, so its arguments stay in **PDDL
signature order** and it stringifies as ``pred_obj1_obj2``.

``dl.util.ROSAME.rosame.Predicate`` stores a signature as a ``{Type: count}``
map and grounds one ``itertools.permutations`` group per key of
``sorted(params.keys(), key=lambda x: x.name)``, so its arguments are in
**sorted-type order** and it stringifies as ``"pred obj1 obj2"``.

:func:`head_key` bridges the two through
:func:`~src.milp.domain_assets.rosame_argument_permutation`, and
:func:`proposition_head_indices` / :func:`action_head_indices` lift that to a
whole grounding. Both raise rather than skip: an alignment that is not a
bijection is a defect in the grounding assets, and downstream cannot detect it.
``constraint_opt``'s ``trans_full_state`` zips a predicted vector against the CP
proposition list positionally, so a permuted vector of the right width decodes
the wrong propositions and raises nothing.

Measured on our five domains with two objects per leaf type, aligning by
predicate name alone and taking the head's argument order at face value:

    domain       propositions            actions
    blocksworld  9/9 correct             8/8 correct
    depot        30/42 correct, 12 miss  0/84 correct, 84 miss
    gripper      12/12 correct           2/18 correct, 16 miss
    hanoi        12/16 correct, 4 miss   8/12 correct, 4 miss
    npuzzle      4/8 correct, 4 miss     0/4 correct, 4 miss

blocksworld's permutation is the identity, which is why every other gate in this
package being a blocksworld gate proves nothing here.

Every one of those failures is a *miss* rather than a wrong hit, on the
synthetic universe and on the real depot and blocksworld object unions alike:
our five domains declare disjoint types, so a mis-ordered argument tuple is not
type-valid and the head has no such key. That is a property of these domains,
not of the scheme, and it is not a licence to align by name — a wrong tuple that
happened to be type-valid would decode silently, and even a miss only reaches
``cv_predictions_to_trace``'s ``unmapped`` counter, which warns and defaults the
column to 0.5.

The ICAPS-24 arms are unaffected and need no permutation step; AMLGym's fork
carries the sort commented out and recovers PDDL order, pinned by
``benchmark/algorithm_adapters/test_check_predicate.py``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple, TypeVar

import src.milp  # noqa: F401  (vendor sys.path bootstrap)
from src.milp.domain_assets import rosame_argument_permutation

from planning_structs.domain import Type as PSType
from planning_structs.instance import Instance as PSInstance

_Item = TypeVar("_Item")


def head_key(name: str, args: Sequence[str], types: Sequence[PSType]) -> str:
    """The DL head's key for ``name`` applied to ``args`` in PDDL signature order."""
    if len(args) != len(types):
        raise ValueError(
            f"'{name}' takes {len(types)} arguments, got {len(args)}: {list(args)}"
        )
    order = rosame_argument_permutation(types)
    return " ".join([name, *(args[i] for i in order)]).strip()


def pddl_argument_order(
    head_args: Sequence[str], types: Sequence[PSType]
) -> List[str]:
    """``head_args``, given in the DL head's order, back in PDDL signature order."""
    if len(head_args) != len(types):
        raise ValueError(
            f"signature takes {len(types)} arguments, got {len(head_args)}: "
            f"{list(head_args)}"
        )
    args: List[str] = [""] * len(head_args)
    for slot, source in enumerate(rosame_argument_permutation(types)):
        args[source] = head_args[slot]
    return args


def proposition_head_indices(
    instance: PSInstance, domain_model: object
) -> List[int]:
    """``out[i]`` is the head column of ``instance.propositions[i]``."""
    return _align(
        instance.propositions,
        getattr(domain_model, "propositions"),
        lambda p: (p.predicate.name, p.predicate.types),
        "proposition",
    )


def action_head_indices(instance: PSInstance, domain_model: object) -> List[int]:
    """``out[i]`` is the head's action index for ``instance.actions[i]``."""
    return _align(
        instance.actions,
        getattr(domain_model, "actions"),
        lambda a: (a.action_schema.name, a.action_schema.types),
        "action",
    )


def invert(indices: Sequence[int]) -> List[int]:
    """The inverse map: ``out[head_index]`` is the CP index that claimed it."""
    inverse: List[int] = [-1] * len(indices)
    for source, target in enumerate(indices):
        if not 0 <= target < len(indices):
            raise ValueError(f"index {target} is outside 0..{len(indices) - 1}")
        if inverse[target] != -1:
            raise ValueError(f"head index {target} is claimed twice; not a bijection")
        inverse[target] = source
    return inverse


def _align(
    cp_items: Sequence[_Item],
    head_index: Dict[str, int],
    signature_of: Callable[[_Item], Tuple[str, Sequence[PSType]]],
    kind: str,
) -> List[int]:
    """CP index to head index, raising unless the two groundings coincide."""
    if len(cp_items) != len(head_index):
        raise ValueError(
            f"the CP grounding has {len(cp_items)} {kind}s and the DL head has "
            f"{len(head_index)}; both must be built over the same object union"
        )

    aligned: List[int] = []
    for item in cp_items:
        name, types = signature_of(item)
        key = head_key(name, [obj.name for obj in item.args], types)
        if key not in head_index:
            raise KeyError(
                f"CP {kind} '{item}' maps to head key {key!r}, which the DL head "
                f"does not carry; its permutation is "
                f"{rosame_argument_permutation(types)} over {[t.name for t in types]}"
            )
        aligned.append(head_index[key])

    if len(set(aligned)) != len(aligned):
        raise ValueError(f"two CP {kind}s map to the same head index")
    return aligned
