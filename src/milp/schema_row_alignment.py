"""Row alignment between an ICAPS-26 schema head and the CP action model.

A schema's ``forward()`` emits one row per (relevant predicate, argument
binding). ``planning_structs`` enumerates the same set as
``domain.predicate_arguments[(schema, predicate)]``. The two agree on the *set*
and on its *size*, and disagree on the *order* — so a positional zip decodes the
wrong rows and raises nothing.

Three independent orderings differ, and all three must be undone:

1. **The schema's arguments.** ``Action_Schema.__init__`` sorts its parameter
   types by name, so slot 2 of ``pick(ball, room, gripper)`` is *gripper* to the
   head and *room* to the CP domain. Undone by keying on the PDDL variable a
   slot holds rather than on the slot number — the same move
   ``benchmark/algorithm_adapters/rosame_milp/model_bridge.py:binding_table``
   makes for the ICAPS-24 arm.
2. **Each predicate's own arguments.** ``Predicate.__init__`` sorts too, so
   ``at(tile, position)`` grounds as ``at(?position, ?tile)``. Undone by
   :func:`~src.milp.head_alignment.pddl_argument_order`. **This is the layer the
   24 arm's bridge does not have**, and does not need: AMLGym's fork carries the
   sort commented out (``models/rosame.py:31``), so its heads are already in
   PDDL order.
3. **Row order.** ``clear(?x - object)`` matches all four of depot's parameters,
   so it contributes four rows that the two sides enumerate in their own slot
   orders. Undone by keying rather than zipping: with a key, order is irrelevant.

WHY UPSTREAM HAS NO SUCH MODULE. Every action schema and predicate in upstream's
five domains is *already* written in sorted-type order, so their sort is a no-op
on their whole corpus and the mismatch cannot arise. Our ``src/domains/*.pddl``
follow IPC convention instead, and reorder on four of five domains.
``model_permutation`` cannot substitute: its ``type_match``
(``util/model_perm.py:20-24``) admits only permutations between *same-typed*
positions — it resolves the semantic symmetry the paper describes, not a
representational mismatch — and every permutation we need crosses type
boundaries, so not one is in its search space.

Verified bijective on all five benchmark domains and independent of the object
universe: ``predicate_arguments`` is keyed on parameter *positions*, so a
domain's alignment is fixed once and cannot drift with fold composition.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import src.milp  # noqa: F401  (vendor sys.path bootstrap)
from src.milp.head_alignment import pddl_argument_order

from planning_structs.domain import ActionSchema as PSActionSchema
from planning_structs.domain import Domain as PSDomain

#: ``(predicate name, 1-based PDDL parameter positions)`` — the key both sides share.
BindingRow = Tuple[str, Tuple[int, ...]]


def parameter_variables(schema: PSActionSchema) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """Placeholder variable names for one schema's PDDL signature.

    Returns:
        ``({type name: [variable]}, {variable: 1-based PDDL position})``. The
        names are positional placeholders; only the *mapping back to a position*
        matters, so they need not match the domain file's own parameter names.
    """
    by_type: Dict[str, List[str]] = {}
    position: Dict[str, int] = {}
    for index, one_type in enumerate(schema.types):
        name = f"v{index + 1}"
        position[name] = index + 1
        by_type.setdefault(one_type.name, []).append(name)
    return by_type, position


def head_binding_rows(head_schema: object, ps_domain: PSDomain) -> List[BindingRow]:
    """The head's ``forward()`` rows, each as a CP-side key, in head row order.

    ``out[i]`` is the key of row ``i`` of ``head_schema()``.

    Raises:
        KeyError: if the head carries a predicate or parameter type the CP domain
            does not, which means the two were built from different assets.
    """
    ps_schema = ps_domain.get_action_schema(head_schema.name)
    if ps_schema is None:
        raise KeyError(
            f"the CP domain has no action schema '{head_schema.name}'; it "
            f"carries {[a.name for a in ps_domain.action_schemas]}"
        )

    by_type, position = parameter_variables(ps_schema)
    variables = {}
    for one_type in head_schema.params_types:
        if one_type.name not in by_type:
            raise KeyError(
                f"schema '{head_schema.name}' has a head parameter of type "
                f"'{one_type.name}', which its CP signature does not declare; "
                f"declared types are {sorted(by_type)}"
            )
        variables[one_type] = by_type[one_type.name]

    rows: List[BindingRow] = []
    for predicate in head_schema.predicates:
        ps_predicate = ps_domain.get_predicate(predicate.name)
        if ps_predicate is None:
            raise KeyError(
                f"the CP domain has no predicate '{predicate.name}', which "
                f"schema '{head_schema.name}' grounds over"
            )
        for proposition in predicate.ground(variables):
            head_args = proposition.split()[1:]
            ordered = pddl_argument_order(head_args, ps_predicate.types)
            rows.append((predicate.name, tuple(position[name] for name in ordered)))
    return rows


def check_bijective(
    rows: Sequence[BindingRow], head_schema: object, ps_domain: PSDomain
) -> None:
    """Raise unless ``rows`` is a bijection onto the CP domain's own bindings.

    A silent mis-alignment trains every model pseudo-label against the wrong row
    while the loss falls normally, so this is checked rather than assumed.

    Raises:
        ValueError: if a row repeats, if the row count does not match the head's
            tensor width, or if the rows do not cover the CP bindings exactly.
    """
    if len(set(rows)) != len(rows):
        repeated = sorted({row for row in rows if rows.count(row) > 1})
        raise ValueError(
            f"schema '{head_schema.name}' maps two head rows to {repeated}; "
            f"the alignment is not injective"
        )

    width = int(head_schema.randn.shape[0])
    if len(rows) != width:
        raise ValueError(
            f"schema '{head_schema.name}' grounds {len(rows)} rows but its head "
            f"emits {width}; the head and the CP domain disagree on its "
            f"relevant predicates"
        )

    ps_schema = ps_domain.get_action_schema(head_schema.name)
    expected = {
        (predicate.name, tuple(binding))
        for predicate in head_schema.predicates
        for binding in ps_domain.predicate_arguments[
            (ps_schema, ps_domain.get_predicate(predicate.name))
        ]
    }
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        extra = sorted(set(rows) - expected)
        raise ValueError(
            f"schema '{head_schema.name}' does not cover the CP bindings: "
            f"{len(missing)} missing (e.g. {missing[:3]}), "
            f"{len(extra)} unexpected (e.g. {extra[:3]})"
        )


def schema_row_keys(
    domain_model: object, ps_domain: PSDomain
) -> Dict[str, List[BindingRow]]:
    """``{schema name: rows}`` for every schema of a grounded ``Domain_Model``.

    Every schema is checked bijective before being returned.

    Raises:
        ValueError: via :func:`check_bijective`.
        KeyError: via :func:`head_binding_rows`.
    """
    keys: Dict[str, List[BindingRow]] = {}
    for head_schema in domain_model.action_schemas:
        rows = head_binding_rows(head_schema, ps_domain)
        check_bijective(rows, head_schema, ps_domain)
        keys[head_schema.name] = rows
    return keys
