"""A trained ICAPS-26 ``Domain_Model`` as a PDDL domain in signature order.

The vendored ``extract_pddl`` (``dl/util/ROSAME/rosame.py:389``) writes
``:parameters`` and ``:predicates`` by iterating ``params_types``, which
``Predicate.__init__`` sorted by type name, and names the variables ``a, b,
c, ...`` in that order. The emitted file is internally consistent, and its
*signature* is a permutation of the one ``src/domains/<domain>.pddl`` declares
on four of our five domains (``docs/rosame-i-milp-26-implementation-plan.md``
§4.2b). All three AMLGym metrics bind arguments **positionally** —
``SimpleDomainReader`` renames parameters to ``?param_k`` by index,
``_syntactic`` intersects the resulting strings and ``_solving`` validates the
learned plan against the reference domain — so a semantically perfect model
scores ~0 on depot, gripper, hanoi and npuzzle.

:func:`emit_pddl` writes the same thresholded model with the source PDDL's own
parameter names, in the source PDDL's own order. It is the output end of the
bijection :mod:`src.milp.head_alignment` built the input end of, and it does not
touch the vendored file.

Two further defects of ``extract_pddl`` are not permutation-related and are also
corrected here, because both make the output unparseable rather than merely
mis-scored:

* variables are written bare in the precondition and effect bodies
  (``(at-truck a c)``), so no PDDL reader binds them to the parameters;
* an action with no precondition is written ``:precondition ()``, which
  ``pddl_plus_parser`` rejects; the empty conjunction is ``(and )``.

THRESHOLDING is upstream's, unchanged: ``argmax`` over the schema's four-way
head, where 1 is an add effect, 2 a precondition, and 3 a precondition with a
delete effect.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import torch

import src.milp  # noqa: F401  (vendor sys.path bootstrap)
from src.milp.domain_assets import build_domain
from src.milp.head_alignment import pddl_argument_order

from planning_structs.domain import Domain as PSDomain
from planning_structs.domain import Type as PSType
from planning_structs.util import collect_types_hierarchy

#: The four classes of the schema head, by ``argmax`` index.
PRECONDITION_CLASSES = (2, 3)
ADD_EFFECT_CLASS = 1
DELETE_EFFECT_CLASS = 3


def parameter_names(types: Sequence[PSType]) -> List[str]:
    """``["?x0", "?x1", ...]``, one per PDDL argument slot.

    Positional names rather than the source PDDL's own: ``SimpleDomainReader``
    renames every parameter to ``?param_k`` by index before scoring, so the name
    carries no information and a generated one cannot collide with a type name.
    """
    return [f"?x{i}" for i in range(len(types))]


def variables_by_type(
    types: Sequence[PSType], names: Sequence[str]
) -> Dict[str, List[str]]:
    """``{type name: [variable]}`` in PDDL order, the shape ``ground`` reads.

    Keys are type *names*; :func:`_head_var_map` re-keys them by the head's own
    ``Type`` objects, which are a different class from ``planning_structs``'.
    """
    by_type: Dict[str, List[str]] = {}
    for one_type, name in zip(types, names):
        by_type.setdefault(one_type.name, []).append(name)
    return by_type


def format_signature(types: Sequence[PSType], names: Sequence[str]) -> str:
    """``?x0 - crane ?x1 - package``, in PDDL signature order."""
    return " ".join(f"{name} - {one_type.name}" for name, one_type in zip(names, types))


def format_types(domain: PSDomain) -> str:
    """The ``:types`` block, as a child-per-parent hierarchy."""
    hierarchy: Dict[str, List[str]] = {}
    for one_type in collect_types_hierarchy(domain.types_set):
        if one_type.name == "object":
            continue
        parent = one_type.parent.name if one_type.parent else "object"
        hierarchy.setdefault(parent, []).append(one_type.name)
    return "\n        ".join(
        f"{' '.join(children)} - {parent}" for parent, children in hierarchy.items()
    )


def format_predicates(domain: PSDomain) -> str:
    """The ``:predicates`` block, each predicate in its PDDL signature order."""
    return "\n        ".join(
        f"({predicate.name} {format_signature(predicate.types, parameter_names(predicate.types))})".replace(
            " )", ")"
        )
        for predicate in domain.predicates
    )


def _conjunction(literals: Sequence[str]) -> str:
    """``(and ...)``, empty conjunction included.

    ``()`` is what upstream writes for an empty block and what
    ``pddl_plus_parser`` rejects with an ``IndexError``.
    """
    return "(and " + " ".join(literals) + ")" if literals else "(and )"


def _head_var_map(schema: object, by_type_name: Dict[str, List[str]]) -> Dict[object, List[str]]:
    """``{head Type: [variable]}`` for ``Predicate.ground``.

    Raises:
        KeyError: if the head carries a parameter type the PDDL signature does
            not, which means the two were built from different domain files.
    """
    var_map: Dict[object, List[str]] = {}
    for one_type in schema.params_types:
        if one_type.name not in by_type_name:
            raise KeyError(
                f"schema '{schema.name}' has a head parameter of type "
                f"'{one_type.name}', which its PDDL signature does not declare; "
                f"known types are {sorted(by_type_name)}"
            )
        var_map[one_type] = by_type_name[one_type.name]
    return var_map


def schema_literals(
    schema: object,
    by_type_name: Dict[str, List[str]],
    predicate_types: Mapping[str, Sequence[PSType]],
) -> Tuple[List[str], List[str]]:
    """One schema's ``(preconditions, effects)``, thresholded as upstream does.

    ``Predicate.ground`` is called unchanged, so the propositions stay in the
    enumeration order the schema head's rows are indexed by; each one's
    *arguments* are then put back into PDDL signature order by
    :func:`~src.milp.head_alignment.pddl_argument_order`.
    """
    var_map = _head_var_map(schema, by_type_name)
    propositions = [
        _in_pddl_order(proposition, predicate_types[predicate.name])
        for predicate in schema.predicates
        for proposition in predicate.ground(var_map)
    ]
    classes = torch.argmax(schema(), dim=1)
    if len(classes) != len(propositions):
        raise ValueError(
            f"schema '{schema.name}' has {len(classes)} head rows but grounds "
            f"{len(propositions)} propositions; the head and the domain PDDL "
            f"disagree on its relevant predicates"
        )

    preconditions: List[str] = []
    effects: List[str] = []
    for proposition, one_class in zip(propositions, classes.tolist()):
        literal = f"({proposition})"
        if one_class in PRECONDITION_CLASSES:
            preconditions.append(literal)
        if one_class == ADD_EFFECT_CLASS:
            effects.append(literal)
        elif one_class == DELETE_EFFECT_CLASS:
            effects.append(f"(not {literal})")
    return preconditions, effects


def _in_pddl_order(proposition: str, types: Sequence[PSType]) -> str:
    """One ``"pred a b"`` from ``ground``, with its arguments back in PDDL order.

    ``Predicate.ground`` emits arguments in the head's sorted-type order, so a
    literal written straight out binds the schema's variables to the wrong
    predicate slots even once ``:parameters`` itself is corrected.

    Raises:
        ValueError: if the proposition's arity does not match ``types``.
    """
    name, *args = proposition.split()
    if len(args) != len(types):
        raise ValueError(
            f"proposition '{proposition}' has {len(args)} arguments but "
            f"'{name}' takes {len(types)}"
        )
    return " ".join([name, *pddl_argument_order(args, types)])


def format_action(
    schema: object,
    types: Sequence[PSType],
    predicate_types: Mapping[str, Sequence[PSType]],
) -> str:
    """One ``(:action ...)`` block, with ``types`` the PDDL signature."""
    names = parameter_names(types)
    preconditions, effects = schema_literals(
        schema, variables_by_type(types, names), predicate_types
    )
    return (
        f"    (:action {schema.name}\n"
        f"        :parameters ({format_signature(types, names)})\n"
        f"        :precondition {_conjunction(preconditions)}\n"
        f"        :effect {_conjunction(effects)}\n"
        f"    )"
    )


class DegenerateModelError(RuntimeError):
    """A learned model with no add and no delete effect anywhere (gate 4).

    Such a model is not a weak model, it is an empty one: no action changes any
    state, so every plan of length > 0 is unsound and every solving score is 0.
    The ICAPS-24 arm reached exactly this on several cells and it went unnoticed
    for a whole grid, so the ICAPS-26 arm raises instead of reporting a row.
    """


def effect_counts(text: str) -> Dict[str, int]:
    """``{action: number of effect literals}`` for an emitted domain string.

    Counts syntactically, on the emitter's own one-line ``:effect`` form, so the
    guard does not depend on a PDDL parser accepting the model it is guarding.
    """
    counts: Dict[str, int] = {}
    name = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("(:action "):
            name = stripped[len("(:action ") :].strip()
        elif stripped.startswith(":effect") and name is not None:
            counts[name] = _top_level_literals(stripped[len(":effect") :].strip())
            name = None
    return counts


def _top_level_literals(conjunction: str) -> int:
    """How many literals an ``(and ...)`` block holds, negations counted once.

    Depth is tracked rather than parentheses counted, so ``(not (p ?x0))`` is
    one literal and an empty ``(and )`` is zero.
    """
    depth = 0
    literals = 0
    for character in conjunction:
        if character == "(":
            depth += 1
            if depth == 2:
                literals += 1
        elif character == ")":
            depth -= 1
    return literals


def check_not_degenerate(text: str) -> Dict[str, int]:
    """Raise unless some action of ``text`` has at least one effect (gate 4).

    Returns:
        The per-action effect counts, so a caller can log them.

    Raises:
        DegenerateModelError: if every action's effect block is empty.
    """
    counts = effect_counts(text)
    if not counts:
        raise DegenerateModelError(
            "the emitted domain carries no action at all; nothing was learned"
        )
    if not any(counts.values()):
        raise DegenerateModelError(
            f"every one of the {len(counts)} learned schemas has an empty effect "
            f"block, so no action changes any state; this is the empty-effects "
            f"collapse, not a weak model"
        )
    return counts


def emit_pddl(domain_model: object, bench: str) -> str:
    """The trained ``domain_model`` as a PDDL domain in ``bench``'s own order.

    Args:
        domain_model: A grounded, trained vendored ``Domain_Model``.
        bench: The domain key whose ``src/domains`` PDDL supplies the signatures.

    Raises:
        ValueError: if the head and the domain PDDL do not carry the same
            action schemas, which means the grounding assets are stale.
    """
    domain = build_domain(bench)
    signatures = {schema.name: schema.types for schema in domain.action_schemas}

    head_names = [schema.name for schema in domain_model.action_schemas]
    if sorted(head_names) != sorted(signatures):
        raise ValueError(
            f"the DL head carries schemas {sorted(head_names)} but '{bench}' "
            f"declares {sorted(signatures)}; regenerate the grounding assets"
        )

    predicate_types = {
        predicate.name: predicate.types for predicate in domain.predicates
    }
    actions = "\n\n".join(
        format_action(schema, signatures[schema.name], predicate_types)
        for schema in domain_model.action_schemas
    )
    return (
        f"(define (domain {bench})\n"
        f"    (:requirements :strips :typing)\n"
        f"    (:types\n        {format_types(domain)})\n"
        f"    (:predicates\n        {format_predicates(domain)})\n\n"
        f"{actions}\n)"
    )
