"""Bridge between the trained ROSAME model and the MILP world.

Three responsibilities:
  1. ROSAME -> ObservationM: read each schema's 4-way distribution
     (``forward()`` rows) into pre/add/del probabilities for the MILP objective.
  2. MILP solution -> pseudo-labels: 4-way one-hot targets per ``forward()``
     row (the model-CE supervision channel of the training loop), plus an
     agreement metric.
  3. MILP solution -> PDDL: decode the binary pre/add/del variables into a PDDL
     domain string, formatted with the runner's own helpers so ROSAME and
     ROSAME+MILP models are byte-comparable.

Row <-> binding correspondence: each schema's ``forward()`` output has one row
per (relevant predicate, variable grounding), enumerated exactly as
``Action_Schema.pretty_print`` does — ``schema.predicates`` in order, each
expanded by ``predicate.ground(var)`` with the PDDL variable names from
``runner.get_params_names``. We rebuild that enumeration and map every row to
the planning_structs key ``(schema, predicate, x)`` where ``x`` are 1-based
PDDL parameter positions (the converter builds the vendored Domain in PDDL
signature order, so positions line up by construction).

4-way semantics (from ``Action_Schema.pretty_print`` / upstream translator):
index 0 = irrelevant, 1 = add effect, 2 = precondition, 3 = precondition+delete.
Hence pre = P[2]+P[3], add = P[1], del = P[3]. Label precedence when decoding a
binary solution mirrors upstream ``translator.extract_sol_model``:
add > del > pre > irrelevant.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from planning_structs.domain import Domain as PSDomain
from planning_structs.traces import ObservationM

BindingRow = Tuple[str, Tuple[int, ...]]  # (predicate_name, x)


# ---------------------------------------------------------------- row tables

def _signature_positions(runner, schema_name: str) -> Dict[str, int]:
    """PDDL variable name (without '?') -> 1-based position in the signature."""
    signature = runner.domain.actions[schema_name].signature
    return {param_name[1:]: idx + 1 for idx, param_name in enumerate(signature.keys())}

def binding_table(runner, schema) -> List[BindingRow]:
    """Ordered (predicate_name, x) per ``forward()`` row of one ROSAME schema.

    Replicates ``Action_Schema.pretty_print``'s enumeration with the PDDL
    variable names, then converts each variable tuple to signature positions.
    """
    params = runner.get_params_names(runner.domain.actions[schema.name])
    var = {params_type: params[params_type.name] for params_type in schema.params_types}
    positions = _signature_positions(runner, schema.name)

    rows: List[BindingRow] = []
    for predicate in schema.predicates:
        for prop_str in predicate.ground(var):
            parts = prop_str.split()
            pred_name, var_names = parts[0], parts[1:]
            x = tuple(positions[v] for v in var_names)
            rows.append((pred_name, x))
    return rows


def _ps_key(ps_domain: PSDomain, schema_name: str, row: BindingRow):
    a = ps_domain.get_action_schema(schema_name)
    p = ps_domain.get_predicate(row[0])
    return (a, p, row[1])


# ------------------------------------------------------- ROSAME -> ObservationM

def rosame_to_observation_m(runner, ps_domain: PSDomain) -> ObservationM:
    """Current ROSAME probabilities as the MILP's model prior.

    Structurally irrelevant (schema, predicate, binding) triples — those ROSAME
    excludes from a schema's feature rows — get probability 0.0 (confident
    "not in the model"), matching ROSAME's own semantics.
    """
    pre: Dict = {}
    add: Dict = {}
    dele: Dict = {}
    for a in ps_domain.action_schemas:
        for p in ps_domain.predicates:
            for x in ps_domain.predicate_arguments[(a, p)]:
                pre[(a, p, x)] = 0.0
                add[(a, p, x)] = 0.0
                dele[(a, p, x)] = 0.0

    for schema in runner.rosame.action_schemas:
        probs = schema().detach().cpu()
        rows = binding_table(runner, schema)
        if probs.shape[0] != len(rows):
            raise ValueError(
                f"Row-count mismatch for schema '{schema.name}': "
                f"forward()={probs.shape[0]} vs binding table={len(rows)}"
            )
        for i, row in enumerate(rows):
            key = _ps_key(ps_domain, schema.name, row)
            if key[0] is None or key[1] is None or key not in pre:
                raise ValueError(f"Unmapped binding row {row} for schema '{schema.name}'")
            p_add = float(probs[i, 1])
            p_pre = float(probs[i, 2]) + float(probs[i, 3])
            p_del = float(probs[i, 3])
            pre[key] = p_pre
            add[key] = p_add
            dele[key] = p_del

    return ObservationM(pre, add, dele)


# ------------------------------------------- MILP solution -> labels/agreement

def extract_model_labels(runner, ps_domain: PSDomain, sol: ObservationM) -> Dict[str, "object"]:
    """4-way one-hot pseudo-labels per schema from a solved MILP.

    Returns ``{schema_name: FloatTensor(n_rows, 4)}`` aligned with
    ``forward()`` rows. Precedence mirrors upstream ``extract_sol_model``:
    add > del > pre > irrelevant.
    """
    import torch  # local import: bridge stays importable without torch for tests

    labels: Dict[str, torch.Tensor] = {}
    for schema in runner.rosame.action_schemas:
        rows = binding_table(runner, schema)
        label = torch.zeros(len(rows), 4)
        for i, row in enumerate(rows):
            key = _ps_key(ps_domain, schema.name, row)
            is_pre = bool(sol.pre.get(key, 0))
            is_add = bool(sol.add.get(key, 0))
            is_del = bool(sol.dele.get(key, 0))
            if is_add:
                label[i, 1] = 1
            elif is_del:
                label[i, 3] = 1
            elif is_pre:
                label[i, 2] = 1
            else:
                label[i, 0] = 1
        labels[schema.name] = label
    return labels


def model_cross_entropy(action_schemas, labels: Dict[str, "object"]):
    """CE of each schema's 4-way distribution vs its pseudo-labels, or ``None``.

    Summed over all rows of all schemas and normalized by the total row count,
    undecayed — upstream ``loss_pseudo_m`` (``dl/model.py``). Upstream feeds
    activated outputs into ``F.cross_entropy``, which expects logits; AMLGym's
    ``forward()`` ends in Softmax, so the cross-entropy is applied directly to
    the probabilities.
    """
    import torch

    total_rows = 0
    ce = None
    for schema in action_schemas:
        target = labels.get(schema.name) if labels else None
        if target is None:
            continue
        probs = schema()
        term = -(target * torch.log(probs + 1e-9)).sum()
        ce = term if ce is None else ce + term
        total_rows += probs.shape[0]
    if ce is None or total_rows == 0:
        return None
    return ce / total_rows


def model_agreement(runner, labels: Dict[str, "object"]) -> float:
    """Fraction of rows where argmax(ROSAME 4-way) == argmax(pseudo-label)."""
    import torch

    agree, total = 0, 0
    for schema in runner.rosame.action_schemas:
        if schema.name not in labels:
            continue
        probs = schema().detach().cpu()
        target = labels[schema.name]
        agree += int((torch.argmax(probs, dim=1) == torch.argmax(target, dim=1)).sum())
        total += probs.shape[0]
    return agree / total if total else 1.0


# ------------------------------------------------- MILP solution -> PDDL string

def solution_to_pddl(runner, ps_domain: PSDomain, sol: ObservationM) -> str:
    """Decode the MILP's binary model into a PDDL domain string.

    Mirrors ``Rosame_Runner.rosame_to_pddl`` / ``to_pddl_action`` formatting
    (same helpers, same variable names) so the output is directly comparable
    with the plain ROSAME baseline's model.
    """
    actions = []
    for schema in runner.rosame.action_schemas:
        rows = binding_table(runner, schema)
        params = runner.get_params_names(runner.domain.actions[schema.name])
        var = {pt: params[pt.name] for pt in schema.params_types}

        # ordered proposition strings, aligned with rows
        prop_strs = [p for predicate in schema.predicates for p in predicate.ground(var)]

        precon_list, addeff_list, deleff_list = [], [], []
        for row, prop_str in zip(rows, prop_strs):
            key = _ps_key(ps_domain, schema.name, row)
            if sol.add.get(key, 0):
                addeff_list.append(prop_str)
            if sol.dele.get(key, 0):
                deleff_list.append(prop_str)
            if sol.pre.get(key, 0):
                precon_list.append(prop_str)

        action_string = (
            f"(:action {schema.name}\n"
            f"\t:parameters {runner._signature_to_pddl(schema)}\n"
            f"\t:precondition {runner.precondition_pddl(precon_list)}\n"
            f"\t:effect {runner.effects_pddl(addeff_list, deleff_list)}"
        )
        formatted = "\n".join(line for line in action_string.split("\n") if line.strip())
        actions.append(f"{formatted}\n")

    domain = runner.domain
    predicates = "\n\t".join(str(p) for p in domain.predicates.values())
    predicates_str = f"(:predicates {predicates}\n)\n\n" if domain.predicates else ""
    types_str = f"(:types {runner._types_to_pddl(domain.types)}\n)\n\n" if domain.types else ""
    constants_str = (
        f"(:constants {runner._constants_to_pddl(domain)}\n)\n\n" if domain.constants else ""
    )

    return (
        f"(define (domain {domain.name})\n"
        f"(:requirements {' '.join(domain.requirements)})\n"
        f"{types_str}"
        f"{constants_str}"
        f"{predicates_str}"
        + "\n".join(actions)
        + "\n)"
    )
