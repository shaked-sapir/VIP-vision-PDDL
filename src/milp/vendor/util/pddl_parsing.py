from tarski.syntax import CompoundFormula
from tarski.fstrips.fstrips import AddEffect, DelEffect

from planning_structs.traces import ObservationM

# VENDOR MODIFICATION (see ../UPSTREAM.md): upstream imports `Parser` at module
# scope. `lifted_pddl` is not a project dependency, and this module is on the
# import path of every consumer of `dl/network.py` (via `convertor.convertor`),
# so an unguarded import makes the whole vendored DL tree unimportable.
# `Parser` is reachable only from `parse_pddl_domain`, which raises below.
try:
    from lifted_pddl import Parser
except ImportError:  # lifted_pddl not installed — `parse_pddl_domain` is unavailable
    Parser = None


def parse_pddl_domain(domain, domain_fname) -> ObservationM:
    """
    Parse a PDDL domain (and optional problem) into a UP Problem,
    and pull out for each action schema its parameters, preconditions, and effects.
    """

    def _extract_predicates(formula):
        """
        Break a tarski formula like (and (p x) (q y)) into ["(p x)", "(q y)"].
        Handles non-conjunction atomic formulas too.
        """
        if formula is None:
            return []

        # Conjunction of formulas (CompoundFormula with connective AND)
        if isinstance(formula, CompoundFormula) and formula.connective.name == "And":
            return [str(arg) for arg in formula.subformulas]

        # Single atomic predicate
        return [str(formula)]

    if Parser is None:
        raise ImportError(
            "parse_pddl_domain requires the `lifted_pddl` package, which is not "
            "installed. It is reachable only from Convertor.__init__, which builds "
            "the `gt_am` reference model for the model_permutation diagnostic."
        )

    parser = Parser()
    parser.parse_domain(str(domain_fname))
    problem = parser._reader.problem
    schema_info = {}

    for name, action in problem.actions.items():
        args = [v.symbol for v in action.parameters.vars()]

        # Split preconditions
        preconditions = _extract_predicates(action.precondition)

        # Split effects
        add_effects = []
        delete_effects = []
        for eff in action.effects:
            if isinstance(eff, AddEffect):
                add_effects.append(str(eff.atom))
            elif isinstance(eff, DelEffect):
                delete_effects.append(str(eff.atom))

        schema_info[name] = {
            "args": args,
            "preconditions": preconditions,
            "add effects": add_effects,
            "delete effects": delete_effects,
        }

    parsed_am = ObservationM({}, {}, {})
    for a in domain.action_schemas:
        for p in domain.predicates:
            for x in domain.predicate_arguments[(a, p)]:
                # VENDOR MODIFICATION (see ../UPSTREAM.md): upstream reuses the
                # outer double quotes inside the f-string, which only parses under
                # PEP 701 (Python >= 3.12). Rewritten with single quotes; the
                # produced string is identical.
                p_string = f"{p.name}({','.join([schema_info[a.name]['args'][i - 1] for i in x])})"
                if p_string in schema_info[a.name]["preconditions"]:
                    parsed_am.pre[(a, p, x)] = 1
                else:
                    parsed_am.pre[(a, p, x)] = 0
                if p_string in schema_info[a.name]["add effects"]:
                    parsed_am.add[(a, p, x)] = 1
                else:
                    parsed_am.add[(a, p, x)] = 0
                if p_string in schema_info[a.name]["delete effects"]:
                    parsed_am.dele[(a, p, x)] = 1
                else:
                    parsed_am.dele[(a, p, x)] = 0
    return parsed_am