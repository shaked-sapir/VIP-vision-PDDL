"""Tests for :mod:`src.milp.rosame26_emitter`, on all five benchmark domains.

    python -m pytest src/milp/test_rosame26_emitter.py

The gate this file exists for is :class:`TestGroundTruthRoundTrip`: a head
carrying the reference domain's own semantics must emit a domain that AMLGym
scores at precision 1.0 and recall 1.0. AMLGym binds arguments **positionally**
(``SimpleDomainReader`` renames every parameter to ``?param_k`` by index), so
that score is exactly the property the vendored ``extract_pddl`` loses on the
four domains whose signatures ROSAME reorders.

:class:`TestTheVendoredEmitterWouldScoreZero` is the mutation check. It runs the
same round trip through ``extract_pddl`` and asserts the collapse, which is what
stops the round-trip gate from being read as vacuous — and it is parametrized
over all five domains, so blocksworld's identity permutation is visible as the
one domain the collapse spares.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pytest
import torch
from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain

import src.milp  # noqa: F401  (vendor sys.path bootstrap)
from src.milp.domain_assets import (
    BENCH_DOMAINS,
    build_domain,
    source_pddl_path,
    write_grounding_assets,
)
from src.milp.rosame26_emitter import (
    ADD_EFFECT_CLASS,
    DELETE_EFFECT_CLASS,
    DegenerateModelError,
    check_not_degenerate,
    effect_counts,
    emit_pddl,
    format_signature,
    parameter_names,
    variables_by_type,
    _conjunction,
    _in_pddl_order,
)

from dl.util.ROSAME.rosame import extract_pddl, get_domain_model

#: Domains ROSAME's sorted-type grounding reorders (plan §4.2b).
REORDERED_DOMAINS = ["depot", "gripper", "hanoi", "npuzzle"]

_OBJECTS_PER_TYPE = 2

# The four classes of the schema head: nothing, add, precondition, both.
_NEUTRAL_CLASS = 0


class _FakeType:
    def __init__(self, name: str) -> None:
        self.name = name


def _leaf_types(domain_key: str):
    """Types with no declared child; objects of a parent type would not ground."""
    types = list(build_domain(domain_key).types_set)
    parents = {t.parent.name for t in types if t.parent}
    leaves = [t for t in types if t.name not in parents and t.name != "object"]
    return leaves or [t for t in types if t.name != "object"] or types


def _objects(domain_key: str) -> Dict[str, List[str]]:
    return {
        one_type.name: [f"{one_type.name}{i}" for i in range(1, _OBJECTS_PER_TYPE + 1)]
        for one_type in _leaf_types(domain_key)
    }


def _head(domain_key: str, root: Path):
    """The vendored ``Domain_Model``, grounded over a small universe."""
    write_grounding_assets(domain_key, _objects(domain_key), root)
    return get_domain_model(domain_key, str(root))


def _reference_domain(domain_key: str) -> Domain:
    return DomainParser(source_pddl_path(domain_key), partial_parsing=False).parse_domain()


# ── planting the reference semantics into the head ──────────────────────


def _literal_key(text: str) -> Tuple[str, ...]:
    """``"(at-pile ?pl - pile ?d - depot)"`` as ``("at-pile", "?pl", "?d")``.

    ``pddl_plus_parser`` stringifies a lifted literal with its argument types
    interleaved; the type annotations are dropped, leaving name and variables.
    """
    tokens = text.strip().strip("()").split()
    return (tokens[0], *[t for t in tokens[1:] if t.startswith("?")])


def _reference_classes(action) -> Dict[Tuple[str, ...], int]:
    """``{literal key: head class}`` for one reference action schema."""
    preconditions = {
        _literal_key(str(operand)) for operand in action.preconditions.root.operands
    }
    adds, deletes = set(), set()
    for effect in action.discrete_effects:
        text = str(effect).strip()
        if text.startswith("(not "):
            deletes.add(_literal_key(text[len("(not ") : -1]))
        else:
            adds.add(_literal_key(text))

    classes: Dict[Tuple[str, ...], int] = {}
    for key in preconditions | adds | deletes:
        if key in deletes:
            classes[key] = DELETE_EFFECT_CLASS
        elif key in adds:
            classes[key] = ADD_EFFECT_CLASS
        elif key in preconditions:
            classes[key] = 2
        if key in adds and key in preconditions:
            # An add that is also a precondition: upstream's four classes cannot
            # express it, and no reference schema of ours has one.
            raise AssertionError(f"{key} is both an add effect and a precondition")
    return classes


def unrepresentable_deletes(domain_key: str) -> Dict[str, List[Tuple[str, ...]]]:
    """``{action: [literal key]}`` the four-class head cannot express.

    Class 3 is *precondition and delete effect* jointly, so a schema that deletes
    a literal it does not require is beyond the representation: emitting it costs
    one spurious precondition, and omitting it costs one delete effect. Measured
    on our five domains this is **depot only**, at two literals — see
    :class:`TestTheRepresentationHasACeiling`.
    """
    reference = _reference_domain(domain_key)
    out: Dict[str, List[Tuple[str, ...]]] = {}
    for name, action in reference.actions.items():
        preconditions = {
            _literal_key(str(operand))
            for operand in action.preconditions.root.operands
        }
        stray = [
            _literal_key(str(effect).strip()[len("(not ") : -1])
            for effect in action.discrete_effects
            if str(effect).strip().startswith("(not ")
            and _literal_key(str(effect).strip()[len("(not ") : -1]) not in preconditions
        ]
        if stray:
            out[name] = stray
    return out


def _plant_reference_semantics(domain_model, domain_key: str) -> None:
    """Replace each schema's head with a one-hot of the reference domain's model.

    The propositions are enumerated exactly as :func:`schema_literals` enumerates
    them — ``Predicate.ground`` in head order — and each one's class is looked up
    by its *PDDL-ordered* variable tuple, which is what makes a head-order bug
    show up as a wrong emission rather than as a planting bug.
    """
    reference = _reference_domain(domain_key)
    signatures = {s.name: s.types for s in build_domain(domain_key).action_schemas}
    predicate_types = {p.name: p.types for p in build_domain(domain_key).predicates}

    for schema in domain_model.action_schemas:
        types = signatures[schema.name]
        names = parameter_names(types)
        by_type_name = variables_by_type(types, names)
        var_map = {t: by_type_name[t.name] for t in schema.params_types}

        # Reference variable names, positionally, to ours.
        rename = dict(
            zip(reference.actions[schema.name].signature.keys(), names)
        )
        wanted = {
            (key[0], *[rename[v] for v in key[1:]]): one_class
            for key, one_class in _reference_classes(
                reference.actions[schema.name]
            ).items()
        }

        rows: List[int] = []
        for predicate in schema.predicates:
            for proposition in predicate.ground(var_map):
                ordered = _in_pddl_order(proposition, predicate_types[predicate.name])
                rows.append(wanted.get(tuple(ordered.split()), _NEUTRAL_CLASS))

        schema.forward = (  # type: ignore[method-assign]
            lambda rows=rows: torch.nn.functional.one_hot(
                torch.tensor(rows), num_classes=4
            ).float()
        )


def _emit_and_score(text: str, domain_key: str, tmp_path: Path) -> Tuple[float, float]:
    """``(precision, recall)`` of ``text`` against ``domain_key``'s reference."""
    from amlgym.metrics import syntactic_precision, syntactic_recall

    learned = tmp_path / f"learned_{domain_key}.pddl"
    learned.write_text(text)
    reference = str(source_pddl_path(domain_key))
    precision = syntactic_precision(str(learned), reference)
    recall = syntactic_recall(str(learned), reference)
    return precision, recall


def _overall(scores: Dict[str, float]) -> float:
    """The mean of a metric dict's numeric entries."""
    values = [v for v in scores.values() if isinstance(v, (int, float))]
    return sum(values) / len(values)


# ── the pieces ──────────────────────────────────────────────────────────


class TestParameterNames:
    def test_one_name_per_slot(self) -> None:
        assert parameter_names([_FakeType("a"), _FakeType("b")]) == ["?x0", "?x1"]

    def test_a_nullary_signature_has_none(self) -> None:
        assert parameter_names([]) == []

    def test_names_are_distinct_for_a_repeated_type(self) -> None:
        names = parameter_names([_FakeType("disc"), _FakeType("peg"), _FakeType("disc")])
        assert len(set(names)) == 3


class TestVariablesByType:
    def test_same_typed_slots_group_in_pddl_order(self) -> None:
        types = [_FakeType("disc"), _FakeType("peg"), _FakeType("disc")]
        assert variables_by_type(types, ["?x0", "?x1", "?x2"]) == {
            "disc": ["?x0", "?x2"],
            "peg": ["?x1"],
        }


class TestFormatSignature:
    def test_it_writes_pddl_order(self) -> None:
        types = [_FakeType("crane"), _FakeType("package")]
        assert format_signature(types, ["?x0", "?x1"]) == "?x0 - crane ?x1 - package"

    def test_a_nullary_signature_is_empty(self) -> None:
        assert format_signature([], []) == ""


class TestConjunction:
    def test_an_empty_block_is_a_parsable_empty_and(self) -> None:
        assert _conjunction([]) == "(and )"

    def test_literals_are_conjoined(self) -> None:
        assert _conjunction(["(p ?x0)", "(q ?x1)"]) == "(and (p ?x0) (q ?x1))"


class TestInPddlOrder:
    def test_a_reordered_predicate_is_put_back(self) -> None:
        types = [_FakeType("tile"), _FakeType("position")]
        assert _in_pddl_order("at ?x1 ?x0", types) == "at ?x0 ?x1"

    def test_an_already_ordered_predicate_is_unchanged(self) -> None:
        types = [_FakeType("crane"), _FakeType("depot")]
        assert _in_pddl_order("at-crane ?x0 ?x1", types) == "at-crane ?x0 ?x1"

    def test_a_nullary_predicate_survives(self) -> None:
        assert _in_pddl_order("handempty", []) == "handempty"

    def test_an_arity_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="has 1 arguments but"):
            _in_pddl_order("at ?x0", [_FakeType("tile"), _FakeType("position")])


# ── the whole emission ──────────────────────────────────────────────────


@pytest.mark.parametrize("domain_key", BENCH_DOMAINS)
class TestEmission:
    def test_the_output_parses(self, domain_key, tmp_path) -> None:
        model = _head(domain_key, tmp_path)
        path = tmp_path / "learned.pddl"
        path.write_text(emit_pddl(model, domain_key))
        DomainParser(path, partial_parsing=False).parse_domain()

    def test_every_schema_keeps_its_reference_signature(
        self, domain_key, tmp_path
    ) -> None:
        model = _head(domain_key, tmp_path)
        path = tmp_path / "learned.pddl"
        path.write_text(emit_pddl(model, domain_key))
        learned = DomainParser(path, partial_parsing=False).parse_domain()
        reference = _reference_domain(domain_key)

        assert set(learned.actions) == set(reference.actions)
        for name, action in reference.actions.items():
            assert [str(t) for t in learned.actions[name].signature.values()] == [
                str(t) for t in action.signature.values()
            ], name

    def test_every_predicate_keeps_its_reference_signature(
        self, domain_key, tmp_path
    ) -> None:
        model = _head(domain_key, tmp_path)
        path = tmp_path / "learned.pddl"
        path.write_text(emit_pddl(model, domain_key))
        learned = DomainParser(path, partial_parsing=False).parse_domain()
        reference = _reference_domain(domain_key)

        for name, predicate in reference.predicates.items():
            assert [str(t) for t in learned.predicates[name].signature.values()] == [
                str(t) for t in predicate.signature.values()
            ], name

    def test_a_head_that_wants_nothing_emits_an_empty_but_parsable_model(
        self, domain_key, tmp_path
    ) -> None:
        model = _head(domain_key, tmp_path)
        for schema in model.action_schemas:
            width = schema.randn.shape[0]
            schema.forward = (  # type: ignore[method-assign]
                lambda width=width: torch.nn.functional.one_hot(
                    torch.zeros(width, dtype=torch.long), num_classes=4
                ).float()
            )
        path = tmp_path / "learned.pddl"
        path.write_text(emit_pddl(model, domain_key))
        learned = DomainParser(path, partial_parsing=False).parse_domain()

        assert all(not a.discrete_effects for a in learned.actions.values())


class TestEmissionRejectsAMismatchedHead:
    def test_a_head_for_another_domain_raises(self, tmp_path) -> None:
        model = _head("gripper", tmp_path)
        with pytest.raises(ValueError, match="regenerate the grounding assets"):
            emit_pddl(model, "blocksworld")


# ── the gate ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("domain_key", BENCH_DOMAINS)
class TestGroundTruthRoundTrip:
    """A head carrying the reference semantics scores as well as it can.

    Perfectly, except where the four-class head cannot express the reference
    schema at all — :func:`unrepresentable_deletes`, which is depot's ``load``
    and ``unload`` and nothing else. Recall is unaffected there because the
    spurious precondition is an extra, not a miss.
    """

    def test_recall_is_perfect(self, domain_key, tmp_path) -> None:
        model = _head(domain_key, tmp_path)
        _plant_reference_semantics(model, domain_key)
        _, recall = _emit_and_score(emit_pddl(model, domain_key), domain_key, tmp_path)

        assert _overall(recall) == pytest.approx(1.0), recall

    def test_precision_is_perfect_wherever_the_head_can_express_the_schema(
        self, domain_key, tmp_path
    ) -> None:
        model = _head(domain_key, tmp_path)
        _plant_reference_semantics(model, domain_key)
        precision, _ = _emit_and_score(
            emit_pddl(model, domain_key), domain_key, tmp_path
        )
        beyond = unrepresentable_deletes(domain_key)

        if not beyond:
            assert _overall(precision) == pytest.approx(1.0), precision
        else:
            assert _overall(precision) < 1.0
            assert _overall(precision) > 0.95, precision

    def test_every_expressible_schema_matches_the_reference_exactly(
        self, domain_key, tmp_path
    ) -> None:
        """Literal by literal, under the reference's own parameter positions."""
        model = _head(domain_key, tmp_path)
        _plant_reference_semantics(model, domain_key)
        path = tmp_path / "learned.pddl"
        path.write_text(emit_pddl(model, domain_key))
        learned = DomainParser(path, partial_parsing=False).parse_domain()
        reference = _reference_domain(domain_key)
        beyond = unrepresentable_deletes(domain_key)

        for name, action in reference.actions.items():
            if name in beyond:
                continue
            assert _literals(learned.actions[name]) == _literals(action), name


@pytest.mark.parametrize("domain_key", BENCH_DOMAINS)
class TestTheVendoredEmitterWouldScoreZero:
    """The mutation check: the same head through ``extract_pddl`` collapses.

    blocksworld is the exception the plan names — 0 reordered schemas, so its
    permutation is the identity and the vendored emitter scores it correctly
    once its unparsable ``()`` blocks are patched around. That is precisely why
    a blocksworld-only gate would have shown nothing.
    """

    def test_the_vendored_emission_permutes_the_signature(
        self, domain_key, tmp_path
    ) -> None:
        model = _head(domain_key, tmp_path)
        vendored = extract_pddl(model, domain_name=domain_key)
        reference = _reference_domain(domain_key)

        emitted = {
            line.split("(:action ")[1].strip(): None
            for line in vendored.splitlines()
            if "(:action " in line
        }
        assert set(emitted) == set(reference.actions)

        signatures = _vendored_signatures(vendored)
        differing = [
            name
            for name, types in signatures.items()
            if types != [str(t) for t in reference.actions[name].signature.values()]
        ]
        if domain_key == "blocksworld":
            assert differing == []
        else:
            assert differing, f"{domain_key} is listed as reordered but nothing moved"


def _literals(action) -> Tuple[frozenset, frozenset]:
    """``(preconditions, effects)`` as sets keyed by parameter *position*.

    Parameter names differ between the emitted and reference domains, so each
    variable is replaced by its index in the action's signature — which is also
    how AMLGym compares them.
    """
    positions = {name: f"?{i}" for i, name in enumerate(action.signature)}

    def key(text: str) -> Tuple[str, ...]:
        stripped = text.strip()
        negated = stripped.startswith("(not ")
        if negated:
            stripped = stripped[len("(not ") : -1].strip()
        tokens = stripped.strip("()").split()
        return (
            "not" if negated else "",
            tokens[0],
            *[positions[t] for t in tokens[1:] if t.startswith("?")],
        )

    return (
        frozenset(key(str(o)) for o in action.preconditions.root.operands),
        frozenset(key(str(e)) for e in action.discrete_effects),
    )


class TestTheRepresentationHasACeiling:
    """ROSAME's four classes cannot express a delete without a precondition.

    Class 3 is the two jointly. Measured over all five domains this costs
    **depot** two literals and every other domain nothing, which is a bound on
    what the arm can score there however well it trains — not a defect of the
    emitter, and not something training can close.
    """

    def test_only_depot_is_affected(self) -> None:
        affected = {
            domain_key: unrepresentable_deletes(domain_key)
            for domain_key in BENCH_DOMAINS
        }
        assert {k: v for k, v in affected.items() if v} == {
            "depot": {
                "load": [("at", "?p", "?d")],
                "unload": [("clear", "?p")],
            }
        }


class TestGateFourTheDegenerateModelGuard:
    """A model whose every schema has an empty effect block must raise.

    This is the ICAPS-24 empty-effects collapse: not a weak model but an empty
    one, in which no action changes any state, so every solving score is 0 for a
    structural reason no metric names. It went unnoticed for a whole grid.
    """

    @pytest.mark.parametrize("domain_key", BENCH_DOMAINS)
    def test_a_reference_model_passes(self, domain_key, tmp_path) -> None:
        model = _head(domain_key, tmp_path)
        _plant_reference_semantics(model, domain_key)
        counts = check_not_degenerate(emit_pddl(model, domain_key))

        assert all(count > 0 for count in counts.values()), counts

    @pytest.mark.parametrize("domain_key", BENCH_DOMAINS)
    def test_an_all_neutral_head_raises(self, domain_key, tmp_path) -> None:
        model = _head(domain_key, tmp_path)
        _neutralise(model)
        with pytest.raises(DegenerateModelError, match="empty-effects collapse"):
            check_not_degenerate(emit_pddl(model, domain_key))

    @pytest.mark.parametrize("domain_key", BENCH_DOMAINS)
    def test_a_precondition_only_head_still_raises(
        self, domain_key, tmp_path
    ) -> None:
        """Preconditions are not effects; a model of pure guards is degenerate."""
        model = _head(domain_key, tmp_path)
        for schema in model.action_schemas:
            width = schema.randn.shape[0]
            schema.forward = (  # type: ignore[method-assign]
                lambda width=width: torch.nn.functional.one_hot(
                    torch.full((width,), 2, dtype=torch.long), num_classes=4
                ).float()
            )
        with pytest.raises(DegenerateModelError, match="empty-effects collapse"):
            check_not_degenerate(emit_pddl(model, domain_key))

    def test_one_surviving_effect_is_enough(self, tmp_path) -> None:
        model = _head("blocksworld", tmp_path)
        _neutralise(model)
        schema = model.action_schemas[0]
        width = schema.randn.shape[0]
        rows = torch.zeros(width, dtype=torch.long)
        rows[0] = ADD_EFFECT_CLASS
        schema.forward = (  # type: ignore[method-assign]
            lambda rows=rows: torch.nn.functional.one_hot(rows, num_classes=4).float()
        )

        assert sum(check_not_degenerate(emit_pddl(model, "blocksworld")).values()) == 1

    def test_a_domain_with_no_action_raises(self) -> None:
        with pytest.raises(DegenerateModelError, match="no action at all"):
            check_not_degenerate("(define (domain d) (:predicates (p ?x0 - t)))")


class TestEffectCounts:
    def test_a_negation_counts_once(self) -> None:
        text = (
            "    (:action a\n"
            "        :parameters (?x0 - t)\n"
            "        :precondition (and (p ?x0))\n"
            "        :effect (and (not (p ?x0)) (q ?x0))\n"
            "    )"
        )
        assert effect_counts(text) == {"a": 2}

    def test_an_empty_block_counts_zero(self) -> None:
        text = (
            "    (:action a\n"
            "        :parameters (?x0 - t)\n"
            "        :precondition (and (p ?x0))\n"
            "        :effect (and )\n"
            "    )"
        )
        assert effect_counts(text) == {"a": 0}

    def test_preconditions_are_not_counted(self) -> None:
        text = (
            "    (:action a\n"
            "        :parameters (?x0 - t)\n"
            "        :precondition (and (p ?x0) (q ?x0) (r ?x0))\n"
            "        :effect (and )\n"
            "    )"
        )
        assert effect_counts(text) == {"a": 0}


def _neutralise(domain_model) -> None:
    """Pin every schema head to the neutral class."""
    for schema in domain_model.action_schemas:
        width = schema.randn.shape[0]
        schema.forward = (  # type: ignore[method-assign]
            lambda width=width: torch.nn.functional.one_hot(
                torch.zeros(width, dtype=torch.long), num_classes=4
            ).float()
        )


def _vendored_signatures(text: str) -> Dict[str, List[str]]:
    """``{action: [type, ...]}`` from an ``extract_pddl`` string."""
    signatures: Dict[str, List[str]] = {}
    name = None
    for line in text.splitlines():
        if "(:action " in line:
            name = line.split("(:action ")[1].strip()
        elif ":parameters" in line and name is not None:
            body = line.split(":parameters", 1)[1].strip()[1:-1]
            signatures[name] = _types_of(body)
            name = None
    return signatures


def _types_of(body: str) -> List[str]:
    """The type names of a ``?a - crane ?b - depot`` parameter body."""
    tokens = body.split()
    return [tokens[i] for i in range(2, len(tokens), 3)]
