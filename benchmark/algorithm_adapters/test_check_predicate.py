"""Round-trip every grounded predicate of every domain through ``check_predicate``.

    python -m pytest benchmark/algorithm_adapters/test_check_predicate.py

``Rosame_Runner.check_predicate`` (AMLGym ``rosame_runner.py:254``) looks the
predicate string up verbatim, and on a miss retries every **permutation** of its
arguments, returning the first that hits. It is on the path that builds the GT
final-state vector — the gamma anchor — so a wrong match there is silent
corruption of the only non-self-referential term in the ROSAME-I loss.

The fallback exists because ROSAME grounds from a *type-count* map
(``_get_types_count``) rather than the ordered PDDL signature, so proposition
argument order is not guaranteed to follow the signature. These tests pin down
that on our five domains it nevertheless always does, and that the fallback
never silently substitutes a different proposition.

**That holds for the ICAPS-24 fork only, and not because the count map is
order-preserving — it is not.** AMLGym's ``models/rosame.py`` carries upstream's
sort commented out::

    # self.params_types = sorted(params.keys(), key=lambda x: x.name)
    self.params_types = list(params.keys())

so it recovers PDDL order by accident of dict insertion order. ICAPS-26's
``src/milp/vendor/dl/util/ROSAME/rosame.py`` leaves the sort active and does
reorder, on four of our five domains. See
``src/milp/test_rosame_argument_order.py``; do not read this file's green as
evidence that the 26 arms need no permutation step.

Covers:
  1. Identity: every grounded proposition round-trips to itself, so the
     permutation branch is never reached on well-formed input.
  2. Propositions are unique and signature-ordered (no repeated-object tuples,
     matching the vendored ``Instance``'s ``permutations`` grounding).
  3. A repeated-object predicate is reported as absent rather than matched to a
     distinct-object proposition.
  4. An unknown predicate name returns None rather than raising.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pytest

from benchmark.algorithm_adapters.po_rosame_runner import PORosame_Runner
from src.utils.config import load_config

DOMAINS = ["blocksworld", "depot", "gripper", "hanoi", "npuzzle"]

_OBJECTS_PER_TYPE = 3


def _domain_file(domain_key: str) -> Path:
    return Path(load_config()["domains"][domain_key]["domain_file"])


def _grounded_runner(domain_key: str) -> PORosame_Runner:
    """A runner grounded on synthetic objects — no problem file needed.

    ``prepare_rosame`` only reads ``self.problem`` to derive ``self.objects``, so
    setting the object map directly grounds the same way a real problem would.
    """
    runner = PORosame_Runner(_domain_file(domain_key))
    type_names = [name for name in runner.domain.types if name != "object"]
    runner.objects = {
        name: [f"{name[:2]}{i}" for i in range(1, _OBJECTS_PER_TYPE + 1)]
        for name in type_names
    }
    runner.rosame = runner.get_domain_model(
        {
            "types": runner._prepare_types(runner.domain.types),
            "predicates": runner._prepare_predicates(runner.domain.predicates),
            "action_schemas": runner._prepare_action_schema(runner.domain.actions),
        }
    )
    return runner


@pytest.fixture(scope="module")
def runners() -> Dict[str, PORosame_Runner]:
    return {key: _grounded_runner(key) for key in DOMAINS}


@pytest.mark.parametrize("domain_key", DOMAINS)
def test_every_proposition_round_trips_to_itself(domain_key, runners) -> None:
    """The permutation fallback must never fire on a proposition that exists."""
    runner = runners[domain_key]
    propositions = list(runner.rosame.propositions)
    assert propositions, f"{domain_key} grounded no propositions"

    mismatched = [p for p in propositions if runner.check_predicate(p) != p]
    assert not mismatched, (
        f"{domain_key}: {len(mismatched)} of {len(propositions)} propositions do not "
        f"round-trip; first few {mismatched[:5]}"
    )


@pytest.mark.parametrize("domain_key", DOMAINS)
def test_propositions_are_unique_and_have_distinct_objects(domain_key, runners) -> None:
    """Grounding excludes repeated-object tuples, as the vendored Instance does."""
    propositions = list(runners[domain_key].rosame.propositions)
    assert len(propositions) == len(set(propositions))
    for proposition in propositions:
        args = proposition.split()[1:]
        assert len(args) == len(set(args)), f"{domain_key}: repeated args in {proposition}"


@pytest.mark.parametrize("domain_key", DOMAINS)
def test_repeated_object_predicate_is_not_matched_to_a_neighbour(domain_key, runners) -> None:
    """`on b1 b1` has no proposition; it must resolve to None, not to `on b1 b2`."""
    runner = runners[domain_key]
    probed = 0
    for name, predicate in runner.domain.predicates.items():
        types = [t.name for t in predicate.signature.values()]
        if len(types) < 2 or len(set(types)) != 1:
            continue  # need >=2 same-typed args for a repeated tuple to be type-valid
        obj = runner.objects[types[0]][0]
        repeated = f"{name} " + " ".join([obj] * len(types))
        assert repeated not in runner.rosame.propositions
        assert runner.check_predicate(repeated) is None, (
            f"{domain_key}: check_predicate({repeated!r}) matched a different proposition"
        )
        probed += 1
    if probed == 0:
        pytest.skip(f"{domain_key} has no predicate with repeated argument types")


@pytest.mark.parametrize("domain_key", DOMAINS)
def test_unknown_predicate_returns_none(domain_key, runners) -> None:
    assert runners[domain_key].check_predicate("no-such-predicate x1 x2") is None
