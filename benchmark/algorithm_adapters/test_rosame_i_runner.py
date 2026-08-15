"""Tests for the ROSAME-I adapter's grounding and loss decomposition.

Two properties are load-bearing for the MILP arm built on top of this runner.
The CV head is one ``Linear(256, n_props)`` readout shared by every trace in a
fold, so the grounding it is built on must cover every problem's objects. And
:meth:`RosameI_Runner._trajectory_loss` was split into a forward pass plus a
loss-from-predictions half, which the MILP loop calls separately; the two paths
must stay one function.

    python -m benchmark.algorithm_adapters.test_rosame_i_runner
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser

from benchmark.algorithm_adapters.rosame_i_runner import RosameI_Runner

_DOMAIN = """(define (domain micro)
 (:requirements :strips :typing)
 (:types room)
 (:predicates (at ?r - room) (visited ?r - room))
 (:action move
  :parameters (?from - room ?to - room)
  :precondition (and (at ?from))
  :effect (and (not (at ?from)) (at ?to) (visited ?to)))
)
"""

# Deliberately disjoint universes: r1/r3 exist only in A, r4 only in B.
_PROBLEM_A = """(define (problem micro-a)
 (:domain micro)
 (:objects r1 r2 r3 - room)
 (:init (at r1))
 (:goal (and (at r3)))
)
"""

_PROBLEM_B = """(define (problem micro-b)
 (:domain micro)
 (:objects r2 r4 - room)
 (:init (at r2))
 (:goal (and (at r4)))
)
"""

_SEED = 4242


# ---------------------------------------------------------------- fixtures

def _write_images(directory: Path, count: int) -> List[Path]:
    """``count`` distinct 8x8 PNGs named ``state_i.png`` under ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        path = directory / f"state_{i}.png"
        Image.new("RGB", (8, 8), color=(10 * i, 20, 30)).save(path)
        paths.append(path)
    return paths


def _prepare(root: Path) -> Tuple[Path, List[Tuple[object, List[Path], List[str], List[str]]]]:
    """The micro world as the ``(problem, images, actions, final)`` tuples ROSAME-I takes."""
    domain_path = root / "domain.pddl"
    domain_path.write_text(_DOMAIN)
    partial_domain = DomainParser(domain_path, partial_parsing=True).parse_domain()

    prepared = []
    for name, text, actions, final in (
        ("a", _PROBLEM_A, ["move r1 r2", "move r2 r3"], ["at r3", "visited r2", "visited r3"]),
        ("b", _PROBLEM_B, ["move r2 r4"], ["at r4", "visited r4"]),
    ):
        problem_path = root / f"{name}.pddl"
        problem_path.write_text(text)
        problem = ProblemParser(problem_path, partial_domain).parse_problem()
        images = _write_images(root / name, len(actions) + 1)
        prepared.append((problem, images, actions, final))
    return domain_path, prepared


def _seeded_runner(domain_path: Path) -> RosameI_Runner:
    """A CPU runner whose schema MLPs and CV head come from a fixed seed."""
    torch.manual_seed(_SEED)
    return RosameI_Runner(str(domain_path), device="cpu", seed=_SEED)


# ------------------------------------------------------------------- tests

def test_grounding_covers_every_problems_objects():
    """One head over the union of all four rooms, not just the first problem's three."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        domain_path, prepared = _prepare(root)

        runner = _seeded_runner(domain_path)
        runner.ground_union([problem for problem, _i, _a, _f in prepared])

        props = list(runner.rosame.propositions.keys())

    assert sorted(props) == sorted(
        [f"at r{i}" for i in (1, 2, 3, 4)] + [f"visited r{i}" for i in (1, 2, 3, 4)]
    ), props
    assert runner.cv_model.fc[-1].out_features == len(props)
    print("PASS  grounding is the union of every problem's objects")


def test_a_trace_from_the_smaller_problem_is_encoded_full_width():
    """B knows nothing of r1/r3, yet its label vector spans the shared columns."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        domain_path, prepared = _prepare(root)

        runner = _seeded_runner(domain_path)
        traces = runner.prepare_traces(prepared)

        props = list(runner.rosame.propositions.keys())
        assert [trace.name for trace in traces] == ["a", "b"], traces
        assert traces[1].final_state_vec.shape == (len(props),)

        positive = {props[i] for i, v in enumerate(traces[1].final_state_vec) if v == 1.0}

    assert positive == {"at r4", "visited r4"}, positive
    print("PASS  a smaller problem's trace is encoded in the shared column order")


def test_a_predicate_outside_the_grounding_is_dropped_not_raised():
    """An unmatched fluent is closed-world false, so the encoder leaves it at 0."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        domain_path, prepared = _prepare(root)

        runner = _seeded_runner(domain_path)
        runner.ground_union([problem for problem, _i, _a, _f in prepared])

        vec = runner._encode_state(["at r9", "at r2"])
        props = list(runner.rosame.propositions.keys())

    assert vec.sum().item() == 1.0, vec
    assert vec[props.index("at r2")].item() == 1.0
    print("PASS  a predicate outside the grounding is dropped, not raised")


def test_trajectory_loss_is_its_two_halves_composed():
    """``_trajectory_loss`` == ``_loss_from_predictions(_forward_predictions(...))``.

    The MILP loop calls the halves separately so it can reuse one forward pass
    for both the ROSAME-I loss and the pseudo-label losses. Nothing else forces
    the composed path to stay equal to the one the plain baseline runs.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        domain_path, prepared = _prepare(root)

        runner = _seeded_runner(domain_path)
        traces = runner.prepare_traces(prepared)
        runner.cv_model.eval()  # fixed BatchNorm statistics across both calls

        gamma, lambda_ = 10.0, 0.2
        for trace in traces:
            with torch.no_grad():
                whole = runner._trajectory_loss(trace, gamma, lambda_, augment=False)
                preds = runner._forward_predictions(trace, augment=False)
                halves = runner._loss_from_predictions(preds, trace, gamma, lambda_)
            assert whole.item() > 0.0, (trace.name, whole.item())
            assert torch.equal(whole, halves), (trace.name, whole.item(), halves.item())

    print("PASS  _trajectory_loss is exactly its two halves composed")


if __name__ == "__main__":
    test_grounding_covers_every_problems_objects()
    test_a_trace_from_the_smaller_problem_is_encoded_full_width()
    test_a_predicate_outside_the_grounding_is_dropped_not_raised()
    test_trajectory_loss_is_its_two_halves_composed()
    print("ALL TESTS PASSED")
