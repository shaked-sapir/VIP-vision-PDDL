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

from benchmark.algorithm_adapters.endpoint_anchors import EndpointAnchors
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


def _with_anchors(prepared, anchors: EndpointAnchors):
    """The two-step micro trace ``a`` with a fifth :class:`EndpointAnchors` element."""
    return [(*prepared[0], anchors), prepared[1]]


def test_a_four_element_problem_carries_no_endpoint_anchors():
    """The shape every existing caller passes: full final-state mask, no init anchor."""
    with tempfile.TemporaryDirectory() as tmp:
        domain_path, prepared = _prepare(Path(tmp))
        for trace in _seeded_runner(domain_path).prepare_traces(prepared):
            assert trace.final_state_mask is None
            assert trace.init_state_vec is None
    print("PASS  a four-element problem carries no endpoint anchors")


def test_default_anchors_leave_the_loss_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        domain_path, prepared = _prepare(Path(tmp))
        runner = _seeded_runner(domain_path)
        plain_trace, anchored_trace = runner.prepare_traces(
            [prepared[0], (*prepared[0], EndpointAnchors())]
        )
        preds = torch.rand(3, len(runner.rosame.propositions))
        with torch.no_grad():
            assert torch.equal(
                runner._loss_from_predictions(preds, plain_trace, 10.0, 0.2),
                runner._loss_from_predictions(preds, anchored_trace, 10.0, 0.2),
            )
    print("PASS  default anchors leave the loss unchanged")


def test_an_unknown_final_atom_is_left_out_of_the_final_state_term():
    """The loss is the same whichever value the anchor gives an unknown atom."""
    with tempfile.TemporaryDirectory() as tmp:
        domain_path, prepared = _prepare(Path(tmp))
        runner = _seeded_runner(domain_path)
        problem, images, actions, final = prepared[0]
        unknown = EndpointAnchors(final_unknown=("visited r2",))
        as_true, as_false = runner.prepare_traces([
            (problem, images, actions, final, unknown),
            (problem, images, actions, [p for p in final if p != "visited r2"], unknown),
        ])
        props = list(runner.rosame.propositions.keys())
        assert as_true.final_state_mask[props.index("visited r2")].item() == 0.0
        assert as_true.final_state_mask.sum().item() == len(props) - 1
        assert not torch.equal(as_true.final_state_vec, as_false.final_state_vec)

        preds = torch.rand(3, len(props))
        with torch.no_grad():
            masked_true = runner._loss_from_predictions(preds, as_true, 10.0, 0.2)
            masked_false = runner._loss_from_predictions(preds, as_false, 10.0, 0.2)
            as_true.final_state_mask = None
            unmasked = runner._loss_from_predictions(preds, as_true, 10.0, 0.2)
        assert torch.equal(masked_true, masked_false)
        assert unmasked.item() > masked_true.item()
    print("PASS  an unknown final atom is left out of the final-state term")


def test_an_anchored_initial_state_stands_in_for_the_first_frame():
    """With the GT init given, frame 0's prediction is not read by the loss."""
    with tempfile.TemporaryDirectory() as tmp:
        domain_path, prepared = _prepare(Path(tmp))
        runner = _seeded_runner(domain_path)
        anchors = EndpointAnchors(init_predicates=("at r1",))
        trace = runner.prepare_traces(_with_anchors(prepared, anchors))[0]
        props = list(runner.rosame.propositions.keys())
        assert {props[i] for i, v in enumerate(trace.init_state_vec) if v == 1.0} == {"at r1"}

        preds = torch.rand(3, len(props))
        other_first = preds.clone()
        other_first[0] = torch.rand(len(props))
        as_init = preds.clone()
        as_init[0] = trace.init_state_vec
        with torch.no_grad():
            anchored = runner._loss_from_predictions(preds, trace, 10.0, 0.2)
            assert torch.equal(
                anchored, runner._loss_from_predictions(other_first, trace, 10.0, 0.2)
            )
            trace.init_state_vec = None
            assert torch.equal(
                anchored, runner._loss_from_predictions(as_init, trace, 10.0, 0.2)
            )
            assert not torch.equal(
                anchored, runner._loss_from_predictions(preds, trace, 10.0, 0.2)
            )
    print("PASS  an anchored initial state stands in for the first frame")


def test_a_single_step_trace_takes_an_initial_state_anchor():
    with tempfile.TemporaryDirectory() as tmp:
        domain_path, prepared = _prepare(Path(tmp))
        runner = _seeded_runner(domain_path)
        problem, images, actions, final = prepared[1]
        anchors = EndpointAnchors(init_predicates=("at r2",))
        trace = runner.prepare_traces([prepared[0], (problem, images, actions, final, anchors)])[1]
        preds = torch.rand(2, len(runner.rosame.propositions))
        with torch.no_grad():
            loss = runner._loss_from_predictions(preds, trace, 10.0, 0.2)
        assert torch.isfinite(loss) and loss.item() > 0.0
    print("PASS  a single-step trace takes an initial-state anchor")


if __name__ == "__main__":
    test_grounding_covers_every_problems_objects()
    test_a_trace_from_the_smaller_problem_is_encoded_full_width()
    test_a_predicate_outside_the_grounding_is_dropped_not_raised()
    test_trajectory_loss_is_its_two_halves_composed()
    test_a_four_element_problem_carries_no_endpoint_anchors()
    test_default_anchors_leave_the_loss_unchanged()
    test_an_unknown_final_atom_is_left_out_of_the_final_state_term()
    test_an_anchored_initial_state_stands_in_for_the_first_frame()
    test_a_single_step_trace_takes_an_initial_state_anchor()
    print("ALL TESTS PASSED")
