"""Tests for the ROSAME adapter's training loops and anytime snapshots.

    python -m benchmark.algorithm_adapters.test_po_rosame_runner

The load-bearing one is :func:`test_local_loop_matches_vendored`. ``learn_per_trajectory``
used to delegate to the vendored ``learn_rosame``, which is why it could claim
fidelity for free; it now runs its own loop so that snapshots have somewhere to
hook. That claim is no longer free, so it is asserted here instead.

The rest guard the property that makes the anytime curve meaningful at all:
observing the run must not change it, neither in the model it produces nor in
the clock it reports.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Tuple

import torch
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser

from benchmark.algorithm_adapters.anytime_snapshots import SnapshotWriter
from benchmark.algorithm_adapters.po_rosame_runner import PORosame_Runner

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

_PROBLEM = """(define (problem micro-p)
 (:domain micro)
 (:objects r1 r2 r3 - room)
 (:init (at r1))
 (:goal (and (at r3)))
)
"""

_TRAJECTORY_A = """(
(:init (at r1))
(operator: (move r1 r2))
(:state (at r2) (visited r2))
(operator: (move r2 r3))
(:state (at r3) (visited r2) (visited r3))
)
"""

_TRAJECTORY_B = """(
(:init (at r1))
(operator: (move r1 r3))
(:state (at r3) (visited r3))
(operator: (move r3 r2))
(:state (at r2) (visited r2) (visited r3))
)
"""

_SEED = 12345
_EPOCHS = 5


# ---------------------------------------------------------------- fixtures

def _write_world(root: Path) -> Tuple[Path, List[Path]]:
    """Materialise the micro domain plus two traces, and return their paths."""
    domain_path = root / "domain.pddl"
    domain_path.write_text(_DOMAIN)
    (root / "problem.pddl").write_text(_PROBLEM)

    traj_paths = []
    for name, text in (("a", _TRAJECTORY_A), ("b", _TRAJECTORY_B)):
        path = root / f"trace_{name}.trajectory"
        path.write_text(text)
        traj_paths.append(path)
    return domain_path, traj_paths


def _prepare(root: Path) -> Tuple[Path, List[Tuple[object, object]]]:
    """Parse the micro world into the ``(problem, observation)`` pairs the loops take."""
    domain_path, traj_paths = _write_world(root)
    partial_domain = DomainParser(domain_path, partial_parsing=True).parse_domain()
    problem = ProblemParser(root / "problem.pddl", partial_domain).parse_problem()

    prepared = []
    for traj_path in traj_paths:
        observation = TrajectoryParser(partial_domain, problem).parse_trajectory(traj_path)
        prepared.append((problem, observation))
    return domain_path, prepared


def _seeded_runner(domain_path: Path) -> PORosame_Runner:
    """A runner whose schema MLPs are initialised from a fixed seed."""
    torch.manual_seed(_SEED)
    return PORosame_Runner(str(domain_path))


# ------------------------------------------------------------------- tests

def test_local_loop_matches_vendored():
    """The rewritten per-trajectory loop trains identically to ``learn_rosame``.

    Same seed, same traces, same epoch count — so any divergence in batching,
    optimizer construction or iteration order would show up in the final model.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        domain_path, prepared = _prepare(root)

        vendored = _seeded_runner(domain_path)
        for problem, observation in prepared:
            vendored.add_problem(problem)
            vendored.ground_new_trajectory()
            vendored.learn_rosame(observation, _EPOCHS)
        vendored_pddl = vendored.rosame_to_pddl()

        local = _seeded_runner(domain_path)
        local.learn_per_trajectory(prepared, _EPOCHS)
        local_pddl = local.rosame_to_pddl()

    assert local_pddl == vendored_pddl, (
        "local learn_per_trajectory diverged from the vendored learn_rosame"
    )
    print("PASS  local per-trajectory loop matches the vendored learn_rosame")


def test_snapshots_do_not_perturb_training():
    """Observing the run does not change what the run produces.

    Snapshotting calls the model forward mid-training. If that consumed RNG or
    touched grounding state, the instrumented curve would describe a different
    run from the one whose final number gets reported.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        domain_path, prepared = _prepare(root)

        plain = _seeded_runner(domain_path)
        plain_pddl = plain.learn_full(prepared, train_per_trajectory=True, epochs=_EPOCHS)

        watched = _seeded_runner(domain_path)
        writer = SnapshotWriter(root / "snaps", interval=1)
        watched_pddl = watched.learn_full(
            prepared, train_per_trajectory=True, epochs=_EPOCHS, snapshot=writer
        )

        assert plain_pddl == watched_pddl, "snapshotting changed the learned model"

        # Two traces x _EPOCHS epochs, one snapshot each.
        snaps = sorted((root / "snaps").glob("snapshot_*.pddl"))
        assert len(snaps) == 2 * _EPOCHS, f"expected {2 * _EPOCHS} snapshots, got {len(snaps)}"
        # The last snapshot is taken after the final step, so it is the final model.
        assert snaps[-1].read_text() == watched_pddl, "last snapshot is not the final model"

    print("PASS  snapshots leave the run unchanged and end on the final model")


def test_snapshot_interval_and_index():
    """``interval`` thins the captures, and the index records where each landed."""
    import json

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        domain_path, prepared = _prepare(root)

        runner = _seeded_runner(domain_path)
        writer = SnapshotWriter(root / "snaps", interval=2)
        runner.learn_full(
            prepared, train_per_trajectory=True, epochs=_EPOCHS, snapshot=writer
        )

        payload = json.loads((root / "snaps" / SnapshotWriter.INDEX_NAME).read_text())

    total_steps = 2 * _EPOCHS
    assert payload["num_snapshots"] == total_steps // 2, payload["num_snapshots"]
    assert [s["step"] for s in payload["snapshots"]] == list(range(2, total_steps + 1, 2))
    # Trajectory attribution: the first trace owns the first _EPOCHS steps.
    for snap in payload["snapshots"]:
        expected = 0 if snap["step"] <= _EPOCHS else 1
        assert snap["trajectory"] == expected, snap
    print("PASS  interval thins captures and the index attributes them correctly")


def test_elapsed_excludes_snapshot_overhead():
    """The reported clock is training time, not training-plus-instrumentation.

    Snapshotting costs 12-17% of an epoch on real folds, so charging it to the
    learner would move the anytime curve by more than the gaps it exists to
    show. A fake clock makes the subtraction checkable exactly.
    """
    ticks = iter([
        0.0,     # start()
        10.0,    # elapsed for capture #1
        10.0,    # t_before write #1
        13.0,    # after write #1  -> overhead 3.0
        20.0,    # elapsed for capture #2 -> 20 - 3 = 17.0
        20.0,    # t_before write #2
        24.0,    # after write #2  -> overhead 7.0
        30.0,    # bare elapsed_seconds() -> 30 - 7 = 23.0
    ])

    with tempfile.TemporaryDirectory() as tmp:
        writer = SnapshotWriter(Path(tmp) / "snaps", interval=1, clock=lambda: next(ticks))
        writer.start()
        first = writer.capture(step=1, trajectory=0, epoch=0, render=lambda: "(model 1)")
        second = writer.capture(step=2, trajectory=0, epoch=1, render=lambda: "(model 2)")

        assert first.elapsed_seconds == 10.0, first
        assert second.elapsed_seconds == 17.0, second
        assert writer.overhead_seconds == 7.0, writer.overhead_seconds
        assert writer.elapsed_seconds() == 23.0

    print("PASS  elapsed_seconds subtracts snapshot overhead")


if __name__ == "__main__":
    test_elapsed_excludes_snapshot_overhead()
    test_local_loop_matches_vendored()
    test_snapshots_do_not_perturb_training()
    test_snapshot_interval_and_index()
    print("ALL TESTS PASSED")
