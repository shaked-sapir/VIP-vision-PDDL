"""Tests for the anytime harness.

    python -m benchmark.evaluation.anytime.test_anytime

The reader tests build fold layouts from scratch in a temp dir rather than
pointing at a real run, because ``benchmark/running_results/`` is gitignored --
a test that needs a fold on disk is a test that only passes on one machine.

Two of these guard claims the module makes in prose and would otherwise have to
be taken on trust: that the single-round arm is still visible when it ends with
conflicts (:func:`test_final_model_fallback`), and that checkpoint density does
not inflate a curve (:func:`test_density_does_not_change_the_curve`). The second
is the whole basis for plotting arms with wildly different checkpoint counts on
one axis.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List

from benchmark.evaluation.anytime.checkpoints import (
    Checkpoint,
    read_fold_checkpoints,
)
from benchmark.evaluation.anytime.curves import running_best, step_series
from benchmark.evaluation.anytime.score import ScoredCheckpoint, _canonical_model_text


# ---------------------------------------------------------------- fixtures

def _write_cdps_arm(root: Path, entries: List[dict]) -> None:
    """A CDPS-shaped suite: a solutions log plus one model dir per solution."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "conflict_free_solutions_log.json").write_text(json.dumps(entries))
    for entry in entries:
        model_dir = root / "conflict_free_models" / f"conflict_free_model_{entry['index']}"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.pddl").write_text(f"(define (domain d{entry['index']}))")


def _write_final_model_arm(root: Path, wall_time: float) -> None:
    """What single_round writes when it ends with conflicts: empty log, final_model/."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "conflict_free_solutions_log.json").write_text("[]")
    model_dir = root / "conflict_free_models" / "final_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.pddl").write_text("(define (domain conflicted))")
    (model_dir / "patch_details.json").write_text(
        json.dumps({"wall_time_seconds": wall_time})
    )


def _write_loop_arm(root: Path, rounds: List[dict], skip_models: List[int] = ()) -> None:
    """A loop suite: the round log plus round_{i}/model.pddl for solved rounds."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "milp_loop_rounds.json").write_text(json.dumps({"rounds": rounds}))
    for entry in rounds:
        if entry["round"] in skip_models:
            continue
        model_dir = root / "milp_loop_round_models" / f"round_{entry['round']}"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.pddl").write_text(f"(define (domain r{entry['round']}))")


def _write_snapshot_arm(root: Path, arm: str, elapsed: List[float]) -> None:
    """A SnapshotWriter suite: snapshots.json plus one .pddl per record."""
    snap_dir = root / "anytime_snapshots" / arm
    snap_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, seconds in enumerate(elapsed):
        name = f"snapshot_{index:04d}.pddl"
        (snap_dir / name).write_text(f"(define (domain s{index}))")
        records.append({
            "index": index, "step": index + 1, "trajectory": 0, "epoch": index,
            "elapsed_seconds": seconds, "path": name,
        })
    (snap_dir / "snapshots.json").write_text(json.dumps({"snapshots": records}))


def _scored(times: List[float], rates: List[float]) -> List[ScoredCheckpoint]:
    """Scored checkpoints at the given times with the given success rates."""
    return [
        ScoredCheckpoint(
            Checkpoint("x", i, t, Path(f"/nonexistent/{i}.pddl")),
            success_rate=r, v_raw=None, v_per_transition=None, model_digest=f"d{i}",
        )
        for i, (t, r) in enumerate(zip(times, rates))
    ]


# ------------------------------------------------------------------- tests

def test_reads_every_arm_shape():
    """One fold holding all four layouts yields one stream per arm, time-sorted."""
    with tempfile.TemporaryDirectory() as tmp:
        fold = Path(tmp)
        _write_cdps_arm(fold, [
            {"index": 0, "wall_time_so_far": 3.4},
            {"index": 1, "wall_time_so_far": 5.1},
        ])
        _write_cdps_arm(fold / "cdps_anchored", [{"index": 0, "wall_time_so_far": 9.0}])
        _write_loop_arm(fold / "pisam_milp_loop", [
            {"round": 0, "elapsed_seconds": 2.0},
            {"round": 1, "elapsed_seconds": 7.5},
        ])
        _write_snapshot_arm(fold, "ROSAME_24", [0.4, 0.9, 1.3])

        found = read_fold_checkpoints(fold)

        # Inside the temp dir's lifetime: every checkpoint must point at a file
        # that is really there, since the scorer will go on to read it.
        for stream in found.values():
            assert all(c.model_path.exists() for c in stream), stream

    assert set(found) == {"cdps", "cdps_anchored", "pisam_milp_loop", "ROSAME_24"}, found
    assert [c.elapsed_seconds for c in found["cdps"]] == [3.4, 5.1]
    assert [c.elapsed_seconds for c in found["pisam_milp_loop"]] == [2.0, 7.5]
    assert len(found["ROSAME_24"]) == 3
    print("PASS  every arm's artifact shape is read into one checkpoint stream")


def test_final_model_fallback():
    """An arm that ended with conflicts still contributes its one model.

    single_round writes an *empty* solutions log in that case. Treating the empty
    log as "no checkpoints" would erase the arm from the plot precisely in the
    runs where it performed worst.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fold = Path(tmp)
        _write_final_model_arm(fold / "pisam_milp_single_round", wall_time=12.5)
        found = read_fold_checkpoints(fold, ["pisam_milp_single_round"])

    stream = found["pisam_milp_single_round"]
    assert len(stream) == 1, stream
    assert stream[0].elapsed_seconds == 12.5
    assert stream[0].model_path.parent.name == "final_model"
    print("PASS  a conflicted single_round still appears, via final_model/")


def test_unsolved_rounds_are_skipped_not_fatal():
    """A round that produced no model is absent from the stream, not an error."""
    with tempfile.TemporaryDirectory() as tmp:
        fold = Path(tmp)
        _write_loop_arm(
            fold / "pisam_milp_loop",
            [{"round": r, "elapsed_seconds": float(r)} for r in range(4)],
            skip_models=[1, 2],
        )
        found = read_fold_checkpoints(fold, ["pisam_milp_loop"])

    assert [c.index for c in found["pisam_milp_loop"]] == [0, 3]
    print("PASS  unsolved rounds drop out without taking the arm with them")


def test_arm_with_no_artifacts_is_omitted():
    """"Did not run here" is distinguishable from "ran and produced nothing"."""
    with tempfile.TemporaryDirectory() as tmp:
        fold = Path(tmp)
        _write_snapshot_arm(fold, "ROSAME_24", [1.0])
        found = read_fold_checkpoints(fold)

    assert set(found) == {"ROSAME_24"}, found
    print("PASS  arms that left no artifacts are omitted, not mapped to []")


def test_running_best_is_monotone_and_keeps_only_corners():
    """The profile never descends, and records improvements only."""
    points = running_best(_scored(
        times=[1.0, 2.0, 3.0, 4.0, 5.0],
        rates=[0.2, 0.1, 0.5, 0.5, 0.4],
    ))

    assert [(p.elapsed_seconds, p.best_success_rate) for p in points] == [
        (1.0, 0.2), (3.0, 0.5)
    ], points
    print("PASS  running best is monotone and keeps only the corners")


def test_unparseable_checkpoints_never_advance_the_best():
    """A model that would not parse cannot raise the curve."""
    scored = _scored([1.0, 2.0], [0.3, 0.3])
    broken = ScoredCheckpoint(
        Checkpoint("x", 2, 1.5, Path("/nonexistent/broken.pddl")),
        success_rate=None, v_raw=None, v_per_transition=None,
        model_digest="broken", error="unbalanced parens",
    )
    points = running_best([scored[0], broken, scored[1]])

    assert [p.elapsed_seconds for p in points] == [1.0], points
    print("PASS  unparseable checkpoints are carried but never advance the best")


def test_density_does_not_change_the_curve():
    """Measuring ten times as often does not improve the profile.

    This is the property that licenses plotting ROSAME's hundreds of snapshots
    against the loop's handful of rounds on one axis. The dense arm here sees the
    same models at the same times as the sparse one, plus repeats in between.
    """
    sparse = running_best(_scored([1.0, 5.0], [0.2, 0.6]))

    dense_times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    dense_rates = [0.2, 0.2, 0.2, 0.2, 0.6, 0.6, 0.6]
    dense = running_best(_scored(dense_times, dense_rates))

    assert [(p.elapsed_seconds, p.best_success_rate) for p in sparse] == \
           [(p.elapsed_seconds, p.best_success_rate) for p in dense]

    # And the drawn staircases agree wherever both are defined.
    horizon = 7.0
    assert step_series(sparse, horizon) == step_series(dense, horizon)
    print("PASS  checkpoint density leaves the curve identical")


def test_step_series_holds_flat_to_the_horizon():
    """An arm that stops improving keeps existing until the shared horizon."""
    xs, ys = step_series(running_best(_scored([1.0, 3.0], [0.2, 0.5])), end_seconds=10.0)

    assert xs == [1.0, 3.0, 3.0, 10.0], xs
    assert ys == [0.2, 0.2, 0.5, 0.5], ys
    print("PASS  the staircase is held flat to the shared horizon")


def test_requirements_line_does_not_split_identical_models():
    """Two renders of one model hash together despite reordered :requirements.

    ``LearnerDomain.to_pddl`` rebuilds that line from a set, so without this the
    dedup cache would miss most genuine repeats -- the loop's incumbent would
    look like a new model on every round that failed to improve.
    """
    a = "(define (domain d) (:requirements :strips :typing) (:action move))"
    b = "(define (domain d) (:requirements :typing :strips) (:action move))"
    different = "(define (domain d) (:requirements :strips) (:action push))"

    assert _canonical_model_text(a) == _canonical_model_text(b)
    assert _canonical_model_text(a) != _canonical_model_text(different)
    print("PASS  :requirements ordering does not split one model into two")


if __name__ == "__main__":
    test_reads_every_arm_shape()
    test_final_model_fallback()
    test_unsolved_rounds_are_skipped_not_fatal()
    test_arm_with_no_artifacts_is_omitted()
    test_running_best_is_monotone_and_keeps_only_corners()
    test_unparseable_checkpoints_never_advance_the_best()
    test_density_does_not_change_the_curve()
    test_step_series_holds_flat_to_the_horizon()
    test_requirements_line_does_not_split_identical_models()
    print("ALL TESTS PASSED")
