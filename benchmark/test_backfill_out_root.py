"""With an output root the baseline backfill only reads the source cell.

    python -m pytest benchmark/test_backfill_out_root.py
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

from benchmark.backfill_baseline import backfill_cell


class _StubRunner:
    name = "STUB"

    def row_name(self, domain_path: Path) -> str:
        return self.name


def _packed_cell(tmp_path: Path) -> Path:
    cell = tmp_path / "results" / "dom" / "exp" / "testing" / "fold0_numtrajs1_gtrate0"
    cell.mkdir(parents=True)
    (cell / "domain_reference.pddl").write_text("(define (domain d))")
    (cell / "fold_info.json").write_text(json.dumps({
        "trajectories": [{"problem": "problem1"}], "test_problems": ["problem1"],
    }))
    observations = tmp_path / "original_observations"
    observations.mkdir()
    (observations / "original_observation_problem1.trajectory").write_text("(\n(:init )\n)")
    (observations / "original_observation_problem1.masking_info").write_text("\n")
    with tarfile.open(cell / "original_observations.tar.gz", "w:gz") as archive:
        archive.add(observations, arcname="original_observations")
    return cell


def _problem_dir(tmp_path: Path) -> Path:
    problem = tmp_path / "data" / "problem1"
    problem.mkdir(parents=True)
    (problem / "problem1.pddl").write_text("(define (problem problem1))")
    return tmp_path / "data"


def test_a_dry_run_reads_packed_observations_and_writes_nothing(tmp_path, capsys):
    cell = _packed_cell(tmp_path)
    before = sorted(p.name for p in cell.iterdir())
    out_root = tmp_path / "out"

    status = backfill_cell(
        cell, _problem_dir(tmp_path), "dom", [_StubRunner()],
        planning_timeout=1, learn_timeout=1, force=False, dry_run=True, out_root=out_root,
    )

    assert status == "dry"
    assert "would run [STUB] on 1 trajectories" in capsys.readouterr().out
    assert sorted(p.name for p in cell.iterdir()) == before
    assert not out_root.exists()


def test_a_row_already_in_the_output_cell_is_skipped(tmp_path):
    cell = _packed_cell(tmp_path)
    out_root = tmp_path / "out"
    out_cell = out_root / "dom" / "exp" / "testing" / cell.name
    out_cell.mkdir(parents=True)
    (out_cell / "fold_result.json").write_text(json.dumps([{"algorithm": "STUB"}]))

    status = backfill_cell(
        cell, _problem_dir(tmp_path), "dom", [_StubRunner()],
        planning_timeout=1, learn_timeout=1, force=False, dry_run=True, out_root=out_root,
    )

    assert status == "skip"


def _staged_cell(tmp_path: Path, cell: Path) -> Path:
    """A packed no-clean-state copy of ``cell``'s one observation, under its own root."""
    staged_root = tmp_path / "staged"
    staged_cell = staged_root / "dom" / "exp" / "testing" / cell.name
    staged_cell.mkdir(parents=True)
    observations = tmp_path / "observations_noisy_s0"
    observations.mkdir()
    (observations / "original_observation_problem1.trajectory").write_text("(\n(:init )\n)")
    (observations / "original_observation_problem1.masking_info").write_text("\n")
    with tarfile.open(staged_cell / "observations_noisy_s0.tar.gz", "w:gz") as archive:
        archive.add(observations, arcname="observations_noisy_s0")
    return staged_root


def test_staged_observations_are_read_and_the_row_is_tagged(tmp_path, capsys):
    cell = _packed_cell(tmp_path)
    staged_root = _staged_cell(tmp_path, cell)

    status = backfill_cell(
        cell, _problem_dir(tmp_path), "dom", [_StubRunner()],
        planning_timeout=1, learn_timeout=1, force=False, dry_run=True,
        out_root=tmp_path / "out", staged_root=staged_root,
    )

    assert status == "dry"
    printed = capsys.readouterr().out
    assert "would run [STUB__s0=noisy] on 1 trajectories staged under" in printed


def test_a_cell_with_no_staged_observations_is_skipped(tmp_path):
    cell = _packed_cell(tmp_path)
    empty_staged_root = tmp_path / "staged"
    empty_staged_root.mkdir()

    status = backfill_cell(
        cell, _problem_dir(tmp_path), "dom", [_StubRunner()],
        planning_timeout=1, learn_timeout=1, force=False, dry_run=True,
        out_root=tmp_path / "out", staged_root=empty_staged_root,
    )

    assert status == "skip"


def test_a_tagged_row_already_written_is_skipped(tmp_path):
    cell = _packed_cell(tmp_path)
    staged_root = _staged_cell(tmp_path, cell)
    out_root = tmp_path / "out"
    out_cell = out_root / "dom" / "exp" / "testing" / cell.name
    out_cell.mkdir(parents=True)
    (out_cell / "fold_result.json").write_text(json.dumps([{"algorithm": "STUB__s0=noisy"}]))

    status = backfill_cell(
        cell, _problem_dir(tmp_path), "dom", [_StubRunner()],
        planning_timeout=1, learn_timeout=1, force=False, dry_run=True,
        out_root=out_root, staged_root=staged_root,
    )

    assert status == "skip"
