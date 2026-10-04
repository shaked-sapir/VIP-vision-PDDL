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
