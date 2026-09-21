"""The repair tool re-scores only unclassified records, from the models on disk.

    python -m pytest benchmark/test_repair_solving_metrics.py
"""

from __future__ import annotations

import json

import benchmark.repair_solving_metrics as repair

ZEROS = {"solving_ratio": 0.0, "false_plans_ratio": 0.0, "unsolvable_ratio": 0.0,
         "planning_timed_out_ratio": 0.0}
FIXED = {"solving_ratio": 0.0, "false_plans_ratio": 1.0, "unsolvable_ratio": 0.0,
         "planning_timed_out_ratio": 0.0, "planning_syntax_error_ratio": 0.0,
         "planning_error_ratio": 0.0, "planning_dropped_operators": "move"}


def _cell(tmp_path):
    data = tmp_path / "data"
    (data / "training" / "trajectories" / "problem1").mkdir(parents=True)
    (data / "training" / "trajectories" / "problem1" / "problem1.pddl").write_text("(define)")
    cell = tmp_path / "exp" / "testing" / "fold0_numtrajs3_gtrate0"
    cell.mkdir(parents=True)
    (cell / "fold_info.json").write_text(json.dumps({"test_problems": ["problem1"]}))
    (cell / "domain_reference.pddl").write_text("(define (domain d))")
    rows = [
        {"algorithm": "CDPS", "precision_overall": 0.9, **ZEROS},
        {"algorithm": "ROSAME_24", "precision_overall": 0.5, **{**ZEROS, "solving_ratio": 1.0}},
    ]
    (cell / "fold_result.json").write_text(json.dumps(rows))
    (cell / "learned_domain_CDPS.pddl").write_text("(define (domain d))")
    (cell / "learned_domain_ROSAME_24.pddl").write_text("(define (domain d))")
    models = cell / "conflict_free_models"
    (models / "conflict_free_model_0").mkdir(parents=True)
    (models / "conflict_free_model_0" / "model.pddl").write_text("(define (domain d))")
    (models / "final_model").mkdir()
    (models / "final_model" / "model.pddl").write_text("(define (domain d))")
    (cell / "all_solutions_metrics.json").write_text(json.dumps([
        {"solution_index": 0, **ZEROS},
        {"solution_index": -1, **{**ZEROS, "unsolvable_ratio": 1.0}},
    ]))
    return cell, data


def test_is_unclassified_needs_every_outcome_at_zero():
    assert repair.is_unclassified(ZEROS)
    assert not repair.is_unclassified({**ZEROS, "unsolvable_ratio": 1.0})
    assert not repair.is_unclassified({**ZEROS, "solving_ratio": None})


def test_only_unclassified_records_are_rescored_and_other_fields_survive(tmp_path, monkeypatch):
    cell, data = _cell(tmp_path)
    calls = []
    monkeypatch.setattr(repair, "solving_metrics",
                        lambda model, ref, problems, timeout=60: calls.append(model.name) or FIXED)

    counts = repair.repair_cell(cell, data, timeout=5, dry_run=False)

    assert counts == {"rows": 1, "rows_fixed": 1, "solutions": 1, "solutions_fixed": 1, "no_model": 0}
    assert sorted(calls) == ["learned_domain_CDPS.pddl", "model.pddl"]
    rows = {r["algorithm"]: r for r in json.loads((cell / "fold_result.json").read_text())}
    assert rows["CDPS"]["false_plans_ratio"] == 1.0
    assert rows["CDPS"]["planning_dropped_operators"] == "move"
    assert rows["CDPS"]["precision_overall"] == 0.9
    assert rows["ROSAME_24"]["solving_ratio"] == 1.0 and "planning_error_ratio" not in rows["ROSAME_24"]
    solutions = json.loads((cell / "all_solutions_metrics.json").read_text())
    assert solutions[0]["false_plans_ratio"] == 1.0
    assert solutions[1] == {"solution_index": -1, **{**ZEROS, "unsolvable_ratio": 1.0}}


def test_a_dry_run_counts_and_writes_nothing(tmp_path, monkeypatch):
    cell, data = _cell(tmp_path)
    before = (cell / "fold_result.json").read_text()
    monkeypatch.setattr(repair, "solving_metrics", lambda *a, **k: 1 / 0)
    counts = repair.repair_cell(cell, data, timeout=5, dry_run=True)
    assert counts["rows"] == 1 and counts["solutions"] == 1 and counts["rows_fixed"] == 0
    assert (cell / "fold_result.json").read_text() == before
