"""Tests for the pre-rename arm-name migration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.migrate_arm_names import (
    KEY_RENAMES,
    LABEL_RENAMES,
    MigrationConflict,
    apply_plan,
    build_plan,
    main,
    rename_domain_filename,
    rename_suffixed,
)


# ---------------------------------------------------------------------------
# Name rewriting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("old, new", [
    ("CDPS_MILP_SR", "PISAM_MILP_SR"),
    ("CDPS_MILP_LOOP", "PISAM_MILP_LOOP"),
    ("CDPS_MILP_SR__eq16=0.4", "PISAM_MILP_SR__eq16=0.4"),
    ("CDPS_MILP_LOOP__seed=7__eq16=0.4", "PISAM_MILP_LOOP__seed=7__eq16=0.4"),
])
def test_labels_are_renamed_suffix_and_all(old, new):
    assert rename_suffixed(old, LABEL_RENAMES) == new


@pytest.mark.parametrize("name", ["CDPS", "CDPS_ANCHORED", "ROSAME_MILP", "PISAM_MILP_SR"])
def test_other_labels_are_left_alone(name):
    assert rename_suffixed(name, LABEL_RENAMES) is None


def test_keys_are_renamed_with_their_suffix():
    assert rename_suffixed("cdps_milp_loop", KEY_RENAMES) == "pisam_milp_loop"
    assert rename_suffixed(
        "cdps_milp_single_round__eq16=0.4", KEY_RENAMES
    ) == "pisam_milp_single_round__eq16=0.4"


def test_cdps_directories_are_not_touched():
    assert rename_suffixed("cdps", KEY_RENAMES) is None
    assert rename_suffixed("cdps_anchored", KEY_RENAMES) is None


@pytest.mark.parametrize("old, new", [
    ("learned_domain_CDPS_MILP_SR.pddl", "learned_domain_PISAM_MILP_SR.pddl"),
    ("learned_domain_CDPS_MILP_SR__eq16=0.4.pddl",
     "learned_domain_PISAM_MILP_SR__eq16=0.4.pddl"),
])
def test_domain_filenames_are_renamed(old, new):
    assert rename_domain_filename(old) == new


@pytest.mark.parametrize("name", [
    "learned_domain_CDPS.pddl",
    "learned_domain_ROSAME.pddl",
    "CDPS_MILP_SR.pddl",
    "learned_domain_CDPS_MILP_SR.txt",
])
def test_other_filenames_are_left_alone(name):
    assert rename_domain_filename(name) is None


# ---------------------------------------------------------------------------
# Tree migration
# ---------------------------------------------------------------------------

def _make_tree(root: Path) -> Path:
    """A miniature result tree in the pre-rename shape."""
    fold = root / "blocksworld" / "exp" / "testing" / "fold0_numtrajs3_gtrate0"
    fold.mkdir(parents=True)

    (fold / "fold_result.json").write_text(json.dumps([
        {"algorithm": "CDPS", "precision": 1.0},
        {"algorithm": "CDPS_MILP_SR", "precision": 0.5},
        {"algorithm": "CDPS_MILP_SR__eq16=0.4", "precision": 0.6},
        {"algorithm": "CDPS_MILP_LOOP", "precision": 0.7},
        {"algorithm": "ROSAME_MILP", "precision": 0.2},
    ], indent=2))

    for label in ("CDPS", "CDPS_MILP_SR", "CDPS_MILP_SR__eq16=0.4", "CDPS_MILP_LOOP"):
        (fold / f"learned_domain_{label}.pddl").write_text(f"(define {label})")

    for key in ("cdps", "cdps_milp_single_round",
                "cdps_milp_single_round__eq16=0.4", "cdps_milp_loop"):
        arm = fold / key
        arm.mkdir()
        (arm / "milp_repair_log.json").write_text("{}")

    params = root / "blocksworld" / "exp" / "evaluation_results"
    params.mkdir(parents=True)
    (params / "run_params.json").write_text(json.dumps({
        "domain_key": "blocksworld",
        "run_cdps": True,
        "run_cdps_milp": False,
        "run_cdps_milp_loop": True,
        "algorithms": ["CDPS", "CDPS_MILP_LOOP", "ROSAME"],
    }, indent=2))
    return fold


def test_migration_renames_labels_files_and_dirs(tmp_path):
    fold = _make_tree(tmp_path)
    apply_plan(build_plan(tmp_path))

    rows = json.loads((fold / "fold_result.json").read_text())
    assert [r["algorithm"] for r in rows] == [
        "CDPS", "PISAM_MILP_SR", "PISAM_MILP_SR__eq16=0.4",
        "PISAM_MILP_LOOP", "ROSAME_MILP",
    ]

    names = {p.name for p in fold.iterdir()}
    assert "learned_domain_PISAM_MILP_SR.pddl" in names
    assert "learned_domain_PISAM_MILP_SR__eq16=0.4.pddl" in names
    assert "learned_domain_PISAM_MILP_LOOP.pddl" in names
    assert "learned_domain_CDPS.pddl" in names
    assert not any("CDPS_MILP" in n for n in names)

    assert (fold / "pisam_milp_single_round" / "milp_repair_log.json").exists()
    assert (fold / "pisam_milp_single_round__eq16=0.4").is_dir()
    assert (fold / "pisam_milp_loop").is_dir()
    assert (fold / "cdps").is_dir()


def test_migration_rewrites_run_params_keys_and_labels(tmp_path):
    _make_tree(tmp_path)
    apply_plan(build_plan(tmp_path))

    params = json.loads(
        (tmp_path / "blocksworld" / "exp" / "evaluation_results"
         / "run_params.json").read_text()
    )
    assert params["run_pisam_milp"] is False
    assert params["run_pisam_milp_loop"] is True
    assert "run_cdps_milp" not in params
    assert "run_cdps_milp_loop" not in params
    assert params["run_cdps"] is True
    assert params["algorithms"] == ["CDPS", "PISAM_MILP_LOOP", "ROSAME"]


def test_migration_preserves_pddl_contents(tmp_path):
    fold = _make_tree(tmp_path)
    apply_plan(build_plan(tmp_path))
    assert (fold / "learned_domain_PISAM_MILP_SR.pddl").read_text() == (
        "(define CDPS_MILP_SR)"
    )


def test_migration_is_idempotent(tmp_path):
    fold = _make_tree(tmp_path)
    apply_plan(build_plan(tmp_path))
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))

    second = build_plan(tmp_path)
    assert second.is_empty

    apply_plan(second)
    after = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    assert before == after
    rows = json.loads((fold / "fold_result.json").read_text())
    assert [r["algorithm"] for r in rows][1] == "PISAM_MILP_SR"


def test_half_migrated_rows_abort(tmp_path):
    fold = _make_tree(tmp_path)
    rows = json.loads((fold / "fold_result.json").read_text())
    rows.append({"algorithm": "PISAM_MILP_SR", "precision": 0.9})
    (fold / "fold_result.json").write_text(json.dumps(rows, indent=2))

    with pytest.raises(MigrationConflict, match="PISAM_MILP_SR"):
        build_plan(tmp_path)


def test_half_migrated_directories_abort(tmp_path):
    fold = _make_tree(tmp_path)
    (fold / "pisam_milp_loop").mkdir()

    with pytest.raises(MigrationConflict, match="already exists"):
        build_plan(tmp_path)


def test_half_migrated_run_params_abort(tmp_path):
    _make_tree(tmp_path)
    path = (tmp_path / "blocksworld" / "exp" / "evaluation_results" / "run_params.json")
    params = json.loads(path.read_text())
    params["run_pisam_milp"] = True
    path.write_text(json.dumps(params, indent=2))

    with pytest.raises(MigrationConflict, match="run_cdps_milp"):
        build_plan(tmp_path)


def test_a_conflict_writes_nothing(tmp_path):
    fold = _make_tree(tmp_path)
    (fold / "pisam_milp_loop").mkdir()
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))

    assert main([str(tmp_path)]) == 1
    after = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    assert before == after


def test_dry_run_writes_nothing(tmp_path):
    _make_tree(tmp_path)
    before = {
        str(p.relative_to(tmp_path)): (p.read_text() if p.is_file() else None)
        for p in tmp_path.rglob("*")
    }

    assert main([str(tmp_path), "--dry-run"]) == 0

    after = {
        str(p.relative_to(tmp_path)): (p.read_text() if p.is_file() else None)
        for p in tmp_path.rglob("*")
    }
    assert before == after


def test_main_migrates_and_then_reports_nothing_to_do(tmp_path, capsys):
    _make_tree(tmp_path)
    assert main([str(tmp_path)]) == 0
    capsys.readouterr()

    assert main([str(tmp_path)]) == 0
    assert "nothing to migrate" in capsys.readouterr().out
