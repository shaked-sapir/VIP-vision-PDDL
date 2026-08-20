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
    domain_filename_label,
    is_purged,
    main,
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
    ("ROSAME", "ROSAME_24"),
    ("ROSAME_MILP", "ROSAME_MILP_24"),
    ("ROSAME_MILP_TAG", "ROSAME_MILP_24_TAG"),
    ("ROSAME-I", "ROSAME-I_24"),
    ("ROSAME-I_MILP", "ROSAME-I_MILP_24"),
    ("ROSAME-I__res=64x64", "ROSAME-I_24__res=64x64"),
])
def test_labels_are_renamed_suffix_and_all(old, new):
    assert rename_suffixed(old, LABEL_RENAMES) == new


@pytest.mark.parametrize("name", [
    "CDPS",
    "CDPS_ANCHORED",
    "PISAM_MILP_SR",
    "ROSAME_24",
    "ROSAME_MILP_24",
    "ROSAME-I_24",
])
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


# ---------------------------------------------------------------------------
# Purged labels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["ROSAME_MILP_BASE", "ROSAME_MILP_BASE__seed=7"])
def test_purged_labels_are_recognised(name):
    assert is_purged(name)


@pytest.mark.parametrize("name", ["ROSAME_MILP", "ROSAME_MILP_24", "CDPS", None, 3])
def test_surviving_labels_are_not_purged(name):
    assert not is_purged(name)


def test_a_purged_label_is_never_also_renamed():
    """A label cannot be both deleted and rewritten."""
    assert rename_suffixed("ROSAME_MILP_BASE", LABEL_RENAMES) is None


# ---------------------------------------------------------------------------
# Domain filenames
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, label", [
    ("learned_domain_CDPS_MILP_SR.pddl", "CDPS_MILP_SR"),
    ("learned_domain_CDPS_MILP_SR__eq16=0.4.pddl", "CDPS_MILP_SR__eq16=0.4"),
    ("learned_domain_ROSAME.pddl", "ROSAME"),
])
def test_domain_filename_labels_are_extracted(name, label):
    assert domain_filename_label(name) == label


@pytest.mark.parametrize("name", [
    "CDPS_MILP_SR.pddl",
    "learned_domain_CDPS_MILP_SR.txt",
    "fold_result.json",
])
def test_non_domain_filenames_have_no_label(name):
    assert domain_filename_label(name) is None


# ---------------------------------------------------------------------------
# Tree migration
# ---------------------------------------------------------------------------

_ROWS = [
    "CDPS",
    "CDPS_MILP_SR",
    "CDPS_MILP_SR__eq16=0.4",
    "CDPS_MILP_LOOP",
    "ROSAME",
    "ROSAME_MILP",
    "ROSAME_MILP_BASE",
    "ROSAME-I__res=64x64",
]
_ARM_KEYS = [
    "cdps",
    "cdps_milp_single_round",
    "cdps_milp_single_round__eq16=0.4",
    "cdps_milp_loop",
]
_BASELINE_LABELS = ["ROSAME", "ROSAME_MILP", "ROSAME_MILP_BASE", "ROSAME-I__res=64x64"]


def _make_tree(root: Path) -> Path:
    """A miniature result tree in the pre-rename shape."""
    fold = root / "blocksworld" / "exp" / "testing" / "fold0_numtrajs3_gtrate0"
    fold.mkdir(parents=True)

    (fold / "fold_result.json").write_text(json.dumps(
        [{"algorithm": label, "precision": 0.5} for label in _ROWS], indent=2
    ))

    for label in _ROWS:
        (fold / f"learned_domain_{label}.pddl").write_text(f"(define {label})")

    for key in _ARM_KEYS:
        arm = fold / key
        arm.mkdir()
        (arm / "milp_repair_log.json").write_text("{}")

    for label in _BASELINE_LABELS:
        arm = fold / "baseline_models" / label
        arm.mkdir(parents=True)
        (arm / "model.pddl").write_text(f"(define {label})")

    params = root / "blocksworld" / "exp" / "evaluation_results"
    params.mkdir(parents=True)
    (params / "run_params.json").write_text(json.dumps({
        "domain_key": "blocksworld",
        "run_cdps": True,
        "run_cdps_milp": False,
        "run_cdps_milp_loop": True,
        "algorithms": ["CDPS", "CDPS_MILP_LOOP", "ROSAME", "ROSAME_MILP_BASE"],
    }, indent=2))
    return fold


def test_migration_renames_labels_files_and_dirs(tmp_path):
    fold = _make_tree(tmp_path)
    apply_plan(build_plan(tmp_path))

    rows = json.loads((fold / "fold_result.json").read_text())
    assert [r["algorithm"] for r in rows] == [
        "CDPS", "PISAM_MILP_SR", "PISAM_MILP_SR__eq16=0.4", "PISAM_MILP_LOOP",
        "ROSAME_24", "ROSAME_MILP_24", "ROSAME-I_24__res=64x64",
    ]

    names = {p.name for p in fold.iterdir()}
    assert "learned_domain_PISAM_MILP_SR.pddl" in names
    assert "learned_domain_PISAM_MILP_SR__eq16=0.4.pddl" in names
    assert "learned_domain_PISAM_MILP_LOOP.pddl" in names
    assert "learned_domain_CDPS.pddl" in names
    assert "learned_domain_ROSAME_24.pddl" in names
    assert "learned_domain_ROSAME-I_24__res=64x64.pddl" in names
    assert not any("CDPS_MILP" in n for n in names)

    assert (fold / "pisam_milp_single_round" / "milp_repair_log.json").exists()
    assert (fold / "pisam_milp_single_round__eq16=0.4").is_dir()
    assert (fold / "pisam_milp_loop").is_dir()
    assert (fold / "cdps").is_dir()


def test_migration_renames_baseline_directories_by_label(tmp_path):
    """Baseline directories carry the row label, not the registry key."""
    fold = _make_tree(tmp_path)
    apply_plan(build_plan(tmp_path))

    baselines = {p.name for p in (fold / "baseline_models").iterdir()}
    assert baselines == {"ROSAME_24", "ROSAME_MILP_24", "ROSAME-I_24__res=64x64"}
    assert (fold / "baseline_models" / "ROSAME_24" / "model.pddl").read_text() == (
        "(define ROSAME)"
    )


def test_migration_purges_the_retired_arm(tmp_path):
    fold = _make_tree(tmp_path)
    apply_plan(build_plan(tmp_path))

    survivors = [str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")]
    assert not any("ROSAME_MILP_BASE" in p for p in survivors)

    rows = json.loads((fold / "fold_result.json").read_text())
    assert "ROSAME_MILP_BASE" not in {r["algorithm"] for r in rows}


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
    assert params["algorithms"] == ["CDPS", "PISAM_MILP_LOOP", "ROSAME_24"]


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


def test_half_migrated_baseline_directories_abort(tmp_path):
    fold = _make_tree(tmp_path)
    (fold / "baseline_models" / "ROSAME_24").mkdir()

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


def test_dry_run_reports_the_deletions(tmp_path, capsys):
    """The destructive half of the plan has to be visible before it runs."""
    _make_tree(tmp_path)
    assert main([str(tmp_path), "--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "DROPPED labels:         2" in out
    assert "DELETED paths:          2" in out
    assert "ROSAME_MILP_BASE" in out


def test_main_migrates_and_then_reports_nothing_to_do(tmp_path, capsys):
    _make_tree(tmp_path)
    assert main([str(tmp_path)]) == 0
    capsys.readouterr()

    assert main([str(tmp_path)]) == 0
    assert "nothing to migrate" in capsys.readouterr().out
