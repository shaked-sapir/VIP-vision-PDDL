"""The fold loader hands each trajectory's problem file to the observation loader."""

from pathlib import Path

from benchmark.experiment_running_helpers import learning_helpers


def test_load_masked_observations_passes_each_tuples_problem_file(tmp_path: Path, monkeypatch):
    calls = []

    def fake_load(traj_path, masking_info_path, domain, problem_path=None):
        calls.append((Path(traj_path), Path(masking_info_path), problem_path))
        return f"obs:{Path(traj_path).stem}"

    monkeypatch.setattr(learning_helpers, "load_masked_observation", fake_load)

    prepared = []
    for name in ("problem3", "problem9"):
        traj = tmp_path / f"{name}_gtrate0_frame_axioms.trajectory"
        traj.write_text("(\n(:init)\n)\n")
        masking = traj.with_suffix(".masking_info")
        masking.write_text("\n")
        prepared.append((traj, masking, tmp_path / f"{name}.pddl", {0}))
    unmasked = tmp_path / "problem5_gtrate0_frame_axioms.trajectory"
    unmasked.write_text("(\n(:init)\n)\n")
    prepared.append((unmasked, unmasked.with_suffix(".masking_info"), tmp_path / "problem5.pddl", {0}))

    observations = learning_helpers._load_masked_observations("domain", prepared, None)

    assert observations == ["obs:problem3_gtrate0_frame_axioms", "obs:problem9_gtrate0_frame_axioms"]
    assert [c[2] for c in calls] == [tmp_path / "problem3.pddl", tmp_path / "problem9.pddl"]


def test_load_masked_observations_prefers_pre_built(monkeypatch):
    monkeypatch.setattr(
        learning_helpers, "load_masked_observation",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("disk loader must not run")),
    )
    prebuilt = ["obs-a", "obs-b"]
    assert learning_helpers._load_masked_observations("domain", [], prebuilt) is prebuilt
