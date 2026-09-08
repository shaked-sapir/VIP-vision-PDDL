"""combine_dashboard_reports reads the same data-size blocks the dashboard builder does."""

from pathlib import Path

from benchmark.evaluation.cfm.combine_dashboard_reports import _discover_experiments


def _tree(root: Path) -> None:
    for cell in ("run__mask=0.0__noise=0.0", "run__mask=0.1__noise=0.2", "big__mask=0.0__noise=0.0"):
        (root / "results" / "depot" / cell).mkdir(parents=True)
    (root / "img" / "depot").mkdir(parents=True)


def test_nested_config_resolves_small_and_large(tmp_path: Path) -> None:
    _tree(tmp_path)
    cfg = {
        "results_root": "results", "domains": ["depot"],
        "simulation": {"small": {"prefix": {"depot": "run"}}, "large": {"prefix": {"depot": "big"}}},
        "image": {"small": {"experiment_dir": {"depot": "img/depot"}}, "large": {"experiment_dir": {}}},
    }
    sim, img = _discover_experiments(cfg, tmp_path, "small")
    assert sorted(c["path"].name for c in sim) == ["run__mask=0.0__noise=0.0", "run__mask=0.1__noise=0.2"]
    assert [i["path"] for i in img] == [tmp_path / "img/depot"]
    sim_large, img_large = _discover_experiments(cfg, tmp_path, "large")
    assert [c["path"].name for c in sim_large] == ["big__mask=0.0__noise=0.0"]
    assert img_large == []


def test_flat_pre_split_config_still_reads_as_small(tmp_path: Path) -> None:
    _tree(tmp_path)
    cfg = {"results_root": "results", "domains": ["depot"],
           "simulation": {"prefix": {"depot": "run"}}, "image": {"experiment_dir": {"depot": "img/depot"}}}
    sim, img = _discover_experiments(cfg, tmp_path, "small")
    assert len(sim) == 2 and len(img) == 1
    assert _discover_experiments(cfg, tmp_path, "large") == ([], [])
