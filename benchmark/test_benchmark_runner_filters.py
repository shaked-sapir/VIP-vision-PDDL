"""Tests for the cell selectors a SLURM array uses to run one cell per job.

    python -m pytest benchmark/test_benchmark_runner_filters.py
"""

import pytest

from benchmark.benchmark_runner import _expand_cells, _filter_grid_points

CONFIG = {
    "source": "simulated",
    "simulation": {"grid": {"masking_ps": [0.0, 0.01, 0.1],
                            "noising_ps": [0.0, 0.1, 0.2]}},
    "domains": [
        {"domain_key": "blocksworld", "data_dir": "d/blocks"},
        {"domain_key": "hanoi", "data_dir": "d/hanoi"},
    ],
}


class TestUnfilteredIsUnchanged:
    """Omitting every selector must behave exactly as before they existed."""

    def test_expands_the_whole_grid(self):
        assert len(_expand_cells(CONFIG)) == 2 * 3 * 3

    def test_explicit_none_matches_omission(self):
        assert _expand_cells(CONFIG, None, None, None) == _expand_cells(CONFIG)


class TestSelectors:
    def test_one_selector_narrows_one_axis(self):
        cells = _expand_cells(CONFIG, selected_masks=[0.1])
        assert len(cells) == 2 * 1 * 3
        assert {c["masking_p"] for c in cells} == {0.1}

    def test_all_three_select_a_single_cell(self):
        cells = _expand_cells(
            CONFIG, ["hanoi"], selected_masks=[0.01], selected_noises=[0.2]
        )
        assert len(cells) == 1
        assert cells[0]["domain_key"] == "hanoi"
        assert cells[0]["masking_p"] == 0.01
        assert cells[0]["noising_p"] == 0.2

    def test_the_45_job_shape(self):
        """One job per (domain, mask, noise); L stays inside the job."""
        seen = [
            _expand_cells(CONFIG, [d], [m], [n])
            for d in ("blocksworld", "hanoi")
            for m in (0.0, 0.01, 0.1)
            for n in (0.0, 0.1, 0.2)
        ]
        assert all(len(c) == 1 for c in seen)
        assert len(seen) == 18  # 2 domains here; 5 domains -> 45


class TestUnknownValuesFail:
    """A typo must fail at submit time, not silently run the whole sweep."""

    def test_unknown_mask_raises(self):
        with pytest.raises(ValueError, match="not in the config"):
            _expand_cells(CONFIG, selected_masks=[0.5])

    def test_unknown_noise_raises(self):
        with pytest.raises(ValueError, match="not in the config"):
            _expand_cells(CONFIG, selected_noises=[0.9])


def test_float_comparison_tolerates_csv_round_tripping():
    """A scheduler hands back "0.01" as text; the config holds 0.01."""
    assert _filter_grid_points([0.0, 0.01, 0.1], [float("0.01")], "mask") == [0.01]
