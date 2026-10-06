"""Series selection in the dashboard builder: exclusions by data size, the PI-SAM row.

    python -m pytest benchmark/evaluation/cfm/test_build_dashboard_series.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.evaluation.cfm.build_dashboard import _baseline_rows, exclusions_by_size


class TestExclusions:
    def test_a_list_applies_to_both_sizes(self):
        assert exclusions_by_size(["A", "B"]) == {"all": ["A", "B"], "small": [], "large": []}

    def test_a_mapping_is_kept_per_size(self):
        out = exclusions_by_size({"all": ["A"], "large": ["NOLAM"]})
        assert out == {"all": ["A"], "small": [], "large": ["NOLAM"]}

    def test_nothing_configured(self):
        assert exclusions_by_size(None) == {"all": [], "small": [], "large": []}

    def test_an_unknown_size_is_refused(self):
        with pytest.raises(ValueError):
            exclusions_by_size({"medium": ["A"]})


class TestPisamRow:
    def test_the_baseline_row_is_a_series_and_the_legacy_internal_row_is_not(self, tmp_path: Path):
        (tmp_path / "fold_result.json").write_text(json.dumps([
            {"algorithm": "PISAM", "solving_ratio": 0.5},
            {"algorithm": "CDPS", "solving_ratio": 0.9},
        ]))
        assert _baseline_rows(tmp_path, ["solving_ratio"]) == {"PISAM": {"solving_ratio": 0.5}}

        (tmp_path / "fold_result.json").write_text(json.dumps([
            {"algorithm": "PISAM", "solving_ratio": 0.5, "_internal_phase": "pre_patch"},
        ]))
        assert _baseline_rows(tmp_path, ["solving_ratio"]) == {}
