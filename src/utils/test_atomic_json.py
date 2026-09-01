"""Tests for crash- and concurrency-safe JSON writes."""

import json
from pathlib import Path

from src.utils.atomic_json import read_json_or_none, write_json_atomic


class TestWriteJsonAtomic:
    def test_round_trips(self, tmp_path):
        p = tmp_path / "run_params.json"
        write_json_atomic(p, {"a": 1, "b": [1, 2]})
        assert json.loads(p.read_text()) == {"a": 1, "b": [1, 2]}

    def test_creates_missing_parents(self, tmp_path):
        p = tmp_path / "deep" / "nested" / "x.json"
        write_json_atomic(p, [1])
        assert p.is_file()

    def test_leaves_no_temp_file(self, tmp_path):
        p = tmp_path / "x.json"
        write_json_atomic(p, {"k": "v"})
        assert list(tmp_path.glob("*.tmp*")) == []

    def test_overwrites_in_place(self, tmp_path):
        p = tmp_path / "x.json"
        write_json_atomic(p, {"v": 1})
        write_json_atomic(p, {"v": 2})
        assert json.loads(p.read_text()) == {"v": 2}


class TestReadJsonOrNone:
    def test_absent_is_none(self, tmp_path):
        assert read_json_or_none(tmp_path / "nope.json") is None

    def test_empty_is_none(self, tmp_path):
        """A killed writer leaves a zero-byte file; resume must not die on it."""
        p = tmp_path / "x.json"
        p.write_text("")
        assert read_json_or_none(p) is None

    def test_truncated_is_none(self, tmp_path):
        p = tmp_path / "x.json"
        p.write_text('{"a": 1')
        assert read_json_or_none(p) is None

    def test_directory_is_none(self, tmp_path):
        assert read_json_or_none(tmp_path) is None

    def test_valid_parses(self, tmp_path):
        p = tmp_path / "x.json"
        write_json_atomic(p, {"a": 1})
        assert read_json_or_none(p) == {"a": 1}
