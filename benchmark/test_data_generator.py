"""Tests for `data_generator`'s trace-mode helpers."""

import pytest

from pathlib import Path

from benchmark.data_generator import (
    _describe_cut,
    _describe_problem_count,
    _resolve_trace_domain_file,
    _trace_dir_name,
)


class TestDescribeProblemCount:
    """`_describe_problem_count` — the produced count, flagging a shortfall."""

    def test_shortfall_names_the_requested_count(self):
        assert _describe_problem_count(2, 10) == "2  *** SHORT: asked for 10 ***"

    def test_exact_match_is_the_bare_count(self):
        assert _describe_problem_count(10, 10) == "10"

    def test_overshoot_is_the_bare_count(self):
        assert _describe_problem_count(12, 10) == "12"

    def test_no_request_is_the_bare_count(self):
        assert _describe_problem_count(3, None) == "3"

    def test_zero_asked_none_is_the_bare_count(self):
        assert _describe_problem_count(0, None) == "0"


class TestDescribeCut:
    """`_describe_cut` — only the settings the cut mode actually reads."""

    def test_uniform_names_every_setting_it_reads(self):
        assert _describe_cut("uniform", 10, 2, (9, 20)) == (
            "uniform | problems 10 | skip 2 | length 9-20"
        )

    def test_none_names_only_the_mode(self):
        assert _describe_cut("none", None, 1, None) == (
            "none | the whole trace as one problem"
        )

    def test_none_advertises_neither_a_count_nor_a_skip_it_ignores(self):
        """NONE takes the whole trace, so a leftover count would be a lie."""
        described = _describe_cut("none", 10, 2, None)
        assert "10" not in described
        assert "skip" not in described


class TestTraceDirName:
    """`_trace_dir_name` — the directory name states what the mode read."""

    def test_uniform_carries_the_requested_count(self):
        assert _trace_dir_name("T", "problem", Path("/x/p0.pddl"), "uniform", 10) == (
            "trace_T__problem=p0__cut=uniform__n=10"
        )

    def test_none_omits_the_count_it_ignores(self):
        """NONE yields one window, so an n= segment would misname the corpus."""
        assert _trace_dir_name("T", "problem", Path("/x/p0.pddl"), "none", None) == (
            "trace_T__problem=p0__cut=none"
        )

    def test_trajectory_kind_names_the_trajectory_stem(self):
        assert _trace_dir_name("T", "trajectory", Path("/x/run.trajectory"),
                               "uniform", 3) == (
            "trace_T__trajectory=run__cut=uniform__n=3"
        )


class TestResolveTraceDomainFile:
    """`_resolve_trace_domain_file` — an explicit path, or the registry."""

    def test_an_explicit_file_is_taken_as_given(self, tmp_path):
        foreign = tmp_path / "foreign.pddl"
        foreign.write_text("(define (domain foreign))")
        assert _resolve_trace_domain_file(None, foreign) == foreign

    def test_an_explicit_file_wins_over_the_registry(self, tmp_path):
        foreign = tmp_path / "foreign.pddl"
        foreign.write_text("(define (domain foreign))")
        assert _resolve_trace_domain_file("blocksworld", foreign) == foreign

    def test_an_explicit_file_that_is_absent_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No domain file at"):
            _resolve_trace_domain_file(None, tmp_path / "nope.pddl")

    def test_a_registered_domain_resolves_from_config(self, monkeypatch, tmp_path):
        registered = tmp_path / "blocks.pddl"
        registered.write_text("(define (domain blocks))")
        monkeypatch.setattr(
            "benchmark.data_generator.load_config",
            lambda: {"domains": {"blocksworld": {"domain_file": registered}}})
        assert _resolve_trace_domain_file("blocksworld", None) == registered

    def test_a_registered_domain_whose_config_file_is_absent_is_refused(
            self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "benchmark.data_generator.load_config",
            lambda: {"domains":
                     {"blocksworld": {"domain_file": tmp_path / "gone.pddl"}}})
        with pytest.raises(FileNotFoundError, match="No domain file at"):
            _resolve_trace_domain_file("blocksworld", None)

    def test_an_unregistered_domain_without_a_file_is_refused(self):
        with pytest.raises(ValueError, match="is not registered"):
            _resolve_trace_domain_file("sokoban", None)

    def test_neither_a_domain_nor_a_file_is_refused(self):
        with pytest.raises(ValueError, match="needs a domain"):
            _resolve_trace_domain_file(None, None)
