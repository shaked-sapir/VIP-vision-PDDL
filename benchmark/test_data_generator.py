"""Tests for `data_generator`'s banner helpers."""

from benchmark.data_generator import _describe_lengths, _describe_problem_count


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


class TestDescribeLengths:
    """`_describe_lengths` — whichever length setting the cut mode reads."""

    def test_uniform_names_the_range(self):
        assert _describe_lengths("uniform", (9, 20), None) == "length 9-20"

    def test_buckets_names_the_buckets(self):
        assert _describe_lengths("buckets", None, [4, 6]) == "buckets [4, 6]"

    def test_none_names_neither(self):
        assert _describe_lengths("none", None, None) == (
            "the whole trace as one problem (none)"
        )
