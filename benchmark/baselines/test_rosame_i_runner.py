"""Tests for the ROSAME-I baseline's row naming and per-domain resize.

The GT-anchor and input-dialect cases live in ``test_image_fold_inputs.py``,
which owns the fold walk they exercise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.baselines import rosame_i_runner
from benchmark.baselines.image_fold_inputs import BENCH_KEYS
from benchmark.baselines.rosame_i_milp_runner import RosameIMilpRunner
from benchmark.baselines.rosame_i_runner import (
    RosameIBaselineRunner,
    _HYPERPARAMS,
    _resize_tag,
)

_RESIZE_DOMAIN = """(define (domain {name})
  (:requirements :strips)
  (:predicates (clear ?d))
  (:action noop :parameters () :precondition () :effect ())
)
"""


def _domain_file(tmp_path: Path, name: str) -> Path:
    """A minimal parseable domain whose name maps to a bench key."""
    path = tmp_path / f"{name}.pddl"
    path.write_text(_RESIZE_DOMAIN.format(name=name))
    return path


class TestResizeRowName:
    """The suffix keys on the *effective* resize, not on whether one was passed.

    A per-domain table entry that diverges from the default must be labelled
    exactly as an explicit override is, or two resolutions get averaged under
    one row name.
    """

    def test_table_value_equal_to_the_default_gets_no_suffix(self, tmp_path: Path) -> None:
        runner = RosameIBaselineRunner()
        assert runner.row_name(_domain_file(tmp_path, "blocks")) == "ROSAME-I_24"

    def test_table_value_differing_from_the_default_is_suffixed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(rosame_i_runner._RESIZE, "depot", 224)
        runner = RosameIBaselineRunner()
        assert runner.row_name(_domain_file(tmp_path, "depot")) == "ROSAME-I_24__res=224"

    def test_an_override_equal_to_the_default_gets_no_suffix(self, tmp_path: Path) -> None:
        """Same operation, same name — it must overwrite, not sit beside."""
        runner = RosameIBaselineRunner(resize=64)
        assert runner.row_name(_domain_file(tmp_path, "blocks")) == "ROSAME-I_24"

    def test_a_forced_square_is_not_the_default(self, tmp_path: Path) -> None:
        """Resize((64,64)) distorts where Resize(64) preserves aspect."""
        runner = RosameIBaselineRunner(resize=[64, 64])
        assert runner.row_name(_domain_file(tmp_path, "blocks")) == "ROSAME-I_24__res=64x64"

    def test_native_is_suffixed(self, tmp_path: Path) -> None:
        runner = RosameIBaselineRunner(resize=None)
        assert runner.row_name(_domain_file(tmp_path, "blocks")) == "ROSAME-I_24__res=native"

    def test_a_domain_absent_from_the_table_falls_back_to_the_default(
        self, tmp_path: Path
    ) -> None:
        runner = RosameIBaselineRunner()
        assert runner.row_name(_domain_file(tmp_path, "hiking")) == "ROSAME-I_24"

    def test_name_is_the_stable_identity_without_a_resize(self) -> None:
        assert RosameIBaselineRunner(resize=None).name == "ROSAME-I_24"

    def test_the_label_and_the_run_resolve_the_same_bench(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """row_name and learn must agree, or a row is labelled at one
        resolution and trained at another."""
        monkeypatch.setitem(rosame_i_runner._RESIZE, "depot", 224)
        runner = RosameIBaselineRunner()
        domain_path = _domain_file(tmp_path, "depot")

        bench, _ = runner._bench_and_domain(domain_path)

        assert runner.row_name(domain_path).endswith(
            _resize_tag(runner._resolve_resize(bench))
        )
        assert runner._resolve_resize(bench) == 224


class TestResizeRowNameMilp:
    def test_default_is_unsuffixed(self, tmp_path: Path) -> None:
        assert RosameIMilpRunner().row_name(_domain_file(tmp_path, "blocks")) == (
            "ROSAME-I_MILP_24"
        )

    def test_off_default_is_suffixed(self, tmp_path: Path) -> None:
        runner = RosameIMilpRunner(resize=[64, 64])
        assert runner.row_name(_domain_file(tmp_path, "blocks")) == (
            "ROSAME-I_MILP_24__res=64x64"
        )

    def test_name_is_the_stable_identity(self) -> None:
        assert RosameIMilpRunner(resize=None).name == "ROSAME-I_MILP_24"


def test_every_tuned_domain_is_a_resolvable_bench_key() -> None:
    """``infer_bench_key`` resolves through the alias table alone.

    A hyperparameter entry under a key no alias produces would silently never
    be reached, and that domain would train on ``_DEFAULT_HYPERPARAMS``.
    """
    assert set(_HYPERPARAMS) <= set(BENCH_KEYS)


class TestRowNameDefaultsToName:
    """Runners without a per-domain configuration are untouched by the change."""

    def test_rosame_row_name_is_its_name(self, tmp_path: Path) -> None:
        from benchmark.baselines.rosame_runner import RosameBaselineRunner

        runner = RosameBaselineRunner()
        assert runner.row_name(_domain_file(tmp_path, "blocks")) == runner.name
