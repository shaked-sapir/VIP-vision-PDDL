"""The baseline runners under a ``gt_setup``: which endpoint states each arm fixes.

    python -m pytest benchmark/baselines/test_gt_setup_runners.py

Without a setup every arm anchors as it did before setups existed. With one, the
symbolic ROSAME+MILP arms derive both MILP switches from it and keep their plain
row label, the ICAPS-24 pixel arms take their CV anchors and MILP fixings from
it, and the ICAPS-26 arms run only under ``init_final``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import src.milp  # noqa: F401  (vendor sys.path bootstrap)
from benchmark.algorithm_adapters.endpoint_anchors import EndpointAnchors, anchors_of
from benchmark.baselines import get_baselines
from benchmark.baselines.image_fold_inputs import resolve_fold_inputs
from benchmark.baselines.regime import DegradationRegime, gate_baselines
from benchmark.baselines.rosame26_milp_runner import Rosame26MilpRunner
from benchmark.baselines.rosame26_runner import Rosame26BaselineRunner
from benchmark.baselines.rosame_i_milp_runner import RosameIMilpRunner, as_fluents
from benchmark.baselines.rosame_i_runner import (
    RosameIBaselineRunner,
    as_prepared_problem,
    reads_observed_final,
)
from benchmark.baselines.rosame_milp_runner import (
    FREE_GOAL_MODE,
    GT_GOAL_MODE,
    RosameMilpRunner,
    RosameMilpTagRunner,
    milp_switches_for,
)
from benchmark.baselines.setup_option import only_init_final, runner_gt_setup
from benchmark.baselines.test_image_fold_inputs import _two_predicate_fold
from src.milp.converter import GtAnchoring, build_ps_domain
from src.utils.gt_setup import GtSetup

DOMAIN = Path("unused-domain.pddl")

_MILP_SWITCHES = {
    GtSetup.NONE: (FREE_GOAL_MODE, GtAnchoring.NONE),
    GtSetup.INIT: (FREE_GOAL_MODE, GtAnchoring.INIT_ONLY),
    GtSetup.FINAL: (GT_GOAL_MODE, GtAnchoring.NONE),
    GtSetup.INIT_FINAL: (GT_GOAL_MODE, GtAnchoring.INIT_ONLY),
}


# ── the option itself ───────────────────────────────────────────────────


class TestRunnerGtSetup:
    def test_none_stays_none(self) -> None:
        assert runner_gt_setup(None) is None

    def test_a_config_string_is_parsed(self) -> None:
        assert runner_gt_setup("init_final") is GtSetup.INIT_FINAL
        assert runner_gt_setup(GtSetup.NONE) is GtSetup.NONE

    def test_an_unknown_value_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown gt_setup"):
            runner_gt_setup("both")


# ── symbolic ROSAME+MILP ────────────────────────────────────────────────


class TestSymbolicMilpSwitches:
    @pytest.mark.parametrize("setup", list(GtSetup))
    def test_a_setup_fixes_exactly_the_states_it_gives(self, setup: GtSetup) -> None:
        assert milp_switches_for(setup) == _MILP_SWITCHES[setup]

    def test_no_setup_fixes_both_ends(self) -> None:
        assert milp_switches_for(None) == (GT_GOAL_MODE, GtAnchoring.INIT_ONLY)

    def test_no_setup_honours_the_explicit_switches(self) -> None:
        assert milp_switches_for(None, FREE_GOAL_MODE, GtAnchoring.NONE) == (
            FREE_GOAL_MODE, GtAnchoring.NONE,
        )

    @pytest.mark.parametrize("switches", [
        {"goal_mode": GT_GOAL_MODE},
        {"gt_anchoring": GtAnchoring.INIT_ONLY},
        {"goal_mode": FREE_GOAL_MODE, "gt_anchoring": GtAnchoring.NONE},
    ])
    def test_a_setup_with_an_explicit_switch_raises(self, switches: dict) -> None:
        with pytest.raises(ValueError, match="not both"):
            milp_switches_for(GtSetup.INIT, **switches)
        with pytest.raises(ValueError, match="not both"):
            RosameMilpRunner(gt_setup=GtSetup.INIT, **switches)


class TestSymbolicMilpRunner:
    @pytest.mark.parametrize("runner_cls", [RosameMilpRunner, RosameMilpTagRunner])
    @pytest.mark.parametrize("setup", list(GtSetup))
    def test_the_row_keeps_the_plain_arm_label(self, runner_cls, setup: GtSetup) -> None:
        runner = runner_cls(gt_setup=setup.value)

        assert (runner.goal_mode, runner.gt_anchoring) == _MILP_SWITCHES[setup]
        assert runner.row_name(DOMAIN) == runner.name
        assert runner.run_params()["gt_setup"] == setup.value

    def test_a_solve_cap_still_labels_the_row(self) -> None:
        runner = RosameMilpRunner(gt_setup=GtSetup.NONE, mip_time_limit=3600)
        assert runner.row_name(DOMAIN) == "ROSAME_MILP_24__solve=3600"

    def test_without_a_setup_nothing_changes(self) -> None:
        runner = RosameMilpRunner()

        assert (runner.goal_mode, runner.gt_anchoring) == (GT_GOAL_MODE, GtAnchoring.INIT_ONLY)
        assert runner.gt_setup is None
        assert "gt_setup" not in runner.run_params()
        assert runner.row_name(DOMAIN) == "ROSAME_MILP_24"
        assert RosameMilpRunner(goal_mode=FREE_GOAL_MODE).row_name(DOMAIN) == (
            "ROSAME_MILP_24__goal=none"
        )


# ── the registry ────────────────────────────────────────────────────────


def test_the_setup_reaches_only_the_runners_that_accept_it() -> None:
    keys = [
        "rosame_24", "nolam", "rosame_milp_24", "rosame_milp_24_tag",
        "rosame_i_24", "rosame_i_milp_24", "rosame_i_26", "rosame_i_milp_26",
    ]
    runners = get_baselines(keys, gt_setup="final")

    setups = [getattr(runner, "gt_setup", "absent") for runner in runners]
    assert setups == ["absent", "absent"] + [GtSetup.FINAL] * 6


def test_no_setup_given_leaves_every_runner_without_one() -> None:
    runners = get_baselines(["rosame_milp_24", "rosame_i_24", "rosame_i_milp_24", "rosame_i_26"])
    assert [runner.gt_setup for runner in runners] == [None] * 4


# ── ICAPS-24 pixel arms ─────────────────────────────────────────────────


def _resolved(tmp_path: Path, setup):
    """The toy fold's one trace, resolved as the pixel arms resolve it under ``setup``."""
    domain, prepared = _two_predicate_fold(tmp_path)
    (trace,) = resolve_fold_inputs(
        domain, prepared, "toy", with_observed_final=reads_observed_final(setup)
    )
    return domain, trace


def _atoms(predicates) -> set:
    return {" ".join(p.split()) for p in predicates}


class TestPreparedProblemAnchors:
    def test_without_a_setup_it_is_the_gt_final_state_alone(self, tmp_path: Path) -> None:
        _domain, trace = _resolved(tmp_path, None)

        prepared = as_prepared_problem(trace)

        assert len(prepared) == 4
        assert anchors_of(prepared) is None
        assert _atoms(prepared[3]) == {"big d2"}

    @pytest.mark.parametrize("setup", [GtSetup.NONE, GtSetup.INIT])
    def test_a_final_state_not_given_is_the_trajectorys_own(
        self, tmp_path: Path, setup: GtSetup
    ) -> None:
        _domain, trace = _resolved(tmp_path, setup)

        prepared = as_prepared_problem(trace, setup)

        assert prepared[3] == ["big d2"]
        assert anchors_of(prepared).final_unknown == ("big d1", "clear d2")

    @pytest.mark.parametrize("setup", [GtSetup.FINAL, GtSetup.INIT_FINAL])
    def test_a_given_final_state_is_the_gt_one_with_nothing_unknown(
        self, tmp_path: Path, setup: GtSetup
    ) -> None:
        _domain, trace = _resolved(tmp_path, setup)

        prepared = as_prepared_problem(trace, setup)

        assert _atoms(prepared[3]) == {"big d2"}
        assert anchors_of(prepared).final_unknown == ()

    @pytest.mark.parametrize("setup", list(GtSetup))
    def test_the_gt_initial_state_is_attached_only_when_given(
        self, tmp_path: Path, setup: GtSetup
    ) -> None:
        _domain, trace = _resolved(tmp_path, setup)

        init = anchors_of(as_prepared_problem(trace, setup)).init_predicates

        if setup.gives_init:
            assert _atoms(init) == {"clear d1", "big d2"}
        else:
            assert init is None

    def test_a_trace_resolved_without_its_last_state_raises(self, tmp_path: Path) -> None:
        _domain, trace = _resolved(tmp_path, None)

        with pytest.raises(ValueError, match="resolved without"):
            as_prepared_problem(trace, GtSetup.NONE)


class TestPixelRunnerLabels:
    @pytest.mark.parametrize("runner_cls", [RosameIBaselineRunner, RosameIMilpRunner])
    @pytest.mark.parametrize("setup", list(GtSetup))
    def test_the_setup_is_recorded_and_not_in_the_name(self, runner_cls, setup: GtSetup) -> None:
        runner = runner_cls(gt_setup=setup)

        assert runner.name == runner_cls().name
        assert runner.run_params() == {"gt_setup": setup.value}

    @pytest.mark.parametrize("runner_cls", [RosameIBaselineRunner, RosameIMilpRunner])
    def test_without_a_setup_nothing_is_recorded(self, runner_cls) -> None:
        assert runner_cls().run_params() == {}


class TestPixelMilpFixings:
    @pytest.mark.parametrize("setup", list(GtSetup))
    def test_the_milp_fixes_exactly_the_states_the_setup_gives(self, setup: GtSetup) -> None:
        runner = RosameIMilpRunner(gt_setup=setup)

        assert runner.milp_fixes_init is setup.gives_init
        assert (runner.goal_mode == GT_GOAL_MODE) is setup.gives_final

    def test_without_a_setup_it_fixes_both_ends(self) -> None:
        runner = RosameIMilpRunner()
        assert runner.milp_fixes_init and runner.goal_mode == GT_GOAL_MODE
        assert RosameIMilpRunner(goal_mode=FREE_GOAL_MODE).goal_mode == FREE_GOAL_MODE

    def test_a_setup_with_an_explicit_goal_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            RosameIMilpRunner(gt_setup=GtSetup.FINAL, goal_mode=GT_GOAL_MODE)

    @pytest.mark.parametrize("setup", [None, *GtSetup])
    def test_the_trace_context_carries_the_setups_endpoints(self, tmp_path: Path, setup) -> None:
        domain, trace = _resolved(tmp_path, setup)
        runner = RosameIMilpRunner(gt_setup=setup)
        prepared = [as_prepared_problem(trace, setup)]

        contexts, n_gt_goals = runner._build_contexts(
            build_ps_domain(domain), domain, prepared, [trace.gt_trajectory_path]
        )
        (context,) = contexts.values()

        fixes_init = setup is None or setup.gives_init
        fixes_final = setup is None or setup.gives_final
        assert context.init_fluents == (
            {("clear", ("d1",)), ("big", ("d2",))} if fixes_init else None
        )
        assert context.goal_fluents == ({("big", ("d2",))} if fixes_final else None)
        assert n_gt_goals == int(fixes_final)
        assert context.final_observed == (
            None if fixes_final
            else ({("big", ("d2",))}, {("big", ("d1",)), ("clear", ("d2",))})
        )

    def test_free_goal_mode_without_a_setup_observes_nothing(self, tmp_path: Path) -> None:
        domain, trace = _resolved(tmp_path, None)
        runner = RosameIMilpRunner(goal_mode=FREE_GOAL_MODE)

        contexts, _ = runner._build_contexts(
            build_ps_domain(domain), domain, [as_prepared_problem(trace)],
            [trace.gt_trajectory_path],
        )
        (context,) = contexts.values()

        assert context.goal_fluents is None and context.final_observed is None

    def test_as_fluents_splits_names_from_arguments(self) -> None:
        assert as_fluents(["on a b", "handempty ", ""]) == {("on", ("a", "b")), ("handempty", ())}


def test_default_endpoint_anchors_anchor_nothing_extra() -> None:
    assert EndpointAnchors() == EndpointAnchors(final_unknown=(), init_predicates=None)


# ── ICAPS-26 arms ───────────────────────────────────────────────────────


class TestIcaps26Arms:
    @pytest.mark.parametrize("runner_cls", [Rosame26BaselineRunner, Rosame26MilpRunner])
    @pytest.mark.parametrize("setup", [None, GtSetup.INIT_FINAL])
    def test_both_endpoints_or_no_setup_is_supported(self, runner_cls, setup) -> None:
        runner = runner_cls(gt_setup=setup)
        assert runner.supports(DegradationRegime.image()) == (True, "")

    @pytest.mark.parametrize("runner_cls", [Rosame26BaselineRunner, Rosame26MilpRunner])
    @pytest.mark.parametrize("setup", [GtSetup.NONE, GtSetup.INIT, GtSetup.FINAL])
    def test_any_other_setup_is_gated_out_and_refused(self, runner_cls, setup: GtSetup) -> None:
        runner = runner_cls(gt_setup=setup)

        supported, reason = runner.supports(DegradationRegime.image())
        assert not supported
        assert f"'{setup.value}'" in reason and "init_final" in reason

        with pytest.raises(ValueError, match=f"not '{setup.value}'"):
            runner.learn(DOMAIN, [], Path("unused"), timeout_seconds=1)

    def test_the_gate_drops_them_and_keeps_the_24_arm(self) -> None:
        runners = [
            Rosame26BaselineRunner(gt_setup=GtSetup.INIT),
            Rosame26MilpRunner(gt_setup=GtSetup.INIT),
            RosameIBaselineRunner(gt_setup=GtSetup.INIT),
        ]

        kept, dropped = gate_baselines(runners, DegradationRegime.image())

        assert [runner.name for runner in kept] == ["ROSAME-I_24"]
        assert set(dropped) == {"ROSAME-I_26", "ROSAME-I_MILP_26"}

    def test_the_setup_is_recorded_when_given(self) -> None:
        assert Rosame26BaselineRunner().run_params() == {}
        assert Rosame26MilpRunner(gt_setup="init_final").run_params() == {
            "gt_setup": "init_final"
        }

    def test_only_init_final_names_the_arm(self) -> None:
        assert only_init_final(GtSetup.FINAL, "ARM")[1].startswith("ARM takes both endpoint")
