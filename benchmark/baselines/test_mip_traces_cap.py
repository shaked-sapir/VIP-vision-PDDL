"""The rosame_milp_* arms must cap what reaches the MILP.

    python -m pytest benchmark/baselines/test_mip_traces_cap.py

Upstream caps it with a TraceSelector (vendor default 3, see
src/milp/vendor/UPSTREAM.md:95). Our runner defaulted to None -- the whole fold
-- which was about the paper's subset at the 3-8 trajectories it was written
for, and 2000 traces at L=2000.
"""

import inspect

import yaml

from benchmark.baselines import get_baselines
from benchmark.benchmark_runner import _RUNNER_KWARG_KEYS, _build_main_kwargs

LARGE_CONFIG = "benchmark/run_config_large.yaml"


def _shared():
    return yaml.safe_load(open(LARGE_CONFIG))["shared"]


class TestTheKnobCanReachTheRunner:
    def test_mip_traces_is_a_forwarded_runner_kwarg(self):
        """Without this, naming it in the config has no effect at all."""
        assert "mip_traces" in _RUNNER_KWARG_KEYS

    def test_the_milp_runners_accept_it(self):
        for key in ("rosame_milp_24", "rosame_milp_24_tag"):
            runner = get_baselines([key], mip_traces=4)[0]
            assert runner.mip_traces == 4, key

    def test_the_dl_only_arm_ignores_it(self):
        """ROSAME_24 runs no MILP; _instantiate must drop the kwarg."""
        runner = get_baselines(["rosame_24"], mip_traces=4)[0]
        assert not hasattr(runner, "mip_traces")


class TestTheLargeSweepSetsIt:
    def test_config_caps_every_milp_arm(self):
        for runner in _build_main_kwargs(_shared())["baselines"]:
            if hasattr(runner, "mip_traces"):
                assert runner.mip_traces is not None, (
                    f"{runner.name} would encode the whole fold; at L=2000 that "
                    f"is 2000 traces and ~11 GB in one worker"
                )

    def test_both_milp_families_use_the_same_subset(self):
        """The point of 4 rather than upstream's 3: comparable conditions."""
        kwargs = _build_main_kwargs(_shared())
        pisam = kwargs["pisam_milp_configs"][0].subset_size.value
        rosame = [
            r.mip_traces for r in kwargs["baselines"] if hasattr(r, "mip_traces")
        ]
        assert rosame, "no MILP baseline arms in the sweep config"
        assert all(v == pisam for v in rosame), (
            f"pisam subset_size={pisam} but rosame mip_traces={rosame}"
        )


def test_the_subsample_gate_is_a_no_op_when_unset():
    """Documents the failure mode: None skips the sample() entirely."""
    source = inspect.getsource(
        __import__(
            "benchmark.baselines.rosame_milp_runner", fromlist=["RosameMilpRunner"]
        ).RosameMilpRunner.learn
    )
    assert "self.mip_traces is not None" in source
