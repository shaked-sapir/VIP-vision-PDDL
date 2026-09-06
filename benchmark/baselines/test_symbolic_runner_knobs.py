"""The symbolic ROSAME arms take their training dynamics from config and record them."""

from benchmark.algorithm_adapters.po_rosame_runner import DEFAULT_BATCH_SIZE
from benchmark.baselines import resolve_baselines

SYMBOLIC = ["rosame_24", "rosame_milp_24", "rosame_milp_24_tag"]
SMALL_DATA = {"batch_size": 0, "normalize_base_loss": False, "rosame_seed": 7}


def test_small_data_knobs_reach_every_symbolic_arm():
    for runner in resolve_baselines(SYMBOLIC, **SMALL_DATA):
        assert runner.effective_batch_size == 0, runner.name
        assert runner.rosame_seed == 7, runner.name
        if runner.uses_milp:
            assert runner.normalize_base_loss is False, runner.name


def test_defaults_are_the_large_sweep_dynamics():
    for runner in resolve_baselines(SYMBOLIC):
        assert runner.effective_batch_size == DEFAULT_BATCH_SIZE, runner.name
        assert runner.rosame_seed == 42, runner.name
        if runner.uses_milp:
            assert runner.normalize_base_loss is True, runner.name


def test_run_params_record_what_the_row_was_trained_with():
    dl_only, milp, tag = resolve_baselines(SYMBOLIC, **SMALL_DATA)
    assert dl_only.run_params() == {
        "train_per_trajectory": True, "batch_size": 0, "rosame_seed": 7,
    }
    for arm in (milp, tag):
        assert arm.run_params() == {
            "train_per_trajectory": False, "batch_size": 0, "rosame_seed": 7,
            "normalize_base_loss": False, "epochs": arm.epochs,
        }


def test_the_knobs_do_not_leak_into_the_imaged_arms():
    (runner,) = resolve_baselines(["rosame_i_24"], **SMALL_DATA)
    assert not hasattr(runner, "normalize_base_loss")
    assert runner.run_params() == {}
