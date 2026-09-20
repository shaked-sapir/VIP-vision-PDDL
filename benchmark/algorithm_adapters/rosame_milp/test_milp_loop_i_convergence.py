"""The imaged ROSAME+MILP loop under the training-loss convergence rule.

    python -m pytest benchmark/algorithm_adapters/rosame_milp/test_milp_loop_i_convergence.py
"""

from __future__ import annotations

from benchmark.algorithm_adapters.best_checkpoint import BestModelTracker
from benchmark.algorithm_adapters.rosame_milp.test_milp_loop_i import _Harness, _round_fn, _run


def test_agreement_off_trains_on_and_still_records_it():
    harness = _Harness(["a", "b"])
    milp_round, calls = _round_fn(harness, agreement=1.0)
    report = _run(harness, milp_round, epochs=6, pre_mip_epochs=2, agreement_stop=None)
    assert report["stop_reason"] == "epochs_exhausted"
    assert report["epochs_run"] == 6
    assert len(calls) == 5
    assert all(r["agreement"] == 1.0 for r in report["rounds"])


def test_the_legacy_default_still_stops_on_agreement():
    harness = _Harness(["a", "b"])
    milp_round, _calls = _round_fn(harness, agreement=1.0)
    report = _run(harness, milp_round, epochs=6, pre_mip_epochs=2)
    assert report["stop_reason"] == "agreement_reached"


def test_the_series_starts_after_the_first_successful_solve():
    harness = _Harness(["a"])
    outcomes = iter([None, None])          # the first two rounds fail, the rest solve

    def milp_round(selected, time_limit):
        solution = next(outcomes, object())
        return {}, {}, 0.5, {"exit_status": "OPTIMAL"}, solution

    seen = []

    def stop_check(losses):
        seen.append(len(losses))
        return False

    report = _run(harness, milp_round, epochs=8, pre_mip_epochs=2,
                  agreement_stop=None, stop_check=stop_check)
    assert report["first_solve_epoch"] == 3      # rounds at epochs 1, 2, 3 (0-based)
    assert seen == [1, 2, 3, 4]                  # epochs 4..7 only


def test_converged_is_reported_with_the_epochs_run():
    harness = _Harness(["a"])
    milp_round, _calls = _round_fn(harness, agreement=0.5)
    report = _run(harness, milp_round, epochs=50, pre_mip_epochs=2,
                  agreement_stop=None, stop_check=lambda losses: len(losses) >= 3)
    assert report["stop_reason"] == "converged"
    assert report["epochs_run"] == 2 + 3


def test_the_tracker_scores_only_the_post_solve_epochs():
    harness = _Harness(["a"])
    milp_round, _calls = _round_fn(harness, agreement=0.5)
    tracker = BestModelTracker()
    report = _run(harness, milp_round, epochs=6, pre_mip_epochs=3,
                  agreement_stop=None, tracker=tracker)
    assert report["first_solve_epoch"] == 2
    assert report["best_epoch"] > report["first_solve_epoch"]
    # The base loss is stubbed to zero, so what is scored is the state
    # pseudo-label term, which only exists after the first solve.
    assert report["best_loss"] > 0.0
