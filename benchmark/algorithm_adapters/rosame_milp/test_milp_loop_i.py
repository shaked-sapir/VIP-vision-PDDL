"""Tests for the imaged ROSAME+MILP loop (:mod:`milp_loop_i`).

    python -m benchmark.algorithm_adapters.rosame_milp.test_milp_loop_i

The CV model and the base ROSAME-I loss are stubbed out; what is under test is
the loop's schedule, its two pseudo-label channels and its budget arithmetic.

Covers:
  1. Solve epochs: warmup is respected and the interval counts from its end.
  2. The state term is BCE summed over frames and averaged over propositions.
  3. psi decays the state term per elapsed epoch; relabelling resets it.
  4. psi=0 leaves the state channel inert from the epoch after a solve.
  5. Only the ``mip_traces`` prefix is relabelled; the rest keep decaying.
  6. ``agreement_stop=0.0`` stops at the first round that returns a solution.
  7. A round without a solution installs no labels and does not stop the loop.
  8. The budget splits over the remaining solves and raises ``mip_interval``
     rather than scheduling solves too short to be useful.
  9. ``timeout_check`` and an empty trace list end the loop with their reasons.
 10. State labels of the wrong shape raise instead of broadcasting.
 11. A single-step trace has no interior frame and so contributes no state term.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from benchmark.algorithm_adapters.rosame_i_runner import _PreparedTrace
from benchmark.algorithm_adapters.rosame_milp.milp_loop_i import (
    _MIN_SOLVE_SECONDS,
    MilpRosameI,
)

_N_PROPS = 4


# ---------------------------------------------------------------- harness

class _Harness(MilpRosameI):
    """``MilpRosameI`` with a per-trace parameter standing in for the CV model.

    Does not call ``super().__init__``: there is no domain to parse here. Every
    attribute the loop reaches for is set directly.
    """

    def __init__(self, names, n_frames=4, psi=0.99):
        self.psi = psi
        self.seed = 0
        self.device = torch.device("cpu")
        self._model_labels = None
        self._state_labels = {}
        self._epochs_since_label = {}
        self.rosame = SimpleNamespace(action_schemas=[])

        self._params = {
            name: torch.nn.Parameter(torch.full((n_frames, _N_PROPS), 0.5))
            for name in names
        }
        self._traces = [
            _PreparedTrace(
                images=torch.zeros(n_frames, 3, 4, 4),
                action_indices=[0] * (n_frames - 1),
                final_state_vec=torch.zeros(_N_PROPS),
                name=name,
            )
            for name in names
        ]
        # The empty-trace case still needs a non-empty parameter list.
        params = list(self._params.values()) or [torch.nn.Parameter(torch.zeros(1))]
        self.optimizer = torch.optim.SGD(params, lr=0.0)
        self.base_loss_calls = 0

    def prepare_traces(self, prepared_problems):
        return list(self._traces)

    def _forward_predictions(self, trace, augment):
        return self._params[trace.name]

    def _loss_from_predictions(self, preds, trace, gamma, lambda_):
        self.base_loss_calls += 1
        return (preds * 0.0).sum()

    def _total_loss(self, traces, gamma, lambda_):
        return 0.0


def _run(harness, milp_round, epochs=6, **kwargs):
    return harness.learn_pooled_with_milp(
        prepared_problems=[], milp_round=milp_round, epochs=epochs,
        gamma=1.0, lambda_=1.0, augment=False, **kwargs,
    )


def _labels_for(harness, value=1.0):
    """State labels of the interior shape the loop expects, for every trace."""
    return {
        t.name: torch.full((t.images.shape[0] - 2, _N_PROPS), value)
        for t in harness._traces
    }


def _round_fn(harness, agreement=0.5, solution=object(), value=1.0):
    """A ``milp_round`` that relabels whichever traces it is handed."""
    calls = []

    def milp_round(selected, time_limit):
        calls.append(([t.name for t in selected], time_limit))
        labels = {
            t.name: torch.full((t.images.shape[0] - 2, _N_PROPS), value)
            for t in selected
        }
        return {}, labels, agreement, {"exit_status": "OPTIMAL"}, solution

    return milp_round, calls


# ---------------------------------------------------------------- tests

def test_solve_epochs_follow_warmup_and_interval():
    solve = MilpRosameI._should_solve
    fired = [e for e in range(10) if solve(e, pre_mip_epochs=3, mip_interval=2)]
    assert fired == [2, 4, 6, 8], fired
    every = [e for e in range(6) if solve(e, pre_mip_epochs=1, mip_interval=1)]
    assert every == [0, 1, 2, 3, 4, 5], every
    print("PASS  solves start after warmup and repeat on the interval")


def test_state_term_is_summed_over_frames_and_averaged_over_props():
    harness = _Harness(["a"], n_frames=5)
    harness.set_state_labels(_labels_for(harness, value=1.0))
    trace = harness._traces[0]

    ce = harness._state_ce(harness._params[trace.name], trace)

    # Predictions are all 0.5 against all-ones labels: -log(0.5) per element,
    # summed over the 3 interior frames, averaged over the propositions.
    expected = 3 * _N_PROPS * -torch.log(torch.tensor(0.5)) / _N_PROPS
    assert torch.isclose(ce, expected, atol=1e-6), (ce, expected)
    print("PASS  the state term sums over frames and averages over propositions")


def test_psi_decays_the_state_term_and_relabelling_resets_it():
    harness = _Harness(["a"], n_frames=5, psi=0.5)
    harness.set_state_labels(_labels_for(harness))
    trace = harness._traces[0]
    fresh = harness._state_ce(harness._params["a"], trace)

    harness._age_state_labels()
    harness._age_state_labels()
    aged = harness._state_ce(harness._params["a"], trace)
    assert torch.isclose(aged, fresh * 0.25, atol=1e-6), (aged, fresh)

    harness.set_state_labels(_labels_for(harness))
    assert torch.isclose(harness._state_ce(harness._params["a"], trace), fresh)
    print("PASS  psi decays the state term per epoch and relabelling resets it")


def test_psi_zero_makes_the_state_channel_inert_after_one_epoch():
    harness = _Harness(["a"], n_frames=5, psi=0.0)
    harness.set_state_labels(_labels_for(harness))
    trace = harness._traces[0]
    assert harness._state_ce(harness._params["a"], trace) > 0, "psi**0 == 1"

    harness._age_state_labels()
    assert float(harness._state_ce(harness._params["a"], trace).detach()) == 0.0
    print("PASS  psi=0 keeps only the labelling epoch's state term")


def test_only_the_selected_prefix_is_relabelled():
    harness = _Harness(["a", "b", "c"], n_frames=5)
    milp_round, calls = _round_fn(harness)

    _run(harness, milp_round, epochs=3, pre_mip_epochs=3, mip_interval=1, mip_traces=1)

    assert len(calls) == 1, calls
    selected, _limit = calls[0]
    assert len(selected) == 1, selected
    assert set(harness._state_labels) == set(selected), harness._state_labels.keys()
    print("PASS  a round relabels only the traces it was handed")


def test_unselected_traces_keep_their_labels_and_go_on_decaying():
    harness = _Harness(["a", "b"], n_frames=5, psi=0.5)
    harness.set_state_labels(_labels_for(harness))

    def milp_round(selected, time_limit):
        labels = {"a": torch.full((3, _N_PROPS), 1.0)}
        return {}, labels, 0.5, {}, object()

    harness._run_round(
        report={"rounds": []}, traces=harness._traces, order=[0], epoch=0,
        milp_round=milp_round, epochs=1, pre_mip_epochs=0, mip_interval=1,
        mip_traces=1, mip_time_limit=10.0, agreement_stop=1.0, seconds_left=None,
    )

    assert harness._epochs_since_label == {"a": 0, "b": 0}
    harness._age_state_labels()
    assert harness._epochs_since_label == {"a": 1, "b": 1}
    assert set(harness._state_labels) == {"a", "b"}, "b must keep its old labels"
    print("PASS  unselected traces keep their labels and keep decaying")


def test_agreement_stop_zero_stops_at_the_first_solution():
    harness = _Harness(["a"], n_frames=5)
    milp_round, calls = _round_fn(harness, agreement=0.0)

    report = _run(
        harness, milp_round, epochs=6, pre_mip_epochs=2, mip_interval=1,
        agreement_stop=0.0,
    )

    assert report["stop_reason"] == "agreement_reached", report["stop_reason"]
    assert len(calls) == 1, calls
    assert len(report["rounds"]) == 1
    assert report["final_solution"] is not None
    print("PASS  agreement_stop=0.0 stops at the first round with a solution")


def test_a_solutionless_round_installs_nothing_and_does_not_stop():
    harness = _Harness(["a"], n_frames=5)

    def milp_round(selected, time_limit):
        return {"x": torch.zeros(1, 4)}, _labels_for(harness), 1.0, {}, None

    report = _run(harness, milp_round, epochs=4, pre_mip_epochs=4, mip_interval=1)

    assert report["stop_reason"] == "epochs_exhausted", report["stop_reason"]
    assert report["final_solution"] is None
    assert harness._state_labels == {}
    assert harness._model_labels is None
    print("PASS  an infeasible round installs no labels and lets training run on")


def test_budget_splits_over_remaining_solves():
    harness = _Harness(["a"])
    unbounded = harness._solve_time_limit(60.0, 1, remaining_solves=4, seconds_left=None)
    assert unbounded == (60.0, 1), unbounded

    limit, interval = harness._solve_time_limit(
        60.0, 1, remaining_solves=4, seconds_left=80.0
    )
    assert (limit, interval) == (20.0, 1), (limit, interval)

    limit, _ = harness._solve_time_limit(5.0, 1, remaining_solves=1, seconds_left=999.0)
    assert limit == 5.0, "the configured ceiling still caps a fat budget"
    print("PASS  each solve gets its share of the remaining budget, under the ceiling")


def test_a_thin_budget_raises_the_interval_instead_of_starving_solves():
    harness = _Harness(["a"])
    limit, interval = harness._solve_time_limit(
        60.0, 1, remaining_solves=100, seconds_left=100.0
    )
    assert interval > 1, interval
    assert limit >= _MIN_SOLVE_SECONDS, limit
    print("PASS  a thin budget doubles the interval rather than starving each solve")


def test_timeout_and_empty_traces_report_their_reasons():
    harness = _Harness(["a"], n_frames=5)
    milp_round, _calls = _round_fn(harness)
    report = _run(
        harness, milp_round, epochs=9, pre_mip_epochs=99,
        timeout_check=lambda: True,
    )
    assert report["stop_reason"] == "timeout", report["stop_reason"]
    assert harness.base_loss_calls == 1, "the timeout is checked between epochs"

    empty = _Harness([], n_frames=5)
    report = _run(empty, milp_round, epochs=3)
    assert report["stop_reason"] == "no_usable_traces", report["stop_reason"]
    assert report["rounds"] == []
    print("PASS  timeout and an empty trace list stop the loop with their reasons")


def test_state_labels_of_the_wrong_shape_raise():
    harness = _Harness(["a"], n_frames=5)
    harness.set_state_labels({"a": torch.ones(5, _N_PROPS)})  # frames, not interior
    try:
        harness._state_ce(harness._params["a"], harness._traces[0])
    except ValueError:
        print("PASS  state labels of the wrong shape raise instead of broadcasting")
    else:
        raise AssertionError("expected ValueError for mis-shaped state labels")


def test_a_single_step_trace_contributes_no_state_term():
    """Two frames leave no interior, and an empty BCE is a hard crash on MPS."""
    harness = _Harness(["a"], n_frames=2)
    labels = _labels_for(harness)
    assert labels["a"].shape == (0, _N_PROPS), labels["a"].shape
    harness.set_state_labels(labels)

    assert harness._state_ce(harness._params["a"], harness._traces[0]) is None

    milp_round, calls = _round_fn(harness)
    report = _run(harness, milp_round, epochs=2, pre_mip_epochs=1, mip_interval=1)

    assert len(calls) == 2, calls
    assert report["stop_reason"] == "epochs_exhausted", report["stop_reason"]
    print("PASS  a single-step trace contributes no state term")


if __name__ == "__main__":
    test_solve_epochs_follow_warmup_and_interval()
    test_state_term_is_summed_over_frames_and_averaged_over_props()
    test_psi_decays_the_state_term_and_relabelling_resets_it()
    test_psi_zero_makes_the_state_channel_inert_after_one_epoch()
    test_only_the_selected_prefix_is_relabelled()
    test_unselected_traces_keep_their_labels_and_go_on_decaying()
    test_agreement_stop_zero_stops_at_the_first_solution()
    test_a_solutionless_round_installs_nothing_and_does_not_stop()
    test_budget_splits_over_remaining_solves()
    test_a_thin_budget_raises_the_interval_instead_of_starving_solves()
    test_timeout_and_empty_traces_report_their_reasons()
    test_state_labels_of_the_wrong_shape_raise()
    test_a_single_step_trace_contributes_no_state_term()
    print("ALL TESTS PASSED")
