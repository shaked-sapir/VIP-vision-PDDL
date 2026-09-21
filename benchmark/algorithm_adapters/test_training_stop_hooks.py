"""The stop, timeout and checkpoint hooks of the two symbolic training loops.

    python -m pytest benchmark/algorithm_adapters/test_training_stop_hooks.py
"""

from __future__ import annotations

import torch

from benchmark.algorithm_adapters.best_checkpoint import BestModelTracker
from benchmark.algorithm_adapters.po_rosame_runner import PORosame_Runner
from benchmark.algorithm_adapters.rosame_milp.milp_loop import MilpPORosame
from benchmark.algorithm_adapters.test_po_rosame_runner import _SEED, _prepare


def _milp_runner(domain_path) -> MilpPORosame:
    torch.manual_seed(_SEED)
    return MilpPORosame(str(domain_path), normalize_base_loss=True)


class _Rounds:
    """A scripted ``milp_round``: each call pops the next ``(agreement, solved)``."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def __call__(self):
        agreement, solved = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return {}, agreement, {"exit_status": "OPTIMAL"}, (object() if solved else None)


# ----------------------------------------------------------------- tracker

class _FakeModel:
    def __init__(self):
        self.action_schemas = [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]


class TestTracker:
    def test_restores_the_lowest_loss_weights(self):
        model, tracker = _FakeModel(), BestModelTracker()
        tracker.bind(model)
        tracker.observe(3.0, 0)
        best = [s.weight.detach().clone() for s in model.action_schemas]
        for schema in model.action_schemas:
            torch.nn.init.constant_(schema.weight, 7.0)
        tracker.observe(5.0, 1)                       # worse: not captured
        tracker.restore(model)
        assert tracker.best_epoch == 0 and tracker.best_loss == 3.0
        for schema, weight in zip(model.action_schemas, best):
            assert torch.equal(schema.weight, weight)

    def test_reset_forgets_the_best_but_keeps_the_binding(self):
        model, tracker = _FakeModel(), BestModelTracker()
        tracker.bind(model)
        tracker.observe(1.0, 0)
        tracker.reset()
        assert tracker.best_loss is None and tracker.best_epoch is None
        tracker.observe(9.0, 4)                       # worse than before the reset, still the best since
        assert tracker.best_epoch == 4

    def test_inactive_is_a_no_op(self):
        model, tracker = _FakeModel(), BestModelTracker(active=False)
        tracker.bind(model)
        tracker.observe(1.0, 0)
        tracker.restore(model)
        assert tracker.best_loss is None


# ------------------------------------------------------------- DL-only loop

class TestPooledLoop:
    def test_no_hooks_runs_every_epoch(self, tmp_path):
        domain_path, prepared = _prepare(tmp_path)
        report = PORosame_Runner(str(domain_path)).learn_pooled(prepared, epochs=6)
        assert (report.stop_reason, report.epochs_run, len(report.losses)) == ("epochs_exhausted", 6, 6)

    def test_stop_check_ends_training_as_converged(self, tmp_path):
        domain_path, prepared = _prepare(tmp_path)
        report = PORosame_Runner(str(domain_path)).learn_pooled(
            prepared, epochs=50, stop_check=lambda losses: len(losses) >= 4)
        assert (report.stop_reason, report.epochs_run) == ("converged", 4)

    def test_timeout_check_ends_training_as_timeout(self, tmp_path):
        domain_path, prepared = _prepare(tmp_path)
        report = PORosame_Runner(str(domain_path)).learn_pooled(
            prepared, epochs=50, timeout_check=lambda: True)
        assert (report.stop_reason, report.epochs_run) == ("timeout", 1)

    def test_no_usable_traces_is_its_own_reason(self, tmp_path):
        domain_path, _ = _prepare(tmp_path)
        report = PORosame_Runner(str(domain_path)).learn_pooled([], epochs=5)
        assert (report.stop_reason, report.epochs_run) == ("no_usable_traces", 0)

    def test_the_hooks_do_not_perturb_training(self, tmp_path):
        """Hooks that never fire must leave the learned model identical."""
        domain_path, prepared = _prepare(tmp_path)
        torch.manual_seed(_SEED)
        plain = PORosame_Runner(str(domain_path))
        plain.learn_pooled(prepared, epochs=8, batch_size=None)
        torch.manual_seed(_SEED)
        hooked = PORosame_Runner(str(domain_path))
        hooked.learn_pooled(prepared, epochs=8, batch_size=None,
                            stop_check=lambda losses: False, timeout_check=lambda: False,
                            tracker=BestModelTracker(active=False))
        assert plain.rosame_to_pddl() == hooked.rosame_to_pddl()

    def test_the_emitted_model_is_the_best_epochs(self, tmp_path):
        domain_path, prepared = _prepare(tmp_path)
        torch.manual_seed(_SEED)
        runner = PORosame_Runner(str(domain_path))
        tracker = BestModelTracker()
        pddl, report = runner.learn_full_with_report(
            prepared, train_per_trajectory=False, epochs=12, tracker=tracker)
        assert report.best_epoch == report.losses.index(min(report.losses))
        assert report.best_loss == min(report.losses)
        assert pddl == runner.rosame_to_pddl()        # rendered after the restore

    def test_learn_full_still_returns_just_the_pddl(self, tmp_path):
        domain_path, prepared = _prepare(tmp_path)
        out = PORosame_Runner(str(domain_path)).learn_full(prepared, train_per_trajectory=False, epochs=3)
        assert isinstance(out, str) and "(:action move" in out


# ------------------------------------------------------------------ MILP loop

class TestMilpLoop:
    def test_legacy_agreement_stop_still_stops(self, tmp_path):
        domain_path, prepared = _prepare(tmp_path)
        report = _milp_runner(domain_path).learn_pooled_with_milp(
            prepared, _Rounds([(1.0, True)]), epochs=20, pre_mip_epochs=3, agreement_stop=1.0)
        assert report["stop_reason"] == "agreement_reached"
        assert report["epochs_run"] == 3

    def test_agreement_off_keeps_training_and_still_records_it(self, tmp_path):
        domain_path, prepared = _prepare(tmp_path)
        report = _milp_runner(domain_path).learn_pooled_with_milp(
            prepared, _Rounds([(1.0, True)]), epochs=8, pre_mip_epochs=3, agreement_stop=None)
        assert report["stop_reason"] == "epochs_exhausted"
        assert report["epochs_run"] == 8
        assert [r["agreement"] for r in report["rounds"]] == [1.0] * 6

    def test_the_series_starts_at_the_first_successful_solve(self, tmp_path):
        """Rounds at epochs 2, 3, 4 (0-based); the first two fail. The solve at
        epoch 4 is the first success, so scoring starts at epoch 5."""
        domain_path, prepared = _prepare(tmp_path)
        seen = []

        def stop_check(losses):
            seen.append(len(losses))
            return False

        rounds = _Rounds([(0.0, False), (0.0, False), (0.5, True)])
        report = _milp_runner(domain_path).learn_pooled_with_milp(
            prepared, rounds, epochs=9, pre_mip_epochs=3, agreement_stop=None, stop_check=stop_check)
        assert report["first_solve_epoch"] == 4
        assert seen == [1, 2, 3, 4]                   # epochs 5..8, never the warmup

    def test_converged_is_decided_on_the_post_solve_series_only(self, tmp_path):
        domain_path, prepared = _prepare(tmp_path)
        report = _milp_runner(domain_path).learn_pooled_with_milp(
            prepared, _Rounds([(0.5, True)]), epochs=40, pre_mip_epochs=3,
            agreement_stop=None, stop_check=lambda losses: len(losses) >= 5)
        assert report["stop_reason"] == "converged"
        assert report["epochs_run"] == 3 + 5          # warmup, then five scored epochs

    def test_the_tracker_restarts_at_the_first_solve(self, tmp_path):
        domain_path, prepared = _prepare(tmp_path)
        tracker = BestModelTracker()
        report = _milp_runner(domain_path).learn_pooled_with_milp(
            prepared, _Rounds([(0.5, True)]), epochs=10, pre_mip_epochs=4,
            agreement_stop=None, tracker=tracker)
        assert report["first_solve_epoch"] == 3
        assert report["best_epoch"] is not None and report["best_epoch"] > 3
        assert report["best_loss"] == min(report["losses"][4:])

    def test_timeout_is_checked_before_the_next_solve(self, tmp_path):
        domain_path, prepared = _prepare(tmp_path)
        rounds = _Rounds([(0.5, True)])
        report = _milp_runner(domain_path).learn_pooled_with_milp(
            prepared, rounds, epochs=20, pre_mip_epochs=1, agreement_stop=None,
            timeout_check=lambda: True)
        assert report["stop_reason"] == "timeout"
        assert rounds.calls == 0

    def test_loss_components_are_recorded_per_epoch(self, tmp_path):
        domain_path, prepared = _prepare(tmp_path)
        report = _milp_runner(domain_path).learn_pooled_with_milp(
            prepared, _Rounds([(0.5, True)]), epochs=6, pre_mip_epochs=2, agreement_stop=None)
        assert len(report["losses"]) == len(report["base_losses"]) == len(report["ce_losses"]) == 6
        assert all(ce == 0.0 for ce in report["ce_losses"][:2])      # no labels before the first solve
