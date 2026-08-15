"""The ROSAME+MILP iterative loop in imaged mode: two pseudo-label channels.

Warm up for ``pre_mip_epochs``, then every ``mip_interval`` epochs solve a MILP
over a subset of traces and train on its solution. Unlike the simulation-mode
sibling (:mod:`milp_loop`), states here are network outputs, so both channels of
the paper apply: the action-model CE (undecayed) and a per-trace state BCE whose
weight decays by ``psi`` for every epoch since that trace was last relabelled.
Actions are observed, so the third (action) channel does not apply.

See ``docs/rosame-i-milp-implementation-plan.md`` §5 for the schedule and the
pseudo-term scaling.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from benchmark.algorithm_adapters.rosame_i_runner import (
    RosameI_Runner,
    _PreparedTrace,
    _set_seed,
)
from benchmark.algorithm_adapters.rosame_milp.model_bridge import model_cross_entropy

# Probability floor for the state channel, matching the MILP boundary's clamp.
_EPS = 1e-4

# A solve shorter than this cannot produce anything useful, so the loop raises
# ``mip_interval`` instead of spending the time.
_MIN_SOLVE_SECONDS = 5.0

# One MILP round over the given traces, under the given per-solve time limit.
# Returns (model labels per schema, state labels per trace name, agreement,
# solve stats, decoded solution).
MilpRoundFn = Callable[
    [Sequence[_PreparedTrace], float],
    Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float, Dict, object],
]


class MilpRosameI(RosameI_Runner):
    """ROSAME-I with MILP pseudo-labels on the model and state channels."""

    def __init__(self, *args, psi: float = 0.99, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.psi = psi
        self._model_labels: Optional[Dict[str, torch.Tensor]] = None
        self._state_labels: Dict[str, torch.Tensor] = {}
        self._epochs_since_label: Dict[str, int] = {}

    # ------------------------------------------------------------- labels

    def set_model_labels(self, labels: Optional[Dict[str, torch.Tensor]]) -> None:
        """Replace the action-model pseudo-labels (undecayed channel)."""
        self._model_labels = labels

    def set_state_labels(self, labels: Dict[str, torch.Tensor]) -> None:
        """Install per-trace state labels and reset their decay exponent to zero."""
        for name, label in labels.items():
            self._state_labels[name] = label.to(self.device)
            self._epochs_since_label[name] = 0

    def _age_state_labels(self) -> None:
        """One epoch has elapsed for every labelled trace."""
        for name in self._epochs_since_label:
            self._epochs_since_label[name] += 1

    # ------------------------------------------------------------- losses

    def _state_ce(
        self, preds: torch.Tensor, trace: _PreparedTrace
    ) -> Optional[torch.Tensor]:
        """psi-decayed BCE of the interior frames against the MILP's repaired states.

        ``preds`` are raw logits over ``T + 1`` frames; the labels cover the
        ``T - 1`` interior ones, the endpoints being hard-fixed GT in the MILP.
        Summed over frames and averaged over propositions, which puts the term at
        the same scale as the sum-reduced base loss.
        """
        label = self._state_labels.get(trace.name)
        if label is None:
            return None
        interior = preds[1:-1]
        if interior.shape != label.shape:
            raise ValueError(
                f"state labels for trace '{trace.name}' have shape "
                f"{tuple(label.shape)}, expected {tuple(interior.shape)}"
            )
        weight = self.psi ** self._epochs_since_label.get(trace.name, 0)
        ce = F.binary_cross_entropy(
            interior.clamp(_EPS, 1.0 - _EPS), label, reduction="sum"
        )
        return weight * ce / label.shape[1]

    def _train_step(
        self, trace: _PreparedTrace, gamma: float, lambda_: float, augment: bool
    ) -> float:
        """One optimizer step: base ROSAME-I loss + both pseudo-label channels.

        The base terms read the same forward pass the state channel does, so an
        augmented batch is never scored twice under two different flips.
        """
        self.optimizer.zero_grad()
        preds = self._forward_predictions(trace, augment)
        loss = self._loss_from_predictions(preds, trace, gamma, lambda_)

        model_ce = model_cross_entropy(self.rosame.action_schemas, self._model_labels)
        if model_ce is not None:
            loss = loss + model_ce.to(self.device)
        state_ce = self._state_ce(preds, trace)
        if state_ce is not None:
            loss = loss + state_ce

        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    # ------------------------------------------------------- MILP input

    def predict_frames(self, trace: _PreparedTrace) -> List[List[float]]:
        """``(T + 1, n_props)`` raw CV logits for one trace, unaugmented, no grad."""
        was_training = self.cv_model.training
        self.cv_model.eval()
        try:
            with torch.no_grad():
                preds = self._forward_predictions(trace, augment=False)
        finally:
            self.cv_model.train(was_training)
        return preds.detach().cpu().tolist()

    @property
    def proposition_names(self) -> List[str]:
        """The CV head's column order — ROSAME's grounded proposition keys."""
        return list(self.rosame.propositions.keys())

    # --------------------------------------------------------- the budget

    def _solve_time_limit(
        self,
        mip_time_limit: float,
        mip_interval: int,
        remaining_solves: int,
        seconds_left: Optional[float],
    ) -> Tuple[float, int]:
        """``(time limit for this solve, possibly raised mip_interval)``.

        With no budget reported, the configured limit stands. Otherwise the
        remaining budget is divided over the solves still scheduled, and the
        interval is doubled until each solve can afford ``_MIN_SOLVE_SECONDS``.
        """
        if seconds_left is None:
            return mip_time_limit, mip_interval
        while remaining_solves > 1 and seconds_left / remaining_solves < _MIN_SOLVE_SECONDS:
            mip_interval *= 2
            remaining_solves = max(1, remaining_solves // 2)
        share = seconds_left / max(1, remaining_solves)
        return max(_MIN_SOLVE_SECONDS, min(mip_time_limit, share)), mip_interval

    # ----------------------------------------------------------- the loop

    def learn_pooled_with_milp(
        self,
        prepared_problems: Sequence[Tuple[object, Sequence[Path], Sequence[str], Sequence[str]]],
        milp_round: MilpRoundFn,
        epochs: int,
        gamma: float,
        lambda_: float,
        augment: bool,
        pre_mip_epochs: int = 50,
        mip_interval: int = 1,
        mip_traces: int = 3,
        mip_time_limit: float = 60.0,
        agreement_stop: float = 1.0,
        timeout_check: Optional[Callable[[], bool]] = None,
        seconds_left: Optional[Callable[[], float]] = None,
    ) -> Dict:
        """Pooled ROSAME-I training with interleaved MILP rounds.

        Args:
            prepared_problems: ``(problem, image_paths, action_strings,
                final_state_predicates)`` per trajectory.
            milp_round: solves one MILP over the traces it is given.
            pre_mip_epochs: warmup epochs before the first solve.
            mip_interval: solve every this many epochs after warmup.
            mip_traces: how many of each epoch's shuffled order to hand the solver.
            mip_time_limit: the configured per-solve ceiling, before budgeting.
            agreement_stop: stop once ROSAME/MILP model agreement reaches this.
            timeout_check: consulted between epochs; cannot interrupt a solve.
            seconds_left: fold budget still available, used to size each solve.

        Returns:
            A report: per-round history, final solution, stop reason, final loss,
            and any divergence from the configured schedule.
        """
        _set_seed(self.seed)
        traces = self.prepare_traces(prepared_problems)
        report: Dict = {
            "rounds": [], "stop_reason": "epochs_exhausted", "final_solution": None,
            "final_agreement": None, "final_loss": None, "n_traces": len(traces),
            "psi": self.psi, "pre_mip_epochs": pre_mip_epochs,
            "mip_interval_configured": mip_interval, "mip_interval_used": mip_interval,
            "mip_traces": mip_traces, "mip_time_limit_configured": mip_time_limit,
        }
        if not traces:
            report["stop_reason"] = "no_usable_traces"
            return report

        order = list(range(len(traces)))
        for epoch in range(epochs):
            random.shuffle(order)
            for i in order:
                self._train_step(traces[i], gamma, lambda_, augment)
            self._age_state_labels()

            if self._should_solve(epoch, pre_mip_epochs, mip_interval):
                mip_interval = self._run_round(
                    report, traces, order, epoch, milp_round, epochs,
                    pre_mip_epochs, mip_interval, mip_traces, mip_time_limit,
                    agreement_stop, seconds_left,
                )
                report["mip_interval_used"] = mip_interval
                if report["stop_reason"] == "agreement_reached":
                    break
            if timeout_check is not None and timeout_check():
                report["stop_reason"] = "timeout"
                break

        report["final_loss"] = self._total_loss(traces, gamma, lambda_)
        return report

    @staticmethod
    def _should_solve(epoch: int, pre_mip_epochs: int, mip_interval: int) -> bool:
        """True on an epoch that ends with a MILP round."""
        done = epoch + 1
        return done >= pre_mip_epochs and (done - pre_mip_epochs) % mip_interval == 0

    def _run_round(
        self,
        report: Dict,
        traces: List[_PreparedTrace],
        order: List[int],
        epoch: int,
        milp_round: MilpRoundFn,
        epochs: int,
        pre_mip_epochs: int,
        mip_interval: int,
        mip_traces: int,
        mip_time_limit: float,
        agreement_stop: float,
        seconds_left: Optional[Callable[[], float]],
    ) -> int:
        """Solve once over a FIFO prefix of this epoch's order; return the interval.

        Only the selected traces are relabelled. The rest keep their previous
        state labels and go on decaying, rather than being zeroed.
        """
        remaining = max(1, (epochs - epoch - 1) // mip_interval + 1)
        limit, mip_interval = self._solve_time_limit(
            mip_time_limit, mip_interval, remaining,
            seconds_left() if seconds_left is not None else None,
        )
        selected = [traces[i] for i in order[:mip_traces]]
        model_labels, state_labels, agreement, stats, solution = milp_round(selected, limit)

        report["rounds"].append({
            "epoch": epoch,
            "traces": [t.name for t in selected],
            "time_limit_seconds": round(limit, 2),
            "agreement": round(agreement, 4),
            **{k: stats[k] for k in ("exit_status", "solve_time_seconds",
                                     "objective_value") if k in stats},
        })
        if solution is not None:
            report["final_solution"] = solution
            report["final_agreement"] = agreement
            self.set_model_labels(model_labels)
            self.set_state_labels(state_labels)
            if agreement >= agreement_stop:
                report["stop_reason"] = "agreement_reached"
        return mip_interval
