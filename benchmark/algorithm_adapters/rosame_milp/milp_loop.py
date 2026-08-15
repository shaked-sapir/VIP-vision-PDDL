"""The ROSAME+MILP iterative training loop (paper Sec. 6, "Integrating MILP").

Faithful structure (parameters confirmed against upstream ``train_common.py``,
see vendor/UPSTREAM.md): train the model alone for ``pre_mip_epochs`` (upstream
50), then every ``mip_interval`` epochs (upstream 1) solve a MILP from the
current predictions and use its solution as pseudo-labels for further training.

Simulation-mode specialization: states and actions are *data* here, not network
outputs, so of the paper's three pseudo-label channels only the action-model
channel survives. Per upstream code (``dl/model.py``), the model-channel CE is
**undecayed** — labels are simply replaced at each solve (ψ=0.99 applies only
to the state/action channels, which do not exist in this setting).

The loop runs on the pooled schedule (the paper trains pooled batches); the
per-trajectory schedule is not supported for the iterative variant.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.optim as optim

from benchmark.algorithm_adapters.po_rosame_runner import PORosame_Runner, _BATCH_SZ
from benchmark.algorithm_adapters.rosame_milp.model_bridge import model_cross_entropy

# One MILP round: called with the current model, returns
# (labels per schema, agreement in [0,1], solve stats dict, decoded solution).
MilpRoundFn = Callable[[], Tuple[Dict[str, "torch.Tensor"], float, Dict, object]]


class MilpPORosame(PORosame_Runner):
    """PORosame with an additional model-CE term toward MILP pseudo-labels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_labels: Optional[Dict[str, torch.Tensor]] = None

    def set_model_labels(self, labels: Optional[Dict[str, torch.Tensor]]) -> None:
        self._model_labels = labels

    def _model_ce(self) -> Optional[torch.Tensor]:
        """Cross-entropy of each schema's 4-way distribution vs the pseudo-labels."""
        return model_cross_entropy(self.rosame.action_schemas, self._model_labels)

    def _train_step(
        self,
        state_1: torch.Tensor,
        executed_actions: torch.Tensor,
        state_2: torch.Tensor,
        optimizer: optim.Optimizer,
    ) -> float:
        """One optimizer step: base ROSAME loss + (optional) model pseudo-CE.

        Reimplements ``PORosame_Runner._train_step`` (backward/step are inside
        it, so the CE term cannot be appended externally) and keeps the base
        loss byte-identical.
        """
        import torch.nn.functional as F

        optimizer.zero_grad()
        precon, addeff, deleff = self.rosame.build(executed_actions)
        assert precon.shape[1] == state_2.shape[1], (
            f"grounding/proposition mismatch: build={precon.shape[1]} "
            f"vs encoded={state_2.shape[1]} (re-ground the matching problem first)"
        )
        preds = state_1 * (1 - deleff) + (1 - state_1) * addeff
        loss = F.mse_loss(preds, state_2, reduction="sum")
        validity_constraint = (1 - state_1) * precon
        loss += F.mse_loss(
            validity_constraint,
            torch.zeros(validity_constraint.shape, dtype=validity_constraint.dtype),
            reduction="sum",
        )
        loss += 0.2 * F.mse_loss(
            precon, torch.ones(precon.shape, dtype=precon.dtype), reduction="sum"
        )
        ce = self._model_ce()
        if ce is not None:
            loss = loss + ce
        loss.backward()
        optimizer.step()
        return loss.item()

    # ------------------------------------------------------------ the loop

    def learn_pooled_with_milp(
        self,
        prepared: List[Tuple[object, object]],
        milp_round: MilpRoundFn,
        epochs: int = 100,
        pre_mip_epochs: int = 50,
        mip_interval: int = 1,
        agreement_stop: float = 1.0,
    ) -> Dict:
        """Pooled training with interleaved MILP rounds.

        Args:
            prepared: ``(problem, observation)`` pairs.
            milp_round: solves a MILP from the current model; returns
                ``(labels, agreement, stats, solution)``.
            epochs: total training epochs.
            pre_mip_epochs: warmup epochs without MILP (upstream: 50).
            mip_interval: solve every this many epochs after warmup (upstream: 1).
            agreement_stop: stop once ROSAME/MILP agreement reaches this level.

        Returns:
            Report dict: per-round history, final solution + labels, stop reason.
        """
        cached: List[Tuple[object, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for problem, observation in prepared:
            self.add_problem(problem)
            encoded = self._encode_observation(observation)
            if encoded is None:
                continue
            cached.append((problem, *encoded))

        report: Dict = {"rounds": [], "stop_reason": "epochs_exhausted",
                        "final_solution": None, "final_agreement": None}
        if not cached:
            report["stop_reason"] = "no_usable_traces"
            return report

        optimizer = self._build_optimizer()
        order = list(range(len(cached)))

        for epoch in range(epochs):
            random.shuffle(order)
            loss_final = 0.0
            for i in order:
                problem, s1, a, s2 = cached[i]
                self.problem = problem
                self.ground_new_trajectory()
                loss_final += self._train_step(s1, a, s2, optimizer) / _BATCH_SZ
            if epoch % 10 == 0:
                print(f"Epoch {epoch} RESULTS: Pooled average loss: {loss_final:.10f}")

            past_warmup = epoch + 1 >= pre_mip_epochs
            if past_warmup and (epoch + 1 - pre_mip_epochs) % mip_interval == 0:
                labels, agreement, stats, solution = milp_round()
                report["rounds"].append({
                    "epoch": epoch,
                    "agreement": round(agreement, 4),
                    **{k: stats[k] for k in ("exit_status", "solve_time_seconds",
                                             "objective_value") if k in stats},
                })
                if solution is not None:
                    report["final_solution"] = solution
                    report["final_agreement"] = agreement
                    self.set_model_labels(labels)
                if agreement >= agreement_stop and solution is not None:
                    report["stop_reason"] = "agreement_reached"
                    return report

        return report
