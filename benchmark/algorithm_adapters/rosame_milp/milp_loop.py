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

from benchmark.algorithm_adapters.best_checkpoint import BestModelTracker
from benchmark.algorithm_adapters.po_rosame_runner import (
    DEFAULT_BATCH_SIZE,
    PORosame_Runner,
    StopCheck,
    TimeoutCheck,
    _BATCH_SZ,
    batched_steps,
)
from benchmark.algorithm_adapters.rosame_milp.model_bridge import model_cross_entropy
from src.milp.loss_convergence import (
    AGREEMENT_REACHED,
    CONVERGED,
    EPOCHS_EXHAUSTED,
    NO_USABLE_TRACES,
    TIMEOUT,
)

# One MILP round: called with the current model, returns
# (labels per schema, agreement in [0,1], solve stats dict, decoded solution).
MilpRoundFn = Callable[[], Tuple[Dict[str, "torch.Tensor"], float, Dict, object]]


def base_loss_divisor(n_transitions: int, normalize_base_loss: bool) -> int:
    """The divisor applied to each base loss term of one optimizer step.

    ``normalize_base_loss`` divides by the step's transition count (ICAPS-26's
    ``B * (T+1)``); off, the terms stay ICAPS-24's raw sums.
    """
    if not normalize_base_loss:
        return 1
    return max(int(n_transitions), 1)


class MilpPORosame(PORosame_Runner):
    """PORosame with an additional model-CE term toward MILP pseudo-labels."""

    def __init__(self, *args, normalize_base_loss: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize_base_loss = normalize_base_loss
        self._model_labels: Optional[Dict[str, torch.Tensor]] = None
        # (base, ce) of the most recent optimizer step; ce is 0.0 before the first solve.
        self.last_loss_parts: Tuple[float, float] = (0.0, 0.0)

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
        it, so the CE term cannot be appended externally).

        With ``normalize_base_loss`` on, the three base terms are divided by
        the transition count, which the DL-only arm does not do; off, they are
        the raw sums. ICAPS-24 has no MILP and sums them raw
        (``train.py:84-101``); ICAPS-26 normalises every term by ``B * (T+1)``
        before adding its pseudo-label CE (``dl/model.py:260-261``). Summing
        24's raw base with 26's CE -- which is a mean over schema rows, so O(1)
        -- leaves the CE two to three orders of magnitude smaller than what it
        competes with, and the MILP cannot move the network: measured over 251
        rounds, agreement stayed at a single distinct value in two domains and
        drifted back to its starting value in two more.

        Divides by transitions rather than by cells because that is what
        ``B * (T+1)`` counts upstream; ``symbol_dim`` never enters its divisor.
        """
        import torch.nn.functional as F

        optimizer.zero_grad()
        precon, addeff, deleff = self.rosame.build(executed_actions)
        assert precon.shape[1] == state_2.shape[1], (
            f"grounding/proposition mismatch: build={precon.shape[1]} "
            f"vs encoded={state_2.shape[1]} (re-ground the matching problem first)"
        )
        # Upstream 26's B * (T+1): the transitions this step covers.
        n_transitions = base_loss_divisor(int(state_2.shape[0]), self.normalize_base_loss)
        preds = state_1 * (1 - deleff) + (1 - state_1) * addeff
        loss = F.mse_loss(preds, state_2, reduction="sum") / n_transitions
        validity_constraint = (1 - state_1) * precon
        loss += F.mse_loss(
            validity_constraint,
            torch.zeros(validity_constraint.shape, dtype=validity_constraint.dtype),
            reduction="sum",
        ) / n_transitions
        loss += 0.2 * F.mse_loss(
            precon, torch.ones(precon.shape, dtype=precon.dtype), reduction="sum"
        ) / n_transitions
        base = loss.item()
        ce = self._model_ce()
        if ce is not None:
            loss = loss + ce
        self.last_loss_parts = (base, 0.0 if ce is None else ce.item())
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
        agreement_stop: Optional[float] = 1.0,
        snapshot=None,
        batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
        stop_check: Optional[StopCheck] = None,
        timeout_check: Optional[TimeoutCheck] = None,
        tracker: Optional[BestModelTracker] = None,
    ) -> Dict:
        """Pooled training with interleaved MILP rounds.

        Stops on the first of: ``stop_check`` (``converged``), ``timeout_check``
        (``timeout``), agreement reaching ``agreement_stop``
        (``agreement_reached``), or ``epochs`` (``epochs_exhausted``).

        The convergence series, and ``tracker``, restart at the first
        *successful* solve: the pseudo-label cross-entropy joins the loss there
        and shifts its scale, so warmup losses are recorded but not scored.

        Args:
            prepared: ``(problem, observation)`` pairs.
            milp_round: solves a MILP from the current model; returns
                ``(labels, agreement, stats, solution)``.
            epochs: total training epochs; a ceiling when ``stop_check`` is given.
            pre_mip_epochs: warmup epochs without MILP (upstream: 50).
            mip_interval: solve every this many epochs after warmup (upstream: 1).
            agreement_stop: stop once ROSAME/MILP agreement reaches this level;
                ``None`` records agreement without stopping on it.
            batch_size: Transitions per optimizer step, pooled across the
                traces that share a grounding. ``None`` steps once per trace.
            snapshot: Optional ``SnapshotWriter``; captures the model and that
                epoch's training loss every Nth epoch. The DL phase only — a
                MILP round does not train.
            stop_check: Called with the post-first-solve loss series after each
                epoch; ``True`` ends training.
            timeout_check: Called after each epoch; ``True`` ends training.
            tracker: Keeps the best post-first-solve checkpoint. The caller
                restores it.

        Returns:
            Report dict: per-round history, final solution, stop reason,
            ``epochs_run``, ``first_solve_epoch``, ``best_epoch``, ``best_loss``,
            and the full ``losses`` / ``base_losses`` / ``ce_losses`` series.
        """
        cached: List[Tuple[object, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for problem, observation in prepared:
            self.add_problem(problem)
            encoded = self._encode_observation(observation)
            if encoded is None:
                continue
            cached.append((problem, *encoded))

        report: Dict = {"rounds": [], "stop_reason": EPOCHS_EXHAUSTED,
                        "final_solution": None, "final_agreement": None,
                        "epochs_run": 0, "first_solve_epoch": None,
                        "best_epoch": None, "best_loss": None,
                        "losses": [], "base_losses": [], "ce_losses": []}
        if not cached:
            report["stop_reason"] = NO_USABLE_TRACES
            return report

        if tracker is not None:
            tracker.bind(self.rosame)
        # Losses since the first successful solve; what stop_check and tracker score.
        scored: List[float] = []
        optimizer = self._build_optimizer()
        # The snapshot for an epoch is taken before that epoch's MILP round, so
        # the agreement it carries is the previous round's -- the most recent
        # value known at capture time. None until the first round has run.
        last_agreement = None
        if snapshot is not None:
            snapshot.start()

        for epoch in range(epochs):
            loss_final = base_final = ce_final = 0.0
            for problem, s1, a, s2 in batched_steps(cached, batch_size, random):
                self.problem = problem
                self.ground_new_trajectory()
                loss_final += self._train_step(s1, a, s2, optimizer) / _BATCH_SZ
                base_final += self.last_loss_parts[0] / _BATCH_SZ
                ce_final += self.last_loss_parts[1] / _BATCH_SZ
            if epoch % 10 == 0:
                print(f"Epoch {epoch} RESULTS: Pooled average loss: {loss_final:.10f}")

            report["epochs_run"] = epoch + 1
            report["losses"].append(loss_final)
            report["base_losses"].append(base_final)
            report["ce_losses"].append(ce_final)
            solved = report["first_solve_epoch"] is not None
            if solved:
                scored.append(loss_final)
                if tracker is not None:
                    tracker.observe(loss_final, epoch)

            if snapshot is not None:
                # No single trace owns a pooled epoch, hence -1.
                snapshot.maybe_capture(
                    step=epoch + 1, trajectory=-1, epoch=epoch,
                    render=self.rosame_to_pddl, loss=loss_final,
                    agreement=last_agreement,
                    base_loss=base_final, ce_loss=ce_final,
                )

            if solved and stop_check is not None and stop_check(scored):
                report["stop_reason"] = CONVERGED
                break
            if timeout_check is not None and timeout_check():
                report["stop_reason"] = TIMEOUT
                break

            past_warmup = epoch + 1 >= pre_mip_epochs
            if past_warmup and (epoch + 1 - pre_mip_epochs) % mip_interval == 0:
                labels, agreement, stats, solution = milp_round()
                report["rounds"].append({
                    "epoch": epoch,
                    "agreement": round(agreement, 4),
                    **{k: stats[k] for k in ("exit_status", "solve_time_seconds",
                                             "objective_value") if k in stats},
                })
                last_agreement = agreement
                if solution is not None:
                    report["final_solution"] = solution
                    report["final_agreement"] = agreement
                    self.set_model_labels(labels)
                    if report["first_solve_epoch"] is None:
                        report["first_solve_epoch"] = epoch
                        if tracker is not None:
                            tracker.reset()
                if (agreement_stop is not None and agreement >= agreement_stop
                        and solution is not None):
                    report["stop_reason"] = AGREEMENT_REACHED
                    break

        if tracker is not None:
            report["best_epoch"] = tracker.best_epoch
            report["best_loss"] = tracker.best_loss
        return report
