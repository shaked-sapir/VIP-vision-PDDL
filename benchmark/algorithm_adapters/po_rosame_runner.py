import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

try:
    from amlgym.algorithms.rosame.experiment_runner.rosame_runner import Rosame_Runner
except ModuleNotFoundError:
    # AMLGym's algorithms package imports optional agents at init-time.
    # If one optional dependency is missing, import ROSAME directly.
    import amlgym

    rosame_root = Path(amlgym.__file__).resolve().parent / "algorithms" / "rosame"
    if str(rosame_root) not in sys.path:
        sys.path.insert(0, str(rosame_root))
    from experiment_runner.rosame_runner import Rosame_Runner

if TYPE_CHECKING:  # import-cycle-free: only needed for the annotation
    from benchmark.algorithm_adapters.anytime_snapshots import SnapshotWriter


# Matches the vendored ``learn_rosame`` batch size, so pooled and per-trajectory
# loss logs are printed on the same (per-1000) scale.
_BATCH_SZ = 1000


class PORosame_Runner(Rosame_Runner):

    def add_problem(self, problem):
        """Build the ROSAME domain model once; re-ground for subsequent problems.

        Overrides the vendored ``Rosame_Runner.add_problem``, which rebuilds the
        ``Domain_Model`` (via ``prepare_rosame``) on every call — re-initializing
        the action-schema networks and discarding everything learned from earlier
        trajectories. As a result, per-trajectory learning effectively kept only
        the last trajectory's model.

        The original ROSAME API (xikaioliver/ROSAME, ``models/rosame.py``) builds
        the domain model once and supports regrounding on different object sets:
        "domain models can be regrounded on different sets of objects. This will
        not affect the learned lifted model." This override restores that
        semantics: build on the first problem, only re-ground afterwards.
        """
        self.problem = problem
        if self.rosame is None:
            self.rosame = self.prepare_rosame()  # first problem: build schemas + ground
        else:
            self.ground_new_trajectory()         # later problems: re-ground only

    def prepare_rosame_data(self,observation):
        """
        Prepares structured data from traces to be used within the ROSAME framework.
        Handles partial observability by encoding predicates as:
        - 1.0 if predicate is positive and unmasked
        - 0.5 if predicate is masked (uncertain)
        - 0.0 if predicate is negative and unmasked

        returns:
        - encoded pre_state
        - encoded next_state
        - encoded action
        """
        steps_action = []
        steps_state1, steps_state2 = [], []

        for component in observation.components:
            action_lst = component.grounded_action_call.__str__()[1:-1].split()
            if len(action_lst) != len(set(action_lst)):
                continue

            steps_action.append(self.check_action(component.grounded_action_call.__str__()[1:-1]))

            # Process previous state
            pre_positive_preds = []  # Positive predicates that are unmasked
            pre_masked_preds = []    # Predicates that are masked (pos or neg)
            pre_negative_preds = []  # Negative predicates that are unmasked

            for _, val in component.previous_state.state_predicates.items():
                for pred in val:
                    pred_str = self.check_predicate(pred.untyped_representation[1:-1])

                    if pred.is_masked:
                        pre_masked_preds.append(pred_str)
                    elif pred.is_positive:
                        pre_positive_preds.append(pred_str)
                    else:  # is_negative and not masked
                        pre_negative_preds.append(pred_str)

            # Process next state
            next_positive_preds = []  # Positive predicates that are unmasked
            next_masked_preds = []    # Predicates that are masked (pos or neg)
            next_negative_preds = []  # Negative predicates that are unmasked

            for _, val in component.next_state.state_predicates.items():
                for pred in val:
                    pred_str = self.check_predicate(pred.untyped_representation[1:-1])

                    if pred.is_masked:
                        next_masked_preds.append(pred_str)
                    elif pred.is_positive:
                        next_positive_preds.append(pred_str)
                    else:  # is_negative and not masked
                        next_negative_preds.append(pred_str)

            # Encode states with partial observability
            # 1.0 if in positive_preds, 0.5 if in masked_preds, 0.0 if in negative_preds
            state1 = []
            for p in self.rosame.propositions:
                if p in pre_positive_preds:
                    state1.append(1.0)
                elif p in pre_masked_preds:
                    state1.append(0.5)
                else:  # Either in negative_preds or not observed at all
                    state1.append(0.0)

            state2 = []
            for p in self.rosame.propositions:
                if p in next_positive_preds:
                    state2.append(1.0)
                elif p in next_masked_preds:
                    state2.append(0.5)
                else:  # Either in negative_preds or not observed at all
                    state2.append(0.0)

            steps_state1.append(state1)
            steps_state2.append(state2)

        return steps_state1, steps_action, steps_state2

    # ------------------------------------------------------------ training primitives

    def _encode_observation(
        self, observation
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Encode one (masked) observation into (state1, action, state2) tensors.

        Mirrors the tensor-prep head of the vendored ``learn_rosame``. The
        proposition vectors are sized/ordered by the *currently grounded* object
        set, so the caller must ground the matching problem before encoding.

        Returns:
            ``(s1, a, s2)`` float/long tensors, or ``None`` if the trajectory
            yields no usable transitions (e.g. every action had duplicate args).
        """
        steps_state1, steps_action, steps_state2 = self.prepare_rosame_data(observation)
        if not steps_state1:
            return None
        s1 = torch.tensor(np.array(steps_state1)).float()
        a = torch.tensor(np.array(steps_action))
        s2 = torch.tensor(np.array(steps_state2)).float()
        return s1, a, s2

    def _build_optimizer(self, lr: float = 1e-3) -> optim.Optimizer:
        """One Adam over every action-schema MLP's params (grounding-independent).

        Regrounding on new object sets does not recreate the schema networks
        (documented ROSAME semantics), so a single optimizer built here stays
        valid across the per-trajectory regroundings the pooled loop performs.
        """
        param_groups = [
            {"params": schema.parameters(), "lr": lr}
            for schema in self.rosame.action_schemas
        ]
        return optim.Adam(param_groups)

    def _train_step(
        self,
        state_1: torch.Tensor,
        executed_actions: torch.Tensor,
        state_2: torch.Tensor,
        optimizer: optim.Optimizer,
    ) -> float:
        """One optimizer step of the ROSAME loss (faithful to ``learn_rosame``)."""
        optimizer.zero_grad()
        precon, addeff, deleff = self.rosame.build(executed_actions)
        # Defensive: current grounding must match the encoding's proposition dim.
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
        loss.backward()
        optimizer.step()
        return loss.item()

    # ------------------------------------------------------------ training loops

    def learn_per_trajectory(
        self,
        prepared: List[Tuple[object, object]],
        epochs: int = 100,
        snapshot: Optional["SnapshotWriter"] = None,
        step_offset: int = 0,
    ) -> int:
        """Continual schedule (default): fully train each trace before the next.

        Equivalent to the vendored ``learn_rosame`` — fresh optimizer per trace,
        schema weights carrying over, contiguous batches of ``_BATCH_SZ`` in file
        order — but written out here rather than delegated, because the vendored
        loop offers no per-epoch hook and this is the schedule the benchmark
        actually runs. ``test_po_rosame_runner.test_local_loop_matches_vendored``
        pins the two together; that test is what licenses the rewrite, so do not
        drop it.

        Args:
            prepared: ``(problem, observation)`` pairs, in order.
            epochs: Epochs per trajectory.
            snapshot: Optional writer for anytime curves.
            step_offset: Starting value for the global step counter.

        Returns:
            The global step counter after the last epoch.
        """
        step = step_offset
        for traj_index, (problem, observation) in enumerate(prepared):
            self.add_problem(problem)
            self.ground_new_trajectory()
            encoded = self._encode_observation(observation)
            if encoded is None:
                # The vendored loop would spin `epochs` times over an empty
                # DataLoader here; skipping is the same no-op on the weights.
                continue
            s1, a, s2 = encoded

            optimizer = self._build_optimizer()
            for epoch in range(epochs):
                loss_final = 0.0
                for lo in range(0, s1.shape[0], _BATCH_SZ):
                    hi = lo + _BATCH_SZ
                    loss_final += (
                        self._train_step(s1[lo:hi], a[lo:hi], s2[lo:hi], optimizer)
                        / _BATCH_SZ
                    )
                if epoch % 10 == 0:
                    print(f"Epoch {epoch} RESULTS: Average loss: {loss_final:.10f}")

                step += 1
                if snapshot is not None:
                    snapshot.maybe_capture(
                        step=step,
                        trajectory=traj_index,
                        epoch=epoch,
                        render=self._render_pddl,
                    )
        return step

    def learn_pooled(
        self,
        prepared: List[Tuple[object, object]],
        epochs: int = 100,
        snapshot: Optional["SnapshotWriter"] = None,
        step_offset: int = 0,
    ) -> int:
        """Interleaved schedule: one persistent optimizer, all traces per epoch.

        Removes the per-trajectory ordering bias of the continual loop. Because
        plain ROSAME has no fixed-size CV head (only lifted schema MLPs), this
        needs **no** shared object universe — each trace is regrounded to its own
        objects before its build/step, so different problems can be pooled.

        Args:
            prepared: ``(problem, observation)`` pairs.
            epochs: Epochs over the pooled set of traces.
            snapshot: Optional writer for anytime curves.
            step_offset: Starting value for the global step counter.

        Returns:
            The global step counter after the last epoch.
        """
        # Ground + encode each trace once, caching its problem and tensors.
        cached: List[Tuple[object, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for problem, observation in prepared:
            self.add_problem(problem)
            encoded = self._encode_observation(observation)
            if encoded is None:
                continue
            cached.append((problem, *encoded))

        if not cached:
            return step_offset

        step = step_offset
        optimizer = self._build_optimizer()
        order = list(range(len(cached)))
        for epoch in range(epochs):
            random.shuffle(order)
            loss_final = 0.0
            for i in order:
                problem, s1, a, s2 = cached[i]
                # Re-ground to this trace's objects so build() dims match s2.
                self.problem = problem
                self.ground_new_trajectory()
                # Normalized by batch size to match the vendored learn_rosame
                # log scale (which prints loss.item() / batch_sz).
                loss_final += self._train_step(s1, a, s2, optimizer) / _BATCH_SZ
            if epoch % 10 == 0:
                print(f"Epoch {epoch} RESULTS: Pooled average loss: {loss_final:.10f}")

            step += 1
            if snapshot is not None:
                # No single trace owns a pooled epoch, hence -1.
                snapshot.maybe_capture(
                    step=step, trajectory=-1, epoch=epoch, render=self._render_pddl
                )
        return step

    # ------------------------------------------------------------- snapshots

    def _render_pddl(self) -> str:
        """Serialize the current model, for mid-training snapshots.

        Safe to call between optimizer steps: ``rosame_to_pddl`` reaches
        ``Action_Schema.pretty_print``, which runs the schema MLP over a *fixed*
        buffer created at init and argmaxes it. That consumes no RNG, touches no
        grounding, and mutates nothing, so a snapshot cannot perturb the run it
        is observing. ``no_grad`` only avoids building a throwaway graph.
        """
        with torch.no_grad():
            return self.rosame_to_pddl()

    def learn_full(
        self,
        prepared: List[Tuple[object, object]],
        train_per_trajectory: bool = True,
        epochs: int = 100,
        snapshot: Optional["SnapshotWriter"] = None,
    ) -> str:
        """Run the selected schedule and return the learned PDDL domain string.

        When ``snapshot`` is given, its clock starts here and is closed on the
        way out, so the recorded times cover training only — not the caller's
        parsing or evaluation.
        """
        if snapshot is not None:
            snapshot.start()
        try:
            if train_per_trajectory:
                self.learn_per_trajectory(prepared, epochs, snapshot=snapshot)
            else:
                self.learn_pooled(prepared, epochs, snapshot=snapshot)
        finally:
            if snapshot is not None:
                snapshot.close()
        return self.rosame_to_pddl()
