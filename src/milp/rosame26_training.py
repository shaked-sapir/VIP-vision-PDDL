"""The ICAPS-26 training loop, re-expressed over ragged folds (plan §3).

:class:`Rosame26Trainer` is :class:`~src.milp.rosame26_model.Rosame26Goal` with
``train`` / ``_run_training`` replaced and **nothing else**. Every other method
of the vendored ``dl.network.Network`` — ``build``, ``save``, ``load``, the
encoder/decoder/action-head construction — runs verbatim.

``dump_actions`` is the one exception and is **overridden to raise**. It writes
``domain_model.pddl`` through the vendored ``extract_pddl``, whose signatures are
in sorted-type rather than PDDL order (§4.2b) — a file that scores 0.20 to 0.82
depending on the domain. Its only upstream call site is the every-ninth-epoch
validation block this module already drops, so nothing reaches it today;
overriding it keeps that true if a later phase reinstates any of that block.
:func:`~src.milp.rosame26_emitter.emit_pddl` is the emitter for this arm.

WHAT THE OVERRIDE RE-EXPRESSES. Upstream's ``network.py:225-320``, minus the
logging, is::

    for epoch in range(epochs):
        trace_selector.clear()
        for batch in loader:
            outputs = net(img_traces, action_traces, inits, goals)
            loss = model.loss(outputs, targets, indices, convertor.pseudo_labels)
            step
            if epoch >= pre_mip_epoch and epoch % mip_interval == 0:
                trace_selector.update(indices, outputs['z'], outputs['a'], inits, goals)
        if epoch >= pre_mip_epoch and epoch % mip_interval == 0:
            convertor.run_fixer(trace_selector, mip_time_limit)

The gate is **one** condition read at two sites, so :func:`is_mip_epoch` is
evaluated once per epoch into ``solving`` and used twice; the two cannot drift.

WHAT IT DELIBERATELY DROPS, each with the section that says so:

* **The transform-selection block** (``network.py:218-231``). Every upstream
  augmentation hard-asserts a render layout ours does not have, and ``gripper``
  at resize ``[64, 96]`` is the one combination that would pass the guard and
  then permute a scene with no cell structure while leaving the symbolic labels
  intact (§3, deviation 5). ``TraceDataset(...)`` is constructed with no
  transform and there is no parameter that reinstates one.
* **``num_workers=64, prefetch_factor=8, persistent_workers=True``** (§1.1,
  deviation 13). Sized from the fold instead; ``num_workers`` defaults to 0
  because a fold is 3-9 traces.
* **``self.evaluate(val_data)``**, the every-ninth-epoch validation. It opens
  with ``compute_permutation()`` → ``model_permutation``, which §0.1 bypasses
  under option B, and it builds a ``Convertor`` this arm does not use (§6.1).
* **``SummaryWriter``**, tensorboard not being a project dependency. The
  per-epoch record upstream writes to the event file is returned instead, so
  "is the arm undertrained?" is answerable from one run (§9.8).

MILP AS A COLLABORATOR. Upstream reaches for ``Convertor.run_fixer``, whose
encoder takes ``max_t`` from the *first* trace in the bundle and so mis-encodes
every other trace in a ragged fold (§6.1). Both solve sites therefore route
through an injected :class:`MipRepairer` instead. With none injected the loop is
DL-only, and :func:`mip_epoch_count` is checked **before** the first epoch so a
misconfigured run fails at second zero rather than at epoch 50.
``pre_mip_epoch >= epoch`` is upstream's own way of asking for that, and it is
what makes the ``ROSAME-I_26`` arm the same code as ``ROSAME-I_MILP_26``.

METRICS ARE NOT ACCUMULATED. ``_register_metrics`` still registers ``state_acc``
and ``action_acc``, and this loop calls neither. Interior ``state_traces`` rows
are zero filler (§4.3, deviation 10), so ``state_acc`` scores predictions against
padding; and under option B ``a`` *is* the observed action, so ``action_acc`` is
1.0 by construction. Reporting either as an accuracy would be wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import src.milp  # noqa: F401  (vendor sys.path bootstrap)

from dl.util.dataset import TraceDataset

from src.milp.rosame26_model import Rosame26Goal
from src.milp.trace_tensors import PaddedTraces


UPSTREAM_PARAMETERS: Dict[str, Any] = {
    "aae_width": 1000,
    "aae_depth": 3,
    "feature_dim": 512,
    "hidden_dim": 256,
    "lambda": 0.2,
    "gamma": 10,
    "beta_pred": 1,
    "beta_app": 1,
    "beta_reconst": 0,
    "optimizer": "Adam",
    "lr": 1e-4,
    "mip_interval": 1,
    "mip_traces": 3,
    "pseudo_weight_decay": 0.99,
    "mip_time_limit": 60,
    "DL_to_MIP": ["state", "action", "model"],
}
"""``train_common.py``'s dict, verbatim, minus the four entries we do not match.

Those four are named in :data:`DEVIATING_PARAMETERS`. ``lr`` and ``DL_to_MIP``
are single-element tuning-sweep lists upstream; unwrapped here.
"""


DEVIATING_PARAMETERS: Dict[str, str] = {
    "epoch": "5000 is the code default; the grid runs a calibrated value inside "
    "a 600 s cell budget (§1.2). Report the count actually used.",
    "batch_size": "128 upstream over a large corpus; a fold is 3-9 traces, so "
    "min(batch_size, N) is the whole fold (§1.1).",
    "MIP_to_DL": "drops 'action' under option B — the action is observed, so "
    "the solver has nothing to teach the action head (§0). DL_to_MIP keeps it.",
    "cp_type": "cp-sat rather than mip-gurobi, for want of a licence (§6).",
}
"""The four starred rows of §6, each mapped to why it is starred."""


def select_device() -> str:
    """``"cuda"`` when available, else ``"cpu"``. MPS is excluded (§1.3)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_parameters(
    *,
    domain: str,
    domain_assets_root: str | Path,
    epoch: int,
    batch_size: int = 128,
    pre_mip_epoch: int = 50,
    device: Optional[str] = None,
    seed: int = 0,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """:data:`UPSTREAM_PARAMETERS` plus everything the skipped CLI layer supplied.

    ``prop_dim`` and ``action_dim`` are absent on purpose: ``ROSAMEMixin`` writes
    them from the domain model during ``build``. ``data_loc``, ``mean`` and
    ``std`` are absent because nothing on this arm's path reads them — a
    ``KeyError`` is the wanted failure if that stops being true.

    Args:
        domain: The domain key, which selects the generated spec assets.
        domain_assets_root: Where :func:`write_grounding_assets` put them.
        epoch: Total epochs. ``pre_mip_epoch >= epoch`` is the DL-only arm.
        batch_size: Capped at the fold size by :func:`resolve_batch_size`.
        pre_mip_epoch: First epoch at which the MILP may run.
        device: Defaults to :func:`select_device`.
        seed: Seeds weight init and the loader's shuffle.
        num_workers: DataLoader workers. 0 for a fold-sized corpus.
    """
    return dict(
        UPSTREAM_PARAMETERS,
        domain=domain,
        domain_assets_root=str(domain_assets_root),
        epoch=epoch,
        batch_size=batch_size,
        pre_mip_epoch=pre_mip_epoch,
        MIP_to_DL=["state", "model"],
        cp_type="cp-sat",
        device=device if device is not None else select_device(),
        seed=seed,
        num_workers=num_workers,
    )


class MipRepairer(Protocol):
    """The MILP half of the loop, in the shape ``_run_training`` uses it.

    Upstream splits this across a ``TraceSelector`` (``clear`` / ``update``) and
    a ``Convertor`` (``pseudo_labels`` / ``run_fixer``). One object here, because
    ``src/milp/encoder.py`` replaces both and the selector is an implementation
    detail of which traces the solve covers.
    """

    @property
    def pseudo_labels(self) -> Any:
        """The label store the loss reads; ``PseudoLabels``-shaped."""

    def clear(self) -> None:
        """Drop the traces collected for the previous solve."""

    def update(
        self,
        indices: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
        inits: torch.Tensor,
        goals: torch.Tensor,
    ) -> None:
        """Offer a batch's outputs for inclusion in the next solve."""

    def run_fixer(self, time_limit: float) -> Optional[float]:
        """Solve over the collected traces and write pseudo-labels.

        Returns:
            The ``mip_gt_dist`` diagnostic, or ``None`` when unavailable.
        """


def is_mip_epoch(epoch: int, parameters: Mapping[str, Any]) -> bool:
    """Upstream's gate, ``network.py:272`` and ``:303``, as one function."""
    return (
        epoch >= parameters["pre_mip_epoch"]
        and epoch % parameters["mip_interval"] == 0
    )


def mip_epoch_count(parameters: Mapping[str, Any]) -> int:
    """How many epochs of the run would solve. Zero means a DL-only run."""
    return sum(
        is_mip_epoch(epoch, parameters) for epoch in range(parameters["epoch"])
    )


def resolve_batch_size(requested: int, n_traces: int) -> int:
    """``min(requested, n_traces)`` — upstream's own line, ``network.py:233``.

    Raises:
        ValueError: if the fold is empty, which would otherwise reach
            ``DataLoader`` as ``batch_size=0``.
    """
    if n_traces < 1:
        raise ValueError("a fold with no traces cannot be trained on")
    return min(requested, n_traces)


def build_loader(
    dataset: TraceDataset, batch_size: int, num_workers: int, seed: int
) -> DataLoader:
    """Upstream's loader with the worker settings sized for a fold (§1.1).

    The shuffle is seeded because the ``TraceSelector`` fills FIFO from the first
    batches, so batch order decides which traces the first solve covers.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )


class Rosame26Trainer(Rosame26Goal):
    """``Rosame26Goal`` trained on a ragged fold, with the MILP injected."""

    def __init__(
        self,
        path: str | Path,
        parameters: Dict[str, Any],
        mip_repairer: Optional[MipRepairer] = None,
        stop_check: Optional[Callable[[List[Dict[str, float]]], bool]] = None,
    ) -> None:
        """
        Args:
            path: The run directory the vendored ``Network`` writes into.
            parameters: :func:`default_parameters`, or a superset of it.
            mip_repairer: The solve collaborator. ``None`` is a DL-only run and
                requires ``pre_mip_epoch >= epoch``.
            stop_check: Called with the history after each epoch; a truthy result
                ends the run early. ``None`` runs every configured epoch. The
                *policy* lives in the caller — this only provides the hook, so
                ``parameters["epoch"]`` stays a ceiling in every mode.
        """
        super().__init__(str(path), parameters)
        self.mip_repairer = mip_repairer
        self.stop_check = stop_check
        self.history: List[Dict[str, float]] = []
        self.stopped_early: bool = False

    def train(self, train_data: PaddedTraces) -> List[Dict[str, float]]:
        """Build, then run the loop. Returns the per-epoch record.

        ``input_shape`` is taken from the images exactly as upstream takes it,
        so every trace in a run must share ``(C, H, W)`` — which the padding in
        :mod:`src.milp.trace_tensors` guarantees.
        """
        input_shape = train_data.images.shape[1:]
        self.input_shape = input_shape
        torch.manual_seed(self.parameters["seed"])
        self.build(input_shape)
        self.build_aux(input_shape)
        self.net.to(torch.device(self.parameters["device"]))
        self.optimizer = getattr(optim, self.parameters["optimizer"])(
            self.net.parameters(), lr=self.parameters["lr"]
        )
        return self._run_training(train_data)

    def dump_actions(self, *args: Any, **kwargs: Any) -> None:
        """Not supported; use :func:`~src.milp.rosame26_emitter.emit_pddl`.

        Raises:
            NotImplementedError: always. The vendored implementation emits
                through ``extract_pddl``, which writes ``:parameters`` in
                sorted-type order and its literals with bare variables.
        """
        raise NotImplementedError(
            "Rosame26Trainer does not dump actions through the vendored "
            "extract_pddl, whose signatures are permuted against the domain "
            "PDDL; use src.milp.rosame26_emitter.emit_pddl"
        )

    def resume(self, *args: Any, **kwargs: Any) -> None:
        """Not supported.

        Raises:
            NotImplementedError: always. The vendored ``resume`` calls
                ``_run_training(train_data, val_data)``, an arity this override
                does not have, and reaches ``model_permutation`` through
                ``evaluate``.
        """
        raise NotImplementedError("Rosame26Trainer does not support resume()")

    def _pseudo_labels(self) -> Optional[Any]:
        """The repairer's label store, or ``None`` on a DL-only run.

        Read per batch rather than cached, so a repairer that replaces the store
        during ``run_fixer`` is seen.
        """
        return None if self.mip_repairer is None else self.mip_repairer.pseudo_labels

    def _run_training(self, train_data: PaddedTraces) -> List[Dict[str, float]]:
        """Upstream's loop over a padded fold, with lengths carried to the loss.

        Raises:
            ValueError: if the schedule would solve but no repairer was injected.

        Returns:
            One record per epoch: the mean of each loss term over the epoch's
            batches, plus ``epoch`` and, on solving epochs, ``mip_gt_dist``.
        """
        parameters = self.parameters
        if self.mip_repairer is None and mip_epoch_count(parameters) > 0:
            raise ValueError(
                f"the schedule solves on {mip_epoch_count(parameters)} epochs "
                f"(pre_mip_epoch={parameters['pre_mip_epoch']}, "
                f"epoch={parameters['epoch']}) but no mip_repairer was given; "
                f"set pre_mip_epoch >= epoch for a DL-only run"
            )

        device = torch.device(parameters["device"])
        lengths = train_data.lengths.to(device)
        dataset = TraceDataset(
            (train_data.images, train_data.state_traces, train_data.action_traces)
        )
        loader = build_loader(
            dataset,
            resolve_batch_size(parameters["batch_size"], len(dataset)),
            parameters["num_workers"],
            parameters["seed"],
        )

        self.history = []
        self.stopped_early = False
        for epoch in range(parameters["epoch"]):
            self.net.train()
            solving = is_mip_epoch(epoch, parameters)
            if self.mip_repairer is not None:
                self.mip_repairer.clear()

            running: Dict[str, float] = {}
            n_batches = 0
            for img_traces, state_traces, action_traces, indices in loader:
                n_batches += 1
                img_traces = img_traces.to(device)
                state_traces = state_traces.to(device)
                action_traces = action_traces.to(device)
                inits = state_traces[:, 0, :]
                goals = state_traces[:, -1, :]
                targets = [state_traces, action_traces]

                self.optimizer.zero_grad()
                outputs = self.net(img_traces, action_traces, inits, goals)
                loss_dict = self.loss(
                    outputs,
                    targets,
                    indices,
                    self._pseudo_labels(),
                    lengths[indices],
                )
                loss_dict["total_loss"].backward()
                self.optimizer.step()

                if solving:
                    self.mip_repairer.update(
                        indices, outputs["z"], outputs["a"], inits, goals
                    )
                for key, value in loss_dict.items():
                    running[key] = running.get(key, 0.0) + value.detach().item()

            record: Dict[str, float] = {
                key: value / n_batches for key, value in running.items()
            }
            record["epoch"] = epoch
            if solving:
                distance = self.mip_repairer.run_fixer(parameters["mip_time_limit"])
                if distance is not None:
                    record["mip_gt_dist"] = distance
            self.history.append(record)
            self.epoch = epoch + 1

            if self.stop_check is not None and self.stop_check(self.history):
                self.stopped_early = True
                break

        return self.history
