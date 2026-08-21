"""The ICAPS-26 network under observed actions and ragged trace lengths.

Two mixins over the vendored ``dl.model.ROSAMEGoal``, composed by
:class:`Rosame26Goal`. Neither touches the architecture, the loss terms or their
weights; both exist because our data regime differs from upstream's in exactly
two ways.

OBSERVED ACTIONS — option B (plan §0). Upstream predicts the action and every
loss term is weighted by that prediction. We observe it. ``a`` is therefore
pinned to the observed one-hot while ``a_logit`` is still computed exactly as
upstream computes it, so the action head keeps its gradient path
(``a_logit -> action -> z_ext -> z -> encoder``) and every ``@`` contraction in
``loss_pred`` / ``loss_app`` / ``loss_prior`` simply selects the observed row.
:class:`ObservedActionNet` does this by overwriting ``outputs['a']`` after the
fact, which is exact rather than approximate: the vendored ``Net.forward``
ignores its ``a`` argument entirely, so nothing else in the output dict can
depend on it. ``test_rosame26_model.py`` pins that invariant.

RAGGED LENGTHS (plan §6.1a). Upstream's corpus is length-homogeneous; ours is
not, so :mod:`src.milp.trace_tensors` right-pads a batch to ``T_max`` and hands
over a per-trace ``lengths``. Zero-action padding makes most of the loss mask
itself — an all-zero row in ``a`` zeroes its own contraction — but four things
break regardless, and :class:`LengthMaskedLossMixin` is exactly those four:

======================  =========================================================
``loss_pred``           the ``gamma``-weighted goal anchor is read at ``a[:, -1]``,
                        which is a pad for every short trace. Gathered at
                        ``a[:, L]`` instead, and the prefix stops at ``L``
                        so that row is not also counted as an interior
                        transition.
``loss_prior`` /        normalised by ``B*(T_max+1)``, which counts pad steps.
``loss_pred`` /         Divided by ``sum(L_i + 1)`` instead.
``loss_app``
``loss_pseudo_a``       ``weight_mask_a`` is initialised to ones, so padded steps
                        train the action head against a self-generated argmax at
                        full weight. Zeroed past each trace's end.
``loss_pseudo_s``       BCEs the full ``[T_max, S]``, filler rows included.
                        Restricted to each trace's live rows.
======================  =========================================================

``loss_app`` needs no masking of its own: its ``gamma`` endpoint is ``a[:, 0]``,
which under right-padding is the first real step of every trace, and its suffix
is action-weighted throughout.

Two things are deliberately *not* masked. ``loss_reconst`` reconstructs the
padded frames as well as the real ones; it is inert at the pinned
``beta_reconst: 0`` but is still reported, so it is not comparable across traces
of different length. ``loss_pseudo_m`` scores the lifted schemas and has no step
dimension at all.

With ``lengths=None``, or with every trace at ``T_max``, every expression here
reduces to the vendored one **bit-identically** — pinned by gate 2
(``test_vendor_net_contract.py``), which is what confines the deviation to the
ragged case upstream never has.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

import src.milp  # noqa: F401  (vendor sys.path bootstrap)

from dl.model import ROSAMEGoal


class ObservedActionNet(nn.Module):
    """The vendored ``Net`` with ``a`` pinned to the observed one-hot (§0).

    The wrapped module computes ``a_logit`` and then discards its own ``a``
    argument, so replacing ``outputs['a']`` afterwards is equivalent to
    replacing the ``a = action_activation(a_logit)`` line and cannot affect
    ``z``, ``a_logit``, ``z_suc_aae``, ``p_applicable`` or ``all_precons``.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    @property
    def domain_model(self) -> Any:
        """The wrapped net's ``Domain_Model``, which the MILP side reads."""
        return self.inner.domain_model

    @property
    def action_list(self) -> List[int]:
        """The wrapped net's action index list."""
        return self.inner.action_list

    def forward(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        i: torch.Tensor,
        g: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Upstream's outputs with ``a`` replaced by the observed action trace.

        Raises:
            ValueError: if ``a`` is not ``[B, T+1, action_dim]``, which means the
                action trace does not span the frames it was batched with.
        """
        outputs = self.inner(x, a, i, g)
        if a.shape != outputs["a_logit"].shape:
            raise ValueError(
                f"observed actions are {tuple(a.shape)} but the action head "
                f"emits {tuple(outputs['a_logit'].shape)}; a trace of T frames "
                f"must carry T+1 actions"
            )
        outputs["a"] = a
        return outputs


class ObservedActionMixin:
    """Install :class:`ObservedActionNet` over the vendored net (option B).

    Dropping this mixin is exactly option A, the published unsupervised-action
    system.
    """

    def _create_net(self) -> nn.Module:
        """Upstream's net, wrapped so ``a`` is the observed action."""
        return ObservedActionNet(super()._create_net())


def resolve_lengths(
    lengths: Optional[torch.Tensor], batch: int, t_max: int, device: torch.device
) -> torch.Tensor:
    """``lengths`` validated, or every trace at ``t_max`` when it is ``None``.

    Raises:
        ValueError: if the shape is not ``[batch]``, or if a length is outside
            ``[1, t_max]`` — a trace with no interior frame is rejected by
            :func:`~src.milp.trace_tensors.interior_frame_count` upstream of
            here, and one longer than the batch has no rows to occupy.
    """
    if lengths is None:
        return torch.full((batch,), t_max, dtype=torch.int64, device=device)

    if lengths.ndim != 1 or lengths.shape[0] != batch:
        raise ValueError(
            f"lengths must be [B] with B={batch}, got {tuple(lengths.shape)}"
        )
    lengths = lengths.to(device=device, dtype=torch.int64)
    if bool(((lengths < 1) | (lengths > t_max)).any()):
        raise ValueError(
            f"every length must lie in [1, {t_max}], got {lengths.tolist()}"
        )
    return lengths


def live_action_steps(lengths: torch.Tensor, t_max: int) -> torch.Tensor:
    """``[B, t_max + 1]`` float mask: step ``t`` is live iff ``t <= L``.

    A trace of ``L`` interior frames spans ``L + 1`` actions, so this is the
    mask over the rows of ``a`` and ``a_logit``.
    """
    steps = torch.arange(t_max + 1, device=lengths.device)
    return (steps.unsqueeze(0) <= lengths.unsqueeze(1)).float()


def live_prefix_steps(lengths: torch.Tensor, t_max: int) -> torch.Tensor:
    """``[B, t_max]`` float mask: transition ``t`` is live iff ``t < L``.

    ``loss_pred``'s prefix compares each transition against the *next* observed
    frame, so a trace of ``L`` frames has ``L`` such transitions — the ``L``-th
    action lands on the goal and is the endpoint term instead.
    """
    steps = torch.arange(t_max, device=lengths.device)
    return (steps.unsqueeze(0) < lengths.unsqueeze(1)).float()


def total_action_steps(lengths: torch.Tensor) -> int:
    """``sum(L_i + 1)``, the batch's true step count."""
    return int(lengths.sum().item()) + int(lengths.shape[0])


class LengthMaskedLossMixin:
    """``ROSAMEGoal``'s loss, read against a per-trace length (§6.1a)."""

    def loss_prior(
        self, outputs: Mapping[str, torch.Tensor], lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Upstream's term, normalised by true steps rather than ``B*(T+1)``.

        The numerator needs no mask: a padded row of ``a`` is all-zero, so it
        contributes exactly zero to the contraction.
        """
        batch, t_max, _ = outputs["z"].shape
        lengths = resolve_lengths(lengths, batch, t_max, outputs["z"].device)

        all_precons = outputs["all_precons"]
        a = outputs["a"].unsqueeze(2)
        weighted = a @ F.mse_loss(
            all_precons, torch.ones_like(all_precons), reduction="none"
        )
        return weighted.sum() / total_action_steps(lengths)

    def loss_pred(
        self,
        outputs: Mapping[str, torch.Tensor],
        goals: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Upstream's term with the ``gamma`` goal anchor read at ``a[:, L]``.

        Returns:
            ``[B]``, unnormalised, as upstream returns it.
        """
        batch, t_max, _ = outputs["z"].shape
        lengths = resolve_lengths(lengths, batch, t_max, outputs["z"].device)

        z_suc_aae = outputs["z_suc_aae"]  # [B, T+1, action_dim, symbol_dim]
        a = outputs["a"].unsqueeze(2)  # [B, T+1, 1, action_dim]

        target_prefix = outputs["z"].unsqueeze(2).expand_as(z_suc_aae[:, :-1, ...])
        mse_prefix = F.mse_loss(z_suc_aae[:, :-1, ...], target_prefix, reduction="none")
        keep = live_prefix_steps(lengths, t_max)[:, :, None, None]
        loss_prefix = ((a[:, :-1, :] * keep) @ mse_prefix).sum(dim=(1, 2, 3))

        rows = torch.arange(batch, device=lengths.device)
        z_suc_last = z_suc_aae[rows, lengths]  # [B, action_dim, symbol_dim]
        target_last = goals.unsqueeze(1).expand_as(z_suc_last)
        mse_last = F.mse_loss(z_suc_last, target_last, reduction="none")
        loss_last = (a[rows, lengths] @ mse_last).sum(dim=(1, 2))

        return loss_prefix + self.parameters["gamma"] * loss_last

    def loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Sequence[Any],
        indices: Sequence[torch.Tensor] = (),
        mip_pseudo_labels: Optional[Any] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Upstream's ``loss`` with the four length-dependent terms masked.

        Args:
            outputs: What the net returned.
            targets: ``[state_traces, action_traces]``; only the former is read.
            indices: Each row's index into the dataset, as the loader yields it.
            mip_pseudo_labels: The solver's label store, or ``None`` for a
                DL-only run.
            lengths: ``[B]`` per-trace interior frame counts. ``None`` treats
                every trace as full-length, which is upstream exactly.
        """
        batch, t_max, _ = outputs["z"].shape
        lengths = resolve_lengths(lengths, batch, t_max, outputs["z"].device)
        total_steps = total_action_steps(lengths)

        loss_prior_val = self.loss_prior(outputs, lengths)
        goals = targets[0][:, -1, :]
        loss_pred_val = self.loss_pred(outputs, goals, lengths)
        loss_app_val = self.loss_app(outputs)
        loss_reconst_val = self.loss_reconst(outputs)
        loss_pred_val = loss_pred_val.sum() / total_steps
        loss_app_val = loss_app_val.sum() / total_steps

        loss_dl_relaxed = (
            self.parameters["lambda"] * loss_prior_val
            + self.parameters["beta_pred"] * loss_pred_val
            + self.parameters["beta_app"] * loss_app_val
            + self.parameters["beta_reconst"] * loss_reconst_val
        )

        preds_a = outputs["a_logit"].view(-1, self.adim())
        pseudo_labels_a = self.pseudo_label(outputs, goals)
        weight_mask_a = live_action_steps(lengths, t_max)

        trace_labels = mip_pseudo_labels.traces if mip_pseudo_labels is not None else {}
        loss_pseudo_s: Any = 0
        for batch_id, idx in enumerate(indices):
            idx = idx.item()
            if idx not in trace_labels:
                continue
            weight, state_label, action_label = trace_labels[idx]
            live = int(lengths[batch_id].item())
            if "state" in self.parameters["MIP_to_DL"]:
                state_label = state_label.to(outputs["z"].device)
                loss_pseudo_s += (
                    F.binary_cross_entropy(
                        outputs["z"][batch_id][:live], state_label[:live]
                    )
                    * weight
                )
            if "action" in self.parameters["MIP_to_DL"]:
                action_label = action_label.to(outputs["a_logit"].device)
                pseudo_labels_a[batch_id] = action_label
                weight_mask_a[batch_id] = weight_mask_a[batch_id] * weight
            trace_labels[idx] = (
                weight * self.parameters["pseudo_weight_decay"],
                state_label,
                action_label,
            )

        if len(trace_labels) > 0:
            loss_pseudo_s /= len(trace_labels)

        per_step_a = F.cross_entropy(
            preds_a, pseudo_labels_a.reshape(-1), reduction="none"
        )
        loss_pseudo_a = (weight_mask_a.view(-1) * per_step_a).sum() / total_steps

        model_label = mip_pseudo_labels.model if mip_pseudo_labels is not None else None
        loss_pseudo_m: Any = 0
        if "model" in self.parameters["MIP_to_DL"] and model_label is not None:
            len_model = 0
            for schema in self.domain_model.action_schemas:
                schema_params = self.domain_model.activation(schema())
                target = model_label[schema.name].to(schema_params.device)
                loss_pseudo_m += F.cross_entropy(schema_params, target, reduction="sum")
                len_model += schema.randn.shape[0]
            loss_pseudo_m /= len_model

        return {
            "total_loss": loss_dl_relaxed
            + loss_pseudo_a
            + loss_pseudo_s
            + loss_pseudo_m,
            "loss_pred": loss_pred_val,
            "loss_app": loss_app_val,
            "loss_reconst": loss_reconst_val,
            "loss_prior": loss_prior_val,
        }


class Rosame26Goal(ObservedActionMixin, LengthMaskedLossMixin, ROSAMEGoal):
    """``ROSAMEGoal`` with observed actions (§0) and a length-aware loss (§6.1a)."""
