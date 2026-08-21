"""Ragged image traces as the vendored ICAPS-26 network's input tensors.

``dl.network.ROSAMEGoal`` is fed a ``TraceDataset`` of three parallel tensors and
slices its symbolic anchors straight out of the second one::

    inits = state_traces[:, 0, :]
    goals = state_traces[:, -1, :]
    outputs = self.net(img_traces, action_traces, inits, goals)

FRAME ALIGNMENT (§4.1). Our traces carry ``N+1`` images for ``N`` actions. Both
endpoint frames are dropped and re-enter symbolically, so the network sees
``T = N-1`` interior images::

    init    = GT s_0            enters as `inits`, not as an image
    images  = frames 1..N-1     T = N-1
    goal    = GT s_N            enters as `goals`, not as an image
    actions = a_1..a_N          T+1 = N
    states  = s_0..s_N          T+2 = N+1

:func:`interior_frame_count` is the one implementation of that arithmetic; a
2-image trace has ``T = 0`` and is rejected rather than padded into existence.

PADDING (§6.1a). Upstream's corpus is length-homogeneous and ours is not, so a
batch is right-padded to ``T_max`` and :class:`PaddedTraces` carries a per-trace
``lengths`` array. Padded action rows are all-zero, which is what makes
``loss_pred``, ``loss_app`` and ``loss_prior`` ignore them: each contracts ``a``
against its per-step term, so a zero row contributes exactly zero. That holds
only under option B, where actions are observed rather than softmaxed. The three
terms it does not cover — the two ``gamma`` endpoint anchors, ``weight_mask_a``
and ``loss_pseudo_s`` — are the length-aware loss subclass's job; this module
owes it ``lengths``.

Interior ``state_traces`` rows are zero filler, never our VLM states (§4.3), so
the network's ``state_acc`` metric is measured against filler and is not an
accuracy. The GT goal is held from each trace's true final row to the end of the
batch, so ``state_traces[:, -1, :]`` is that trace's goal whatever its length.

Propositions and actions are named in **PDDL argument order** here, as
:mod:`benchmark.baselines.image_fold_inputs` emits them; the index maps that
turn those names into head columns come from
:func:`~src.milp.head_alignment.proposition_index_by_name` and
:func:`~src.milp.head_alignment.action_index_by_name`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence

import torch


def interior_frame_count(n_images: int) -> int:
    """Images the network sees, given a trace of ``n_images`` frames (§4.1).

    Both endpoints are dropped, so ``T = n_images - 2``. Raises when that leaves
    nothing: a 2-image trace has no interior frame and must be rejected rather
    than handed to the network as ``T = 0``.
    """
    interior = n_images - 2
    if interior < 1:
        raise ValueError(
            f"a {n_images}-image trace has no interior frame (T={interior}); "
            "both endpoints enter symbolically as inits/goals, so at least 3 "
            "images are needed"
        )
    return interior


def interior_frame_indices(n_images: int) -> List[int]:
    """Positions of the frames the network sees, into the full frame list."""
    return list(range(1, interior_frame_count(n_images) + 1))


@dataclass(frozen=True)
class TraceInput:
    """One trace, its endpoint frames already dropped.

    Attributes:
        images: ``[T, C, H, W]``, the interior frames, resized and normalised.
        init_predicates: GT ``s_0`` positive fluents, in PDDL argument order.
        goal_predicates: GT ``s_N`` positive fluents, in PDDL argument order.
        action_strings: ``a_1..a_N``, so ``T + 1`` of them, in PDDL order.
    """

    images: torch.Tensor
    init_predicates: Sequence[str]
    goal_predicates: Sequence[str]
    action_strings: Sequence[str]

    @property
    def length(self) -> int:
        """This trace's ``T``, the number of interior frames."""
        return int(self.images.shape[0])


@dataclass(frozen=True)
class PaddedTraces:
    """A right-padded batch, in the shapes ``ROSAMEGoal`` expects.

    Attributes:
        images: ``[B, T_max, C, H, W]``.
        state_traces: ``[B, T_max + 2, S]``.
        action_traces: ``[B, T_max + 1, A]``.
        lengths: ``[B]`` of ``int64``; ``lengths[i]`` is trace ``i``'s ``T``.
    """

    images: torch.Tensor
    state_traces: torch.Tensor
    action_traces: torch.Tensor
    lengths: torch.Tensor


def multi_hot(
    names: Sequence[str], index: Mapping[str, int], width: int
) -> torch.Tensor:
    """A ``[width]`` indicator over ``names``; everything unnamed reads false."""
    vector = torch.zeros(width, dtype=torch.float32)
    for name in names:
        vector[_column(name, index, "proposition")] = 1.0
    return vector


def one_hot_rows(
    names: Sequence[str], index: Mapping[str, int], width: int
) -> torch.Tensor:
    """A ``[len(names), width]`` stack of one-hot rows, one per name."""
    rows = torch.zeros(len(names), width, dtype=torch.float32)
    for step, name in enumerate(names):
        rows[step, _column(name, index, "action")] = 1.0
    return rows


def build_state_trace(
    trace: TraceInput, proposition_index: Mapping[str, int], t_max: int
) -> torch.Tensor:
    """``[t_max + 2, S]``: GT init, zero filler, then the GT goal held to the end."""
    width = len(proposition_index)
    rows = torch.zeros(t_max + 2, width, dtype=torch.float32)
    rows[0] = multi_hot(trace.init_predicates, proposition_index, width)
    rows[trace.length + 1 :] = multi_hot(
        trace.goal_predicates, proposition_index, width
    )
    return rows


def build_action_trace(
    trace: TraceInput, action_index: Mapping[str, int], t_max: int
) -> torch.Tensor:
    """``[t_max + 1, A]``: one-hot ``a_1..a_N``, then all-zero padded rows."""
    width = len(action_index)
    rows = torch.zeros(t_max + 1, width, dtype=torch.float32)
    rows[: trace.length + 1] = one_hot_rows(
        trace.action_strings, action_index, width
    )
    return rows


def build_padded_traces(
    traces: Sequence[TraceInput],
    *,
    proposition_index: Mapping[str, int],
    action_index: Mapping[str, int],
) -> PaddedTraces:
    """Right-pad ``traces`` to a common ``T`` and assemble the three tensors."""
    if not traces:
        raise ValueError("no traces to batch")
    for trace in traces:
        _check_trace(trace)

    frame_shape = tuple(traces[0].images.shape[1:])
    mismatched = [tuple(t.images.shape[1:]) for t in traces if tuple(t.images.shape[1:]) != frame_shape]
    if mismatched:
        raise ValueError(
            f"every trace must carry the same frame shape; got {frame_shape} and "
            f"{mismatched[0]}"
        )

    t_max = max(trace.length for trace in traces)
    images = torch.zeros(len(traces), t_max, *frame_shape, dtype=torch.float32)
    for position, trace in enumerate(traces):
        images[position, : trace.length] = trace.images

    return PaddedTraces(
        images=images,
        state_traces=torch.stack(
            [build_state_trace(t, proposition_index, t_max) for t in traces]
        ),
        action_traces=torch.stack(
            [build_action_trace(t, action_index, t_max) for t in traces]
        ),
        lengths=torch.tensor([t.length for t in traces], dtype=torch.int64),
    )


def _check_trace(trace: TraceInput) -> None:
    """Reject a trace the network's frame arithmetic cannot accept."""
    if trace.images.ndim != 4:
        raise ValueError(
            f"images must be [T, C, H, W], got shape {tuple(trace.images.shape)}"
        )
    interior_frame_count(trace.length + 2)
    if len(trace.action_strings) != trace.length + 1:
        raise ValueError(
            f"a trace of {trace.length} interior frames spans "
            f"{trace.length + 1} actions, got {len(trace.action_strings)}: "
            f"{list(trace.action_strings)}"
        )


def _column(name: str, index: Mapping[str, int], kind: str) -> int:
    """``index[name]``, raising with the name spelt out rather than defaulting."""
    if name not in index:
        raise KeyError(
            f"{kind} {name!r} is outside the grounding, which carries "
            f"{len(index)} {kind}s; it must be built over the object union of "
            "the whole data_dir"
        )
    return index[name]
