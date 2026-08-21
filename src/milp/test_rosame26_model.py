"""Unit tests for :mod:`src.milp.rosame26_model` — option B and the ragged loss.

    python -m pytest src/milp/test_rosame26_model.py

Two claims are under test, and they are independent.

**The observed action reaches the loss unchanged, and nothing else moves.**
:class:`ObservedActionNet` overwrites ``outputs['a']`` after the vendored forward
has run, which is only equivalent to replacing the ``a = action_activation(...)``
line if that line's output feeds nothing else. ``test_wrapping_changes_only_a``
is what makes that an asserted fact rather than a reading of the code.

**The masked loss is the length-aware loss §6.1a specifies.** The reference
values here are written from the specification — an explicit Python loop over
each trace's own steps — not by re-deriving the vectorised implementation. A
test that recomputed the implementation would agree with any bug in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

import src.milp  # noqa: F401  (vendor sys.path bootstrap)
from src.milp.domain_assets import write_grounding_assets
from src.milp.rosame26_model import (
    LengthMaskedLossMixin,
    ObservedActionNet,
    Rosame26Goal,
    live_action_steps,
    live_prefix_steps,
    resolve_lengths,
    total_action_steps,
)

from dl.model import ROSAMEGoal

DOMAIN = "blocksworld"
OBJECTS = {"block": ["a", "b", "c"]}
SEED = 0

B, T_MAX, S, A = 3, 4, 6, 5
C, H, W = 3, 64, 64

LOSS_PARAMETERS: Dict[str, Any] = {
    "lambda": 0.2,
    "gamma": 10,
    "beta_pred": 1,
    "beta_app": 1,
    "beta_reconst": 0,
    "pseudo_weight_decay": 0.99,
    "action_dim": A,
    # The arm's own value: option B drops "action" from this channel (§0, §6).
    "MIP_to_DL": ["state", "model"],
    "device": "cpu",
}

BUILD_PARAMETERS: Dict[str, Any] = {
    "aae_width": 1000,
    "aae_depth": 3,
    "feature_dim": 512,
    "hidden_dim": 256,
    "optimizer": "Adam",
    "lr": 1e-4,
}


# --------------------------------------------------------------------------- #
# synthetic outputs                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class StubPseudoLabels:
    """``convertor.pseudo_labels`` reduced to what ``loss`` reads."""

    traces: Dict[int, Tuple[float, torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    model: Any = None


def make_outputs(lengths: torch.Tensor, seed: int = SEED) -> Dict[str, torch.Tensor]:
    """A synthetic ``Net.forward`` result with padded rows of ``a`` zeroed.

    Probabilities are drawn strictly inside ``(0, 1)`` because ``pseudo_label``
    and ``loss_pseudo_s`` take logs of them.
    """
    generator = torch.Generator().manual_seed(seed)

    def unit(*shape: int) -> torch.Tensor:
        return 0.05 + 0.9 * torch.rand(*shape, generator=generator)

    actions = torch.zeros(B, T_MAX + 1, A)
    for row, length in enumerate(lengths.tolist()):
        for step in range(length + 1):
            actions[row, step, (row + step) % A] = 1.0

    return {
        "z": unit(B, T_MAX, S),
        "a": actions,
        "a_logit": torch.randn(B, T_MAX + 1, A, generator=generator),
        "z_suc_aae": unit(B, T_MAX + 1, A, S),
        "p_applicable": unit(B, T_MAX + 1, A, S),
        "all_precons": unit(A, S),
        "x": unit(B, T_MAX, C, 4, 4),
        "y": unit(B, T_MAX, C, 4, 4),
    }


def make_targets(seed: int = SEED) -> list:
    """``[state_traces, action_traces]``; only ``state_traces[:, -1]`` is read."""
    generator = torch.Generator().manual_seed(seed + 1)
    states = torch.randint(0, 2, (B, T_MAX + 2, S), generator=generator).float()
    return [states, torch.zeros(B, T_MAX + 1, A)]


@pytest.fixture()
def masked(tmp_path: Path) -> Rosame26Goal:
    """A :class:`Rosame26Goal` that is never built — only its loss is used."""
    return Rosame26Goal(str(tmp_path / "masked"), dict(LOSS_PARAMETERS))


@pytest.fixture()
def vendored(tmp_path: Path) -> ROSAMEGoal:
    """The unmodified upstream loss, for the bit-identity comparison."""
    return ROSAMEGoal(str(tmp_path / "vendored"), dict(LOSS_PARAMETERS))


# --------------------------------------------------------------------------- #
# specification references, written from §6.1a rather than from the code       #
# --------------------------------------------------------------------------- #


def reference_loss_pred(
    outputs: Mapping[str, torch.Tensor],
    goals: torch.Tensor,
    lengths: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Each trace's own ``L`` transitions plus its own goal anchor at step ``L``."""
    per_trace = []
    for row, length in enumerate(lengths.tolist()):
        prefix = torch.zeros(())
        for step in range(length):
            mse = (outputs["z_suc_aae"][row, step] - outputs["z"][row, step]) ** 2
            prefix = prefix + (outputs["a"][row, step] @ mse).sum()
        mse_last = (outputs["z_suc_aae"][row, length] - goals[row]) ** 2
        anchor = (outputs["a"][row, length] @ mse_last).sum()
        per_trace.append(prefix + gamma * anchor)
    return torch.stack(per_trace)


def reference_pseudo_a(
    outputs: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    lengths: torch.Tensor,
    weights: Mapping[int, float],
) -> torch.Tensor:
    """Cross-entropy over live steps only, divided by the true step total."""
    total = torch.zeros(())
    for row, length in enumerate(lengths.tolist()):
        weight = weights.get(row, 1.0)
        for step in range(length + 1):
            total = total + weight * F.cross_entropy(
                outputs["a_logit"][row, step], labels[row, step]
            )
    return total / total_action_steps(lengths)


def pseudo_total(result: Mapping[str, torch.Tensor], parameters: Mapping[str, Any]):
    """``loss_pseudo_a + loss_pseudo_s + loss_pseudo_m``, by subtraction.

    ``loss`` folds the three into ``total_loss`` and reports none of them.
    """
    reported = (
        parameters["lambda"] * result["loss_prior"]
        + parameters["beta_pred"] * result["loss_pred"]
        + parameters["beta_app"] * result["loss_app"]
        + parameters["beta_reconst"] * result["loss_reconst"]
    )
    return result["total_loss"] - reported


# --------------------------------------------------------------------------- #
# option B: the observed action                                                #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def built_pair(tmp_path_factory) -> Tuple[Rosame26Goal, ROSAMEGoal, int, int]:
    """One built :class:`Rosame26Goal` and one built vendored ``ROSAMEGoal``.

    Both are seeded identically before ``build``, so their weights agree and any
    difference in ``forward`` is the wrapper's doing.
    """
    root = tmp_path_factory.mktemp("rosame26_model_assets")
    write_grounding_assets(DOMAIN, OBJECTS, root)
    parameters = dict(
        LOSS_PARAMETERS,
        **BUILD_PARAMETERS,
        domain=DOMAIN,
        domain_assets_root=str(root),
    )

    torch.manual_seed(SEED)
    ours = Rosame26Goal(str(root / "ours"), dict(parameters))
    ours.build([2, C, H, W])

    torch.manual_seed(SEED)
    theirs = ROSAMEGoal(str(root / "theirs"), dict(parameters))
    theirs.build([2, C, H, W])

    return ours, theirs, ours.parameters["prop_dim"], ours.parameters["action_dim"]


def observed_batch(n_actions: int, n_props: int, seed: int = SEED):
    """``(images, actions, inits, goals)`` for a two-frame trace."""
    torch.manual_seed(seed)
    images = torch.rand(2, 2, C, H, W)
    actions = torch.zeros(2, 3, n_actions)
    actions[0, :, 1] = 1.0
    actions[1, :, 2] = 1.0
    return images, actions, torch.zeros(2, n_props), torch.zeros(2, n_props)


def test_a_is_the_observed_action(built_pair) -> None:
    ours, _, n_props, n_actions = built_pair
    images, actions, inits, goals = observed_batch(n_actions, n_props)

    with torch.no_grad():
        outputs = ours.net(images, actions, inits, goals)

    assert torch.equal(outputs["a"], actions)


def test_the_softmax_it_replaces_is_not_the_observed_action(built_pair) -> None:
    """Without the override ``a`` is a dense softmax, so the swap is not a no-op."""
    _, theirs, n_props, n_actions = built_pair
    images, actions, inits, goals = observed_batch(n_actions, n_props)

    with torch.no_grad():
        outputs = theirs.net(images, actions, inits, goals)

    assert not torch.equal(outputs["a"], actions)
    assert torch.allclose(outputs["a"].sum(dim=-1), torch.ones(2, 3))


def test_wrapping_changes_only_a(built_pair) -> None:
    """Every other output is bit-identical to the vendored net's."""
    ours, theirs, n_props, n_actions = built_pair
    images, actions, inits, goals = observed_batch(n_actions, n_props)

    with torch.no_grad():
        mine = ours.net(images, actions, inits, goals)
        upstream = theirs.net(images, actions, inits, goals)

    assert set(mine) == set(upstream)
    differing = [
        key for key in upstream if not torch.equal(mine[key], upstream[key])
    ]
    assert differing == ["a"]


def test_action_trace_of_the_wrong_length_raises(built_pair) -> None:
    ours, _, n_props, n_actions = built_pair
    images, actions, inits, goals = observed_batch(n_actions, n_props)

    with pytest.raises(ValueError, match="must carry T\\+1 actions"):
        ours.net(images, actions[:, :-1, :], inits, goals)


def test_domain_model_and_action_list_delegate(built_pair) -> None:
    """The MILP side reaches ``net.domain_model``; the wrapper must not hide it."""
    ours, _, _, _ = built_pair
    assert isinstance(ours.net, ObservedActionNet)
    assert ours.net.domain_model is ours.net.inner.domain_model
    assert ours.net.action_list == ours.net.inner.action_list


# --------------------------------------------------------------------------- #
# length helpers                                                               #
# --------------------------------------------------------------------------- #


def test_resolve_lengths_defaults_to_full_length() -> None:
    resolved = resolve_lengths(None, B, T_MAX, torch.device("cpu"))
    assert torch.equal(resolved, torch.full((B,), T_MAX, dtype=torch.int64))


def test_resolve_lengths_rejects_a_mismatched_batch() -> None:
    with pytest.raises(ValueError, match="lengths must be"):
        resolve_lengths(torch.tensor([1, 2]), B, T_MAX, torch.device("cpu"))


@pytest.mark.parametrize("bad", [0, T_MAX + 1, -1])
def test_resolve_lengths_rejects_out_of_range(bad: int) -> None:
    lengths = torch.tensor([T_MAX, bad, T_MAX])
    with pytest.raises(ValueError, match="every length must lie in"):
        resolve_lengths(lengths, B, T_MAX, torch.device("cpu"))


def test_live_action_steps_covers_l_plus_one_steps() -> None:
    mask = live_action_steps(torch.tensor([1, 3]), 3)
    expected = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
    )
    assert torch.equal(mask, expected)


def test_live_prefix_steps_covers_l_transitions() -> None:
    mask = live_prefix_steps(torch.tensor([1, 3]), 3)
    expected = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert torch.equal(mask, expected)


def test_total_action_steps_counts_each_traces_own_steps() -> None:
    assert total_action_steps(torch.tensor([1, 3, 4])) == (1 + 1) + (3 + 1) + (4 + 1)


# --------------------------------------------------------------------------- #
# the masked loss                                                              #
# --------------------------------------------------------------------------- #


def test_full_length_batch_is_bit_identical_to_the_vendored_loss(
    masked, vendored
) -> None:
    """The no-op property that confines the deviation to the ragged case (§6.1a)."""
    lengths = torch.full((B,), T_MAX, dtype=torch.int64)
    outputs, targets = make_outputs(lengths), make_targets()

    ours = masked.loss(dict(outputs), targets, [], None, lengths)
    theirs = vendored.loss(dict(outputs), targets, [], None)

    assert set(ours) == set(theirs)
    for key in theirs:
        assert torch.equal(ours[key], theirs[key]), key


def test_lengths_of_none_is_the_vendored_loss(masked, vendored) -> None:
    """Omitting ``lengths`` must not quietly mean something else."""
    outputs, targets = make_outputs(torch.full((B,), T_MAX, dtype=torch.int64)), make_targets()

    ours = masked.loss(dict(outputs), targets, [], None)
    theirs = vendored.loss(dict(outputs), targets, [], None)

    for key in theirs:
        assert torch.equal(ours[key], theirs[key]), key


def test_loss_pred_matches_the_per_trace_specification(masked) -> None:
    """Anchor at ``a[:, L]`` and prefix over ``t < L``, against an explicit loop."""
    lengths = torch.tensor([4, 1, 2])
    outputs, targets = make_outputs(lengths), make_targets()
    goals = targets[0][:, -1, :]

    ours = masked.loss_pred(outputs, goals, lengths)
    expected = reference_loss_pred(outputs, goals, lengths, LOSS_PARAMETERS["gamma"])

    assert torch.allclose(ours, expected, atol=1e-6)


def test_the_vendored_loss_pred_loses_the_goal_anchor_on_short_traces(
    masked, vendored
) -> None:
    """The failure §6.1a names: index ``-1`` is a zero-action pad."""
    lengths = torch.tensor([T_MAX, 1, 2])
    outputs, targets = make_outputs(lengths), make_targets()
    goals = targets[0][:, -1, :]

    ours = masked.loss_pred(outputs, goals, lengths)
    theirs = vendored.loss_pred(outputs, goals)

    assert torch.equal(ours[0], theirs[0])  # the full-length trace is untouched
    assert (ours[1:] > theirs[1:]).all()


def test_loss_pred_prefix_stops_before_the_anchor_step(masked) -> None:
    """Step ``L`` is the anchor, so it is not also an interior transition."""
    lengths = torch.tensor([1, 1, 1])
    outputs, targets = make_outputs(lengths), make_targets()
    goals = targets[0][:, -1, :]

    ours = masked.loss_pred(outputs, goals, lengths)

    anchor_only = []
    for row in range(B):
        mse = (outputs["z_suc_aae"][row, 1] - goals[row]) ** 2
        anchor_only.append((outputs["a"][row, 1] @ mse).sum())
    interior = ours - LOSS_PARAMETERS["gamma"] * torch.stack(anchor_only)

    single = []
    for row in range(B):
        mse = (outputs["z_suc_aae"][row, 0] - outputs["z"][row, 0]) ** 2
        single.append((outputs["a"][row, 0] @ mse).sum())
    assert torch.allclose(interior, torch.stack(single), atol=1e-6)


def test_normalisers_divide_by_true_steps(masked, vendored) -> None:
    """``sum(L_i + 1)`` replaces ``B*(T_max + 1)`` on prior, pred and app."""
    lengths = torch.tensor([4, 1, 2])
    outputs, targets = make_outputs(lengths), make_targets()

    ours = masked.loss(dict(outputs), targets, [], None, lengths)
    theirs = vendored.loss(dict(outputs), targets, [], None)

    padded_steps = B * (T_MAX + 1)
    true_steps = total_action_steps(lengths)
    assert true_steps < padded_steps

    # `loss_prior`'s numerator is already self-masking, so only the divisor moved.
    assert torch.allclose(
        ours["loss_prior"] * true_steps,
        theirs["loss_prior"] * padded_steps,
        atol=1e-6,
    )
    # `loss_app` needs no masking at all under right-padding, so likewise.
    assert torch.allclose(
        ours["loss_app"] * true_steps, theirs["loss_app"] * padded_steps, atol=1e-6
    )


def test_loss_reconst_is_untouched(masked, vendored) -> None:
    """Documented as deliberately unmasked; pinned so a change is visible."""
    lengths = torch.tensor([4, 1, 2])
    outputs, targets = make_outputs(lengths), make_targets()

    ours = masked.loss(dict(outputs), targets, [], None, lengths)
    theirs = vendored.loss(dict(outputs), targets, [], None)

    assert torch.equal(ours["loss_reconst"], theirs["loss_reconst"])


def test_pseudo_a_ignores_padded_steps(masked) -> None:
    """``weight_mask_a`` is ones upstream, so padding trains at full weight."""
    lengths = torch.tensor([4, 1, 2])
    outputs, targets = make_outputs(lengths), make_targets()
    goals = targets[0][:, -1, :]

    result = masked.loss(dict(outputs), targets, [], None, lengths)
    labels = masked.pseudo_label(outputs, goals)
    expected = reference_pseudo_a(outputs, labels, lengths, {})

    assert torch.allclose(pseudo_total(result, LOSS_PARAMETERS), expected, atol=1e-6)


def test_padding_a_trace_does_not_change_its_pseudo_a_contribution(masked) -> None:
    """The same trace at ``T_max`` and padded from ``L`` must weigh the same."""
    lengths = torch.tensor([2, 2, 2])
    outputs, targets = make_outputs(lengths), make_targets()

    padded = masked.loss(dict(outputs), targets, [], None, lengths)
    unpadded_steps = total_action_steps(lengths)

    contribution = pseudo_total(padded, LOSS_PARAMETERS) * unpadded_steps
    labels = masked.pseudo_label(outputs, targets[0][:, -1, :])
    expected = reference_pseudo_a(outputs, labels, lengths, {}) * unpadded_steps

    assert torch.allclose(contribution, expected, atol=1e-6)


def test_pseudo_s_scores_only_live_rows(masked) -> None:
    """Filler rows of ``z`` carry no state supervision."""
    lengths = torch.tensor([4, 1, 2])
    outputs, targets = make_outputs(lengths), make_targets()
    indices = torch.arange(B)

    generator = torch.Generator().manual_seed(SEED + 7)
    state_label = torch.randint(0, 2, (B, T_MAX, S), generator=generator).float()
    action_label = torch.zeros(T_MAX + 1, dtype=torch.long)

    def labels_for(row: int) -> StubPseudoLabels:
        return StubPseudoLabels(
            traces={row: (1.0, state_label[row], action_label.clone())}
        )

    with_labels = masked.loss(
        dict(outputs), targets, indices, labels_for(1), lengths
    )
    without = masked.loss(dict(outputs), targets, indices, None, lengths)

    added = pseudo_total(with_labels, LOSS_PARAMETERS) - pseudo_total(
        without, LOSS_PARAMETERS
    )
    live = int(lengths[1].item())
    expected = F.binary_cross_entropy(outputs["z"][1][:live], state_label[1][:live])

    assert torch.allclose(added, expected, atol=1e-6)


def test_a_mip_weight_multiplies_the_live_mask_rather_than_replacing_it(
    tmp_path: Path,
) -> None:
    """Upstream assigns ``weight_mask_a[row] = weight``; that would unmask padding.

    Runs with ``"action"`` restored to ``MIP_to_DL``, which the arm itself drops
    (§0). The branch stays upstream-faithful, so it has to be masked correctly
    for option A to remain a drop-in.
    """
    parameters = dict(LOSS_PARAMETERS, MIP_to_DL=["state", "action", "model"])
    masked = Rosame26Goal(str(tmp_path / "with_action"), parameters)

    lengths = torch.tensor([4, 1, 2])
    outputs, targets = make_outputs(lengths), make_targets()
    indices = torch.arange(B)
    weight = 0.5

    generator = torch.Generator().manual_seed(SEED + 7)
    state_label = torch.randint(0, 2, (T_MAX, S), generator=generator).float()
    action_label = torch.zeros(T_MAX + 1, dtype=torch.long)
    labels = StubPseudoLabels(traces={1: (weight, state_label, action_label)})

    result = masked.loss(dict(outputs), targets, indices, labels, lengths)

    goals = targets[0][:, -1, :]
    self_labels = masked.pseudo_label(outputs, goals)
    self_labels[1] = action_label
    live = int(lengths[1].item())
    expected_a = reference_pseudo_a(outputs, self_labels, lengths, {1: weight})
    expected_s = (
        F.binary_cross_entropy(outputs["z"][1][:live], state_label[:live]) * weight
    )

    assert torch.allclose(
        pseudo_total(result, parameters), expected_a + expected_s, atol=1e-6
    )


def test_dropping_action_from_mip_to_dl_leaves_pseudo_a_self_labelled(masked) -> None:
    """The arm's own ``MIP_to_DL`` never lets a solver label reach the action head."""
    lengths = torch.tensor([4, 1, 2])
    outputs, targets = make_outputs(lengths), make_targets()
    indices = torch.arange(B)

    state_label = torch.zeros(T_MAX, S)
    action_label = torch.full((T_MAX + 1,), A - 1, dtype=torch.long)
    labels = StubPseudoLabels(traces={1: (0.5, state_label, action_label)})

    result = masked.loss(dict(outputs), targets, indices, labels, lengths)

    goals = targets[0][:, -1, :]
    self_labels = masked.pseudo_label(outputs, goals)
    live = int(lengths[1].item())
    expected = reference_pseudo_a(
        outputs, self_labels, lengths, {}
    ) + 0.5 * F.binary_cross_entropy(outputs["z"][1][:live], state_label[:live])

    assert torch.allclose(pseudo_total(result, LOSS_PARAMETERS), expected, atol=1e-6)


def test_a_mip_label_decays_for_the_next_call(masked) -> None:
    """ψ decay lives in the loss, once per call (§3)."""
    lengths = torch.tensor([4, 1, 2])
    outputs, targets = make_outputs(lengths), make_targets()
    indices = torch.arange(B)

    state_label = torch.zeros(T_MAX, S)
    labels = StubPseudoLabels(
        traces={1: (1.0, state_label, torch.zeros(T_MAX + 1, dtype=torch.long))}
    )

    masked.loss(dict(outputs), targets, indices, labels, lengths)

    assert labels.traces[1][0] == pytest.approx(
        LOSS_PARAMETERS["pseudo_weight_decay"]
    )


def test_the_mixin_order_puts_masking_ahead_of_the_vendored_loss() -> None:
    """``Rosame26Goal`` must resolve ``loss`` to the mixin, not to ``ROSAMEGoal``."""
    order = Rosame26Goal.__mro__
    assert order.index(LengthMaskedLossMixin) < order.index(ROSAMEGoal)
    assert Rosame26Goal.loss is LengthMaskedLossMixin.loss
