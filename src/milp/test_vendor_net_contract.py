"""Gate 1 (shapes) and gate 2 (losses) for the vendored ICAPS-26 ``ROSAMEGoal``.

    python -m pytest src/milp/test_vendor_net_contract.py

These are the executable form of plan §9.1 and §9.2. They run a synthetic trace
through the vendored ``Net.forward`` and ``ROSAMEGoal.loss`` and pin every shape
and every normalisation the Phase-2 adapter will have to satisfy. No adapter is
imported: it does not exist yet, and the point of the gate is to fix the target
before the code that has to hit it is written.

Gate 1 — shapes (§9.1):
  * ``z``                          ``[B, T, S]``, and ``z in [0, 1]``
  * ``a_logit`` / ``a``            ``[B, T+1, adim]``
  * ``z_suc_aae`` / ``p_applicable`` ``[B, T+1, adim, S]``
  * ``state_traces`` (``targets[0]``) ``[B, T+2, S]``
  * the ``T = N-1`` frame alignment of §4.1, and rejection of a 2-image trace

Gate 2 — losses (§9.2):
  * ``loss_pred`` / ``loss_app`` are normalised by ``B*(T+1)``
  * ``beta_reconst: 0`` leaves ``loss_reconst`` out of ``total_loss``
  * both ``loss_pseudo_a`` regimes: full strength with no MILP labels (because
    ``weight_mask_a`` is initialised to ones), and re-weighted with them

FRAME ALIGNMENT (§4.1). Our data has ``N+1`` images for ``N`` actions. Both
endpoint frames are dropped and re-enter symbolically, as ``inits`` and
``goals``, so the network sees ``T = N-1`` interior images::

    init   = GT s_0             enters as `inits`, not as an image
    images = frames 1..N-1      T = N-1
    goal   = GT s_N             enters as `goals`, not as an image
    actions = a_1..a_N          T+1 = N
    states  = s_0..s_N          T+2 = N+1

``interior_frame_count`` below states that arithmetic. Phase 2's adapter must
call one shared implementation and this gate must be repointed at it; until then
the contract lives here so the number cannot be silently re-derived.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

torch = pytest.importorskip("torch")

import src.milp  # noqa: F401  (vendor sys.path bootstrap)
from src.milp.domain_assets import write_grounding_assets

from dl.model import ROSAMEGoal

DOMAIN = "blocksworld"
OBJECTS = {"block": ["a", "b", "c"]}

B, T, C, H, W = 2, 4, 3, 64, 64

# §6, verbatim from `train_common.py` except where §6 records a deviation.
PINNED_PARAMETERS: Dict[str, Any] = {
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
    "pseudo_weight_decay": 0.99,
    "device": "cpu",
    # Upstream verbatim; the arm itself drops "action" under option B (§6), and
    # `test_dropping_action_from_mip_to_dl_leaves_pseudo_a_self_labelled` pins
    # what that costs. `DL_to_MIP` is unaffected and is not a network parameter.
    "MIP_to_DL": ["state", "action", "model"],
}


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


# A gate is a fixed target, so nothing here may vary run to run. This seeds the
# weight init as well as the synthetic trace.
SEED = 0


@pytest.fixture(scope="module")
def built_network(tmp_path_factory) -> Tuple[ROSAMEGoal, int, int]:
    """A built ``ROSAMEGoal`` plus its ``(prop_dim, action_dim)``."""
    root = tmp_path_factory.mktemp("rosame26_assets")
    write_grounding_assets(DOMAIN, OBJECTS, root)

    parameters = dict(PINNED_PARAMETERS, domain=DOMAIN, domain_assets_root=str(root))
    torch.manual_seed(SEED)
    network = ROSAMEGoal(str(Path(root) / "run"), parameters)
    network.build([T, C, H, W])
    return network, parameters["prop_dim"], parameters["action_dim"]


@pytest.fixture(autouse=True)
def pinned_parameters_are_intact(request) -> None:
    """Fail any test that reaches a network whose parameters have been left mutated.

    ``built_network`` is module-scoped, so an in-test assignment to
    ``network.parameters`` would otherwise silently change every test after it.
    Mutate through :func:`override_parameter`, which restores.
    """
    if "built_network" not in request.fixturenames:
        return
    network, _, _ = request.getfixturevalue("built_network")
    drifted = {
        key: (network.parameters[key], value)
        for key, value in PINNED_PARAMETERS.items()
        if network.parameters[key] != value
    }
    assert not drifted, f"a previous test left parameters mutated: {drifted}"


@pytest.fixture(scope="module")
def outputs(built_network) -> Dict[str, Any]:
    network, n_props, n_actions = built_network
    torch.manual_seed(SEED)
    images = torch.rand(B, T, C, H, W)
    actions = torch.zeros(B, T + 1, n_actions)
    actions[:, :, 0] = 1.0  # option B: observed actions, one-hot
    inits = torch.zeros(B, n_props)
    goals = torch.zeros(B, n_props)
    with torch.no_grad():
        return network.net(images, actions, inits, goals)


# --------------------------------------------------------------------------
# Gate 1 — shapes (§9.1)
# --------------------------------------------------------------------------


def test_z_is_b_t_s_and_bounded(outputs, built_network) -> None:
    _, n_props, _ = built_network
    z = outputs["z"]
    assert tuple(z.shape) == (B, T, n_props)
    assert bool((z >= 0).all() and (z <= 1).all()), "z is a sigmoid; it must lie in [0, 1]"


def test_action_head_is_b_t_plus_1_adim(outputs, built_network) -> None:
    _, _, n_actions = built_network
    assert tuple(outputs["a_logit"].shape) == (B, T + 1, n_actions)
    assert tuple(outputs["a"].shape) == (B, T + 1, n_actions)


@pytest.mark.parametrize("key", ["z_suc_aae", "p_applicable"])
def test_applier_tensors_are_b_t_plus_1_adim_s(key, outputs, built_network) -> None:
    _, n_props, n_actions = built_network
    assert tuple(outputs[key].shape) == (B, T + 1, n_actions, n_props)


def test_all_precons_is_adim_s(outputs, built_network) -> None:
    _, n_props, n_actions = built_network
    assert tuple(outputs["all_precons"].shape) == (n_actions, n_props)


def test_reconstruction_matches_the_input_trace(outputs) -> None:
    assert tuple(outputs["y"].shape) == tuple(outputs["x"].shape) == (B, T, C, H, W)


def test_state_traces_target_is_b_t_plus_2_s(outputs, built_network) -> None:
    """``state_accuracy`` slices ``targets[0][:, 1:-1, :]`` against ``z``.

    That only aligns when the target carries both endpoints, i.e. ``T+2`` rows.
    """
    network, n_props, _ = built_network
    state_traces = torch.zeros(B, T + 2, n_props)
    state_traces[:, 1:-1, :] = (outputs["z"] >= 0.5).float()
    assert network.state_accuracy(outputs, [state_traces]) == pytest.approx(1.0), (
        "a T+2 target whose interior equals the thresholded z must score exactly 1"
    )

    too_short = torch.zeros(B, T + 1, n_props)
    with pytest.raises(RuntimeError):
        network.state_accuracy(outputs, [too_short])


def test_action_traces_target_is_b_t_plus_1_adim(outputs, built_network) -> None:
    network, _, n_actions = built_network
    action_traces = torch.zeros(B, T + 1, n_actions)
    predicted = outputs["a"].argmax(dim=-1)
    action_traces.scatter_(2, predicted.unsqueeze(-1), 1.0)
    assert network.action_accuracy(outputs, [None, action_traces]) == pytest.approx(1.0)

    too_short = torch.zeros(B, T, n_actions)
    with pytest.raises(RuntimeError):
        network.action_accuracy(outputs, [None, too_short])


# --- the T = N-1 alignment of §4.1 -----------------------------------------


@pytest.mark.parametrize(
    "n_images, n_actions, expected_t",
    [
        (3, 2, 1),
        (5, 4, 3),
        (6, 5, 4),
        (T + 2, T + 1, T),
    ],
)
def test_interior_frame_count_is_n_minus_one(n_images, n_actions, expected_t) -> None:
    """``T = N-1`` for ``N`` actions, and the action trace is then ``T+1`` long."""
    assert n_actions == n_images - 1
    assert interior_frame_count(n_images) == expected_t == n_actions - 1
    assert n_actions == expected_t + 1  # actions align with a_logit's T+1 rows
    assert n_images == expected_t + 2  # states align with state_traces' T+2 rows


@pytest.mark.parametrize("n_images", [0, 1, 2])
def test_a_two_image_trace_is_rejected_not_reduced_to_t_zero(n_images) -> None:
    with pytest.raises(ValueError, match="no interior frame"):
        interior_frame_count(n_images)


# --------------------------------------------------------------------------
# Gate 2 — losses (§9.2)
# --------------------------------------------------------------------------


@pytest.fixture()
def state_traces(built_network) -> "torch.Tensor":
    _, n_props, _ = built_network
    generator = torch.Generator().manual_seed(0)
    return (torch.rand(B, T + 2, n_props, generator=generator) > 0.5).float()


@pytest.fixture()
def override_parameter(built_network):
    """Set ``network.parameters`` keys for one test, then restore them.

    ``built_network`` is module-scoped, so a test that assigns to ``parameters``
    without this would change every test that runs after it.
    """
    network, _, _ = built_network
    saved: Dict[str, Any] = {}

    def override(key: str, value: Any) -> None:
        saved.setdefault(key, network.parameters[key])
        network.parameters[key] = value

    yield override
    for key, value in saved.items():
        network.parameters[key] = value


# `loss()` returns `total_loss` but not the pseudo-label terms, so the only way to
# read one is to subtract the weighted parts it does report. That recovers a ~2.9
# quantity from two ~17.9 float32 numbers, costing about an order of magnitude of
# precision over the 1.2e-7 machine epsilon. Measured worst case over eight seeds
# was 4.4e-7; the alternative hypothesis in every test below is a wholly different
# number, so 1e-5 discriminates with room to spare.
CANCELLATION_TOLERANCE = 1e-5


def _pseudo_terms(network, losses) -> float:
    """``total_loss`` minus the weighted parts ``loss()`` reports, i.e. the pseudo terms."""
    parameters = network.parameters
    relaxed = (
        parameters["lambda"] * losses["loss_prior"]
        + parameters["beta_pred"] * losses["loss_pred"]
        + parameters["beta_app"] * losses["loss_app"]
        + parameters["beta_reconst"] * losses["loss_reconst"]
    )
    return (losses["total_loss"] - relaxed).item()


def _loss(network, outputs, state_traces, mip_pseudo_labels=None, indices=None):
    indices = torch.arange(B) if indices is None else indices
    return network.loss(
        outputs,
        [state_traces, None],
        indices=indices,
        mip_pseudo_labels=mip_pseudo_labels,
    )


def test_loss_pred_and_app_are_normalised_by_b_times_t_plus_1(
    built_network, outputs, state_traces
) -> None:
    network, _, _ = built_network
    losses = _loss(network, outputs, state_traces)

    goals = state_traces[:, -1, :]
    assert losses["loss_pred"].item() == pytest.approx(
        network.loss_pred(outputs, goals).sum().item() / (B * (T + 1)), rel=1e-6
    )
    assert losses["loss_app"].item() == pytest.approx(
        network.loss_app(outputs).sum().item() / (B * (T + 1)), rel=1e-6
    )


def test_goals_are_the_last_row_of_the_t_plus_2_state_trace(
    built_network, outputs, state_traces
) -> None:
    """``loss()`` reads ``targets[0][:, -1, :]``, so the final GT state is the anchor."""
    network, _, _ = built_network
    baseline = _loss(network, outputs, state_traces)["loss_pred"].item()

    perturbed = state_traces.clone()
    perturbed[:, -1, :] = 1.0 - perturbed[:, -1, :]
    assert _loss(network, outputs, perturbed)["loss_pred"].item() != pytest.approx(
        baseline, rel=1e-6
    )

    interior_only = state_traces.clone()
    interior_only[:, 1:-1, :] = 1.0 - interior_only[:, 1:-1, :]
    assert _loss(network, outputs, interior_only)["loss_pred"].item() == pytest.approx(
        baseline, rel=1e-9
    ), "only the last row feeds loss_pred; the interior rows are metrics-only"


def test_total_loss_is_the_weighted_sum_of_its_reported_parts(
    built_network, outputs, state_traces
) -> None:
    network, _, _ = built_network
    losses = _loss(network, outputs, state_traces)
    parameters = network.parameters

    weighted = (
        parameters["lambda"] * losses["loss_prior"]
        + parameters["beta_pred"] * losses["loss_pred"]
        + parameters["beta_app"] * losses["loss_app"]
        + parameters["beta_reconst"] * losses["loss_reconst"]
    )
    # The remainder is loss_pseudo_a; with no MILP labels it is the unweighted mean.
    goals = state_traces[:, -1, :]
    expected_pseudo_a = torch.nn.functional.cross_entropy(
        outputs["a_logit"].view(-1, network.adim()),
        network.pseudo_label(outputs, goals).reshape(-1),
        reduction="none",
    ).mean()
    assert losses["total_loss"].item() == pytest.approx(
        (weighted + expected_pseudo_a).item(), rel=1e-6
    )


def test_beta_reconst_zero_keeps_reconstruction_out_of_the_total(
    built_network, outputs, state_traces, override_parameter
) -> None:
    """§6: reconstruction is reported but contributes nothing at ``beta_reconst: 0``.

    Pinned by difference, not by inspection: flipping the weight to 1 must raise
    ``total_loss`` by exactly the ``loss_reconst`` that was already being reported.
    """
    network, _, _ = built_network
    assert network.parameters["beta_reconst"] == 0

    at_zero = _loss(network, outputs, state_traces)
    reported = at_zero["loss_reconst"].item()
    assert reported > 0, "the decoder runs even when its weight is zero"

    override_parameter("beta_reconst", 1)
    at_one = _loss(network, outputs, state_traces)

    assert at_one["loss_reconst"].item() == pytest.approx(reported, rel=1e-9)
    # A ~0.85 quantity recovered by differencing two ~17.7 float32 totals, so the
    # cancellation costs about two decimal digits. The alternative hypothesis is a
    # difference of zero, which 1e-4 separates comfortably.
    assert at_one["total_loss"].item() - at_zero["total_loss"].item() == pytest.approx(
        reported, rel=1e-4
    )


def test_loss_pseudo_a_runs_at_full_strength_without_any_milp_labels(
    built_network, outputs, state_traces
) -> None:
    """``weight_mask_a`` is initialised to ones, so the term is live from epoch 0.

    This is the asymmetry of §4.2a: ``loss_pseudo_s`` is exactly zero without
    MILP labels, but ``loss_pseudo_a`` is not — it self-labels from
    ``pseudo_label()``. A DL-only run is therefore *not* free of the pseudo-label
    machinery.
    """
    network, _, _ = built_network
    goals = state_traces[:, -1, :]
    expected = torch.nn.functional.cross_entropy(
        outputs["a_logit"].view(-1, network.adim()),
        network.pseudo_label(outputs, goals).reshape(-1),
        reduction="none",
    ).mean()
    assert expected.item() > 0

    losses = _loss(network, outputs, state_traces)
    assert _pseudo_terms(network, losses) == pytest.approx(
        expected.item(), rel=CANCELLATION_TOLERANCE
    )


def test_milp_labels_reweight_only_their_own_rows_and_then_decay(
    built_network, outputs, state_traces, override_parameter
) -> None:
    """The second ``loss_pseudo_a`` regime: labelled rows are overridden and weighted."""
    from convertor.pseudo_label import PseudoLabels

    network, n_props, n_actions = built_network
    override_parameter("MIP_to_DL", ["state", "action"])

    labelled_row, weight = 0, 0.5
    action_label = torch.zeros(T + 1, dtype=torch.long)
    state_label = torch.zeros(T, n_props)
    labels = PseudoLabels(traces={labelled_row: (weight, state_label, action_label)})

    goals = state_traces[:, -1, :]
    self_labels = network.pseudo_label(outputs, goals)
    expected_labels = self_labels.clone()
    expected_labels[labelled_row] = action_label
    expected_mask = torch.ones(B, T + 1)
    expected_mask[labelled_row] = weight

    per_element = torch.nn.functional.cross_entropy(
        outputs["a_logit"].view(-1, n_actions),
        expected_labels.reshape(-1),
        reduction="none",
    )
    expected_pseudo_a = (expected_mask.view(-1) * per_element).mean()
    expected_pseudo_s = (
        torch.nn.functional.binary_cross_entropy(outputs["z"][labelled_row], state_label)
        * weight
    )

    losses = _loss(network, outputs, state_traces, mip_pseudo_labels=labels)
    assert _pseudo_terms(network, losses) == pytest.approx(
        (expected_pseudo_a + expected_pseudo_s).item(), rel=CANCELLATION_TOLERANCE
    )

    decayed, _, _ = labels.traces[labelled_row]
    assert decayed == pytest.approx(weight * network.parameters["pseudo_weight_decay"])


def test_dropping_action_from_mip_to_dl_leaves_pseudo_a_self_labelled(
    built_network, outputs, state_traces, override_parameter
) -> None:
    """Our §6 deviation, and what it costs.

    Option B observes the actions, so the arm drops ``action`` from ``MIP_to_DL``
    while ``DL_to_MIP`` keeps it. The consequence is not that ``loss_pseudo_a``
    switches off — ``weight_mask_a`` is still ones and ``pseudo_labels_a`` is
    still ``pseudo_label()`` — it is that the solver's action labels never reach
    the term at all. The DL half keeps grading itself against its own argmax.
    """
    from convertor.pseudo_label import PseudoLabels

    network, n_props, _ = built_network
    labelled_row, weight = 0, 0.5
    action_label = torch.zeros(T + 1, dtype=torch.long)
    state_label = torch.zeros(T, n_props)

    def under(channels):
        override_parameter("MIP_to_DL", channels)
        # `loss()` decays the weight in place, so each regime needs its own labels.
        labels = PseudoLabels(traces={labelled_row: (weight, state_label, action_label)})
        return _pseudo_terms(
            network, _loss(network, outputs, state_traces, mip_pseudo_labels=labels)
        )

    goals = state_traces[:, -1, :]
    self_labelled = torch.nn.functional.cross_entropy(
        outputs["a_logit"].view(-1, network.adim()),
        network.pseudo_label(outputs, goals).reshape(-1),
        reduction="none",
    ).mean()
    pseudo_s = (
        torch.nn.functional.binary_cross_entropy(outputs["z"][labelled_row], state_label)
        * weight
    )

    without_action = under(["state", "model"])
    assert without_action == pytest.approx(
        (self_labelled + pseudo_s).item(), rel=CANCELLATION_TOLERANCE
    ), "with `action` dropped, loss_pseudo_a is exactly the no-labels self-labelled term"

    with_action = under(["state", "action", "model"])
    assert abs(with_action - without_action) > 0.01 * abs(without_action), (
        "the deviation must be observable well clear of float noise, or this test "
        f"proves nothing; got {without_action} vs {with_action}"
    )


def test_milp_state_label_must_be_t_rows_not_t_plus_2(
    built_network, outputs, state_traces, override_parameter
) -> None:
    """``loss_pseudo_s`` compares against ``z``, so the label is interior-only (§6.1)."""
    from convertor.pseudo_label import PseudoLabels

    network, n_props, _ = built_network
    override_parameter("MIP_to_DL", ["state", "action"])

    ragged = PseudoLabels(
        traces={0: (1.0, torch.zeros(T + 2, n_props), torch.zeros(T + 1, dtype=torch.long))}
    )
    with pytest.raises((RuntimeError, ValueError)):
        _loss(network, outputs, state_traces, mip_pseudo_labels=ragged)
