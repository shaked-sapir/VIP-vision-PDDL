"""Tests for :mod:`src.milp.trace_tensors`.

    python -m pytest src/milp/test_trace_tensors.py

Three things are pinned here. The frame arithmetic of §4.1, including the loud
rejection of a 2-image trace. The padding contract of §6.1a: shapes, the
``lengths`` array, all-zero padded action rows, and the GT goal staying readable
at ``state_traces[:, -1, :]`` for a trace shorter than the batch. And that the
tensors are actually accepted by the vendored ``Net.forward`` — a shape contract
nobody can satisfy by agreeing with itself.

``TestAgainstTheRealHead`` closes the loop with :mod:`src.milp.head_alignment`:
npuzzle's ``at(tile, position)`` grounds as ``at <position> <tile>`` in the head,
so a multi-hot built by predicate name alone would set the wrong column.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pytest

torch = pytest.importorskip("torch")

import src.milp  # noqa: F401  (vendor sys.path bootstrap)
from src.milp.domain_assets import build_domain, write_grounding_assets
from src.milp.head_alignment import action_index_by_name, proposition_index_by_name
from src.milp.trace_tensors import (
    PaddedTraces,
    TraceInput,
    build_action_trace,
    build_padded_traces,
    build_state_trace,
    interior_frame_count,
    interior_frame_indices,
    multi_hot,
    one_hot_rows,
)

from dl.util.ROSAME.rosame import get_domain_model
from planning_structs.instance import Instance as PSInstance

C, H, W = 3, 8, 8

PROPOSITIONS: Dict[str, int] = {"p": 0, "q": 1, "r": 2}
ACTIONS: Dict[str, int] = {"move a": 0, "move b": 1}


def _trace(length: int, actions: List[str] | None = None, **overrides) -> TraceInput:
    """A synthetic trace of ``length`` interior frames and ``length + 1`` actions."""
    fields = dict(
        images=torch.rand(length, C, H, W),
        init_predicates=["p"],
        goal_predicates=["r"],
        action_strings=actions if actions is not None else ["move a"] * (length + 1),
    )
    fields.update(overrides)
    return TraceInput(**fields)  # type: ignore[arg-type]


# ── the frame arithmetic (§4.1) ─────────────────────────────────────────


class TestInteriorFrameCount:
    @pytest.mark.parametrize("n_images,expected", [(3, 1), (4, 2), (5, 3), (10, 8)])
    def test_both_endpoints_are_dropped(self, n_images, expected) -> None:
        assert interior_frame_count(n_images) == expected

    @pytest.mark.parametrize("n_images", [0, 1, 2])
    def test_a_trace_with_no_interior_frame_is_rejected(self, n_images) -> None:
        with pytest.raises(ValueError, match="no interior frame"):
            interior_frame_count(n_images)

    def test_the_indices_skip_both_endpoints(self) -> None:
        assert interior_frame_indices(5) == [1, 2, 3]

    @pytest.mark.parametrize("n_images", [3, 4, 9])
    def test_the_indices_agree_with_the_count(self, n_images) -> None:
        indices = interior_frame_indices(n_images)
        assert len(indices) == interior_frame_count(n_images)
        assert 0 not in indices and n_images - 1 not in indices


# ── the vectors ─────────────────────────────────────────────────────────


class TestMultiHot:
    def test_named_columns_are_set_and_the_rest_are_false(self) -> None:
        assert multi_hot(["p", "r"], PROPOSITIONS, 3).tolist() == [1.0, 0.0, 1.0]

    def test_an_empty_state_is_all_false(self) -> None:
        assert multi_hot([], PROPOSITIONS, 3).tolist() == [0.0, 0.0, 0.0]

    def test_an_unknown_name_raises_rather_than_being_dropped(self) -> None:
        with pytest.raises(KeyError, match="outside the grounding"):
            multi_hot(["nope"], PROPOSITIONS, 3)


class TestOneHotRows:
    def test_each_row_names_one_action(self) -> None:
        rows = one_hot_rows(["move b", "move a"], ACTIONS, 2)
        assert rows.tolist() == [[0.0, 1.0], [1.0, 0.0]]

    def test_an_unknown_action_raises(self) -> None:
        with pytest.raises(KeyError, match="outside the grounding"):
            one_hot_rows(["fly"], ACTIONS, 2)


# ── the per-trace tensors ───────────────────────────────────────────────


class TestBuildStateTrace:
    def test_it_is_two_rows_longer_than_the_frame_count(self) -> None:
        rows = build_state_trace(_trace(3), PROPOSITIONS, t_max=3)
        assert tuple(rows.shape) == (5, 3)

    def test_the_endpoints_are_the_gt_anchors(self) -> None:
        rows = build_state_trace(_trace(3), PROPOSITIONS, t_max=3)
        assert rows[0].tolist() == [1.0, 0.0, 0.0]
        assert rows[-1].tolist() == [0.0, 0.0, 1.0]

    def test_the_interior_is_zero_filler_not_our_states(self) -> None:
        rows = build_state_trace(_trace(3), PROPOSITIONS, t_max=3)
        assert rows[1:-1].abs().sum().item() == 0.0

    def test_a_short_trace_still_reads_its_goal_at_the_last_row(self) -> None:
        """`network.py` slices `goals = state_traces[:, -1, :]`, index fixed."""
        rows = build_state_trace(_trace(1), PROPOSITIONS, t_max=4)
        assert rows[-1].tolist() == [0.0, 0.0, 1.0]

    def test_the_goal_is_held_from_the_true_final_row_onward(self) -> None:
        rows = build_state_trace(_trace(1), PROPOSITIONS, t_max=4)
        assert rows[2:].sum(dim=0).tolist() == [0.0, 0.0, 4.0]
        assert rows[1].abs().sum().item() == 0.0


class TestBuildActionTrace:
    def test_it_is_one_row_longer_than_the_frame_count(self) -> None:
        rows = build_action_trace(_trace(3), ACTIONS, t_max=3)
        assert tuple(rows.shape) == (4, 2)

    def test_every_true_step_is_one_hot(self) -> None:
        rows = build_action_trace(_trace(2), ACTIONS, t_max=2)
        assert rows.sum(dim=1).tolist() == [1.0, 1.0, 1.0]

    def test_padded_steps_are_all_zero(self) -> None:
        """§6.1a: `loss_pred`/`loss_app`/`loss_prior` contract against `a`, so an
        all-zero row makes a padded step contribute exactly zero."""
        rows = build_action_trace(_trace(1), ACTIONS, t_max=4)
        assert rows[:2].sum().item() == 2.0
        assert rows[2:].abs().sum().item() == 0.0


# ── the batch ───────────────────────────────────────────────────────────


class TestBuildPaddedTraces:
    def test_the_shapes_are_the_ones_gate_1_pins(self) -> None:
        batch = build_padded_traces(
            [_trace(1), _trace(4), _trace(2)],
            proposition_index=PROPOSITIONS,
            action_index=ACTIONS,
        )
        b, t_max, n_props, n_actions = 3, 4, 3, 2
        assert tuple(batch.images.shape) == (b, t_max, C, H, W)
        assert tuple(batch.action_traces.shape) == (b, t_max + 1, n_actions)
        assert tuple(batch.state_traces.shape) == (b, t_max + 2, n_props)
        assert tuple(batch.lengths.shape) == (b,)

    def test_the_lengths_are_the_true_frame_counts(self) -> None:
        batch = build_padded_traces(
            [_trace(1), _trace(4), _trace(2)],
            proposition_index=PROPOSITIONS,
            action_index=ACTIONS,
        )
        assert batch.lengths.tolist() == [1, 4, 2]
        assert batch.lengths.dtype == torch.int64

    def test_every_trace_keeps_its_own_frames_and_pads_with_zeros(self) -> None:
        traces = [_trace(1), _trace(3)]
        batch = build_padded_traces(
            traces, proposition_index=PROPOSITIONS, action_index=ACTIONS
        )
        for position, trace in enumerate(traces):
            assert torch.equal(batch.images[position, : trace.length], trace.images)
            assert batch.images[position, trace.length :].abs().sum().item() == 0.0

    def test_a_length_homogeneous_batch_is_padded_not_at_all(self) -> None:
        traces = [_trace(3), _trace(3)]
        batch = build_padded_traces(
            traces, proposition_index=PROPOSITIONS, action_index=ACTIONS
        )
        assert batch.lengths.tolist() == [3, 3]
        assert batch.action_traces.sum(dim=2).tolist() == [[1.0] * 4] * 2
        assert batch.state_traces[:, 1:-1].abs().sum().item() == 0.0

    def test_the_batch_keeps_the_input_order(self) -> None:
        batch = build_padded_traces(
            [_trace(1, goal_predicates=["p"]), _trace(2, goal_predicates=["q"])],
            proposition_index=PROPOSITIONS,
            action_index=ACTIONS,
        )
        assert batch.state_traces[0, -1].tolist() == [1.0, 0.0, 0.0]
        assert batch.state_traces[1, -1].tolist() == [0.0, 1.0, 0.0]


class TestBuildPaddedTracesRejects:
    def test_an_empty_batch(self) -> None:
        with pytest.raises(ValueError, match="no traces to batch"):
            build_padded_traces(
                [], proposition_index=PROPOSITIONS, action_index=ACTIONS
            )

    def test_a_trace_with_no_interior_frame(self) -> None:
        with pytest.raises(ValueError, match="no interior frame"):
            build_padded_traces(
                [_trace(3), _trace(0, actions=["move a"])],
                proposition_index=PROPOSITIONS,
                action_index=ACTIONS,
            )

    def test_an_action_count_that_does_not_span_the_frames(self) -> None:
        with pytest.raises(ValueError, match="spans 4 actions, got 3"):
            build_padded_traces(
                [_trace(3, actions=["move a"] * 3)],
                proposition_index=PROPOSITIONS,
                action_index=ACTIONS,
            )

    def test_frames_of_two_different_shapes(self) -> None:
        odd = _trace(2, images=torch.rand(2, C, H, W + 1))
        with pytest.raises(ValueError, match="same frame shape"):
            build_padded_traces(
                [_trace(2), odd],
                proposition_index=PROPOSITIONS,
                action_index=ACTIONS,
            )

    def test_images_that_are_not_a_frame_stack(self) -> None:
        with pytest.raises(ValueError, match=r"must be \[T, C, H, W\]"):
            build_padded_traces(
                [_trace(2, images=torch.rand(2, H, W))],
                proposition_index=PROPOSITIONS,
                action_index=ACTIONS,
            )


# ── against the real grounding ──────────────────────────────────────────


def _indices(domain_key: str, objects: Dict[str, List[str]], root: Path):
    """``(proposition_index, action_index, head_propositions)`` for a domain."""
    instance = PSInstance(
        build_domain(domain_key),
        [(name, type_name) for type_name, names in objects.items() for name in names],
    )
    write_grounding_assets(domain_key, objects, root)
    model = get_domain_model(domain_key, str(root))
    return (
        proposition_index_by_name(instance, model),
        action_index_by_name(instance, model),
        model.propositions,
    )


class TestAgainstTheRealHead:
    def test_a_blocksworld_state_sets_exactly_its_own_columns(self, tmp_path) -> None:
        propositions, _, head = _indices(
            "blocksworld", {"block": ["a", "b", "c"]}, tmp_path
        )
        named = ["on a b", "clear c", "handempty"]
        vector = multi_hot(named, propositions, len(propositions))

        by_column = {column: key for key, column in head.items()}
        set_columns = (vector > 0).nonzero().flatten().tolist()
        assert sorted(by_column[c] for c in set_columns) == sorted(named)

    def test_npuzzle_at_lands_on_the_permuted_column(self, tmp_path) -> None:
        """`at(tile, position)` grounds as `at <position> <tile>` in the head."""
        objects = {"tile": ["tile1", "tile2"], "position": ["pos1", "pos2"]}
        propositions, _, head = _indices("npuzzle", objects, tmp_path)

        column = propositions["at tile1 pos1"]
        by_column = {index: key for key, index in head.items()}
        assert by_column[column] == "at pos1 tile1"

    def test_a_blocksworld_action_is_one_hot_on_its_head_index(self, tmp_path) -> None:
        _, actions, _ = _indices("blocksworld", {"block": ["a", "b", "c"]}, tmp_path)
        rows = one_hot_rows(["stack a b"], actions, len(actions))
        assert rows.sum().item() == 1.0
        assert int(rows[0].argmax()) == actions["stack a b"]


# ── through the vendored network ────────────────────────────────────────

_NET_OBJECTS = {"block": ["a", "b", "c"]}
_NET_PARAMETERS = {
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
    "MIP_to_DL": ["state", "action", "model"],
}


@pytest.fixture(scope="module")
def blocksworld_batch(tmp_path_factory) -> Tuple[PaddedTraces, object]:
    """A ragged blocksworld batch and a ``ROSAMEGoal`` built for its shape."""
    from dl.model import ROSAMEGoal

    root = tmp_path_factory.mktemp("trace_tensor_assets")
    propositions, actions, _ = _indices("blocksworld", _NET_OBJECTS, root)

    def one(length: int) -> TraceInput:
        return TraceInput(
            images=torch.rand(length, C, 64, 64),
            init_predicates=["on a b", "clear a", "handempty"],
            goal_predicates=["on b a", "clear b", "handempty"],
            action_strings=["unstack a b"] * (length + 1),
        )

    batch = build_padded_traces(
        [one(1), one(3), one(2)],
        proposition_index=propositions,
        action_index=actions,
    )

    parameters = dict(
        _NET_PARAMETERS, domain="blocksworld", domain_assets_root=str(root)
    )
    torch.manual_seed(0)
    network = ROSAMEGoal(str(Path(root) / "run"), parameters)
    network.build(list(batch.images.shape[1:]))
    return batch, network


class TestThroughTheVendoredNetwork:
    def test_the_batch_is_accepted_and_the_outputs_have_the_pinned_shapes(
        self, blocksworld_batch
    ) -> None:
        batch, network = blocksworld_batch
        b, t_max = batch.images.shape[0], batch.images.shape[1]

        outputs = network.net(
            batch.images,
            batch.action_traces,
            batch.state_traces[:, 0, :],
            batch.state_traces[:, -1, :],
        )

        n_props = network.parameters["prop_dim"]
        n_actions = network.parameters["action_dim"]
        assert tuple(outputs["z"].shape) == (b, t_max, n_props)
        assert tuple(outputs["a"].shape) == (b, t_max + 1, n_actions)
        assert tuple(outputs["z_suc_aae"].shape) == (
            b,
            t_max + 1,
            n_actions,
            n_props,
        )

    def test_the_anchors_slice_out_of_the_state_trace_at_the_vendored_indices(
        self, blocksworld_batch
    ) -> None:
        batch, network = blocksworld_batch
        n_props = network.parameters["prop_dim"]

        assert tuple(batch.state_traces[:, 0, :].shape) == (3, n_props)
        assert batch.state_traces[:, 0, :].sum(dim=1).tolist() == [3.0, 3.0, 3.0]
        assert batch.state_traces[:, -1, :].sum(dim=1).tolist() == [3.0, 3.0, 3.0]
