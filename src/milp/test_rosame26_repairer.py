"""Tests for :mod:`src.milp.rosame26_repairer`, the MILP half of the 26 loop.

    python -m pytest src/milp/test_rosame26_repairer.py

The gate is :class:`TestAgainstTheRealEncoder`: a real ragged fold, through the
real CP-SAT encoder, producing labels whose shapes match the network's ``z``
row-for-row. That is what settles §6.1's ragged-label worry — the vendored
``extract_sol_label`` sizes from one shared ``max_t``, but this module reads
``repaired_states`` per trace, as the ICAPS-24 arm does, so the shapes come out
right without a second padding layer.

:class:`TestModelLabelsGoThroughTheAlignment` is the non-vacuity check for the
third permutation end (plan §0.1a): a label placed by row index instead of by
key must differ on the four domains that reorder, and agree on blocksworld.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pytest
import torch

import src.milp  # noqa: F401  (vendor sys.path bootstrap)
from src.milp.domain_assets import BENCH_DOMAINS, build_domain, write_grounding_assets
from src.milp.rosame26_repairer import (
    EPS,
    OBJECTIVES,
    Rosame26MipRepairer,
    TraceContext,
    model_labels_in_head_order,
    state_labels_for,
    state_probabilities,
)
from src.milp.schema_row_alignment import schema_row_keys

from dl.util.ROSAME.rosame import get_domain_model
from planning_structs.traces import ObservationM

#: Domains whose schema rows the head reorders against the CP domain.
REORDERED = {"depot", "gripper", "hanoi", "npuzzle"}

_OBJECTS = {
    "blocksworld": {"block": ["a", "b"]},
    "hanoi": {"disc": ["d1", "d2"], "peg": ["p1"]},
    "depot": {
        "truck": ["t1"], "depot": ["d1"], "crane": ["c1"],
        "package": ["p1"], "pile": ["pl1"],
    },
    "gripper": {"ball": ["b1"], "room": ["r1"], "gripper": ["g1"]},
    "npuzzle": {"tile": ["t1"], "position": ["p1", "p2"]},
}


def _head(domain_key: str, root: Path):
    write_grounding_assets(domain_key, _OBJECTS[domain_key], root)
    return get_domain_model(domain_key, str(root))


def _solution_for(domain_key: str, ps_domain, chooser) -> ObservationM:
    """An ``ObservationM`` whose truth on each binding is ``chooser(key)``."""
    pre: Dict = {}
    add: Dict = {}
    dele: Dict = {}
    for schema in ps_domain.action_schemas:
        for predicate in ps_domain.predicates:
            for binding in ps_domain.predicate_arguments.get((schema, predicate), []):
                key = (schema, predicate, tuple(binding))
                kind = chooser(key)
                pre[key] = 1 if kind == "pre" else 0
                add[key] = 1 if kind == "add" else 0
                dele[key] = 1 if kind == "del" else 0
    return ObservationM(pre, add, dele)


# ── the pieces ──────────────────────────────────────────────────────────


class TestStateProbabilities:
    def test_it_pads_both_endpoints(self) -> None:
        """``cv_predictions_to_trace`` indexes by image, and reads only 1..T."""
        rows = state_probabilities(torch.rand(5, 4), live=3)
        assert len(rows) == 5  # T + 2
        assert all(len(row) == 4 for row in rows)

    def test_the_interior_rows_are_the_live_prefix(self) -> None:
        z_row = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.9, 0.9]])
        rows = state_probabilities(z_row, live=2)
        assert rows[1] == pytest.approx([0.1, 0.2])
        assert rows[2] == pytest.approx([0.3, 0.4])

    def test_padded_rows_past_live_are_dropped(self) -> None:
        """A padded ``z`` row must never reach the solver."""
        z_row = torch.tensor([[0.1, 0.1], [0.0, 0.0], [0.0, 0.0]])
        assert len(state_probabilities(z_row, live=1)) == 3

    def test_the_placeholders_are_the_zero_weight_value(self) -> None:
        rows = state_probabilities(torch.rand(3, 2), live=2)
        assert rows[0] == [0.5, 0.5]
        assert rows[-1] == [0.5, 0.5]


class TestStateLabels:
    def test_endpoints_are_dropped(self) -> None:
        """``repaired_states`` is per image; the endpoints are hard-fixed."""
        label = state_labels_for(_FakeInstance(), [set()] * 5, {"p a": 0})
        assert label.shape == (3, 1)

    def test_a_true_proposition_sets_its_column(self) -> None:
        instance = _FakeInstance({("p", ("a",)): "P"})
        label = state_labels_for(instance, [set(), {"P"}, set()], {"p a": 0})
        assert label.tolist() == [[1.0]]

    def test_a_phantom_column_reads_false_not_absent(self) -> None:
        """Supervised 0, and correct: the object is genuinely not in this problem."""
        instance = _FakeInstance({("p", ("a",)): "P"})
        label = state_labels_for(
            instance, [set(), {"P"}, set()], {"p a": 0, "p zzz": 1}
        )
        assert label.tolist() == [[1.0, 0.0]]


class _FakeInstance:
    """Minimal stand-in: ``proposition_of`` resolves through its mapping."""

    def __init__(self, mapping=None) -> None:
        self._mapping = mapping or {}

    @property
    def propositions(self):
        return list(self._mapping.values())


@pytest.fixture(autouse=True)
def _stub_proposition_of(monkeypatch, request):
    """Route ``proposition_of`` at :class:`_FakeInstance` only."""
    from src.milp import rosame26_repairer

    original = rosame26_repairer.proposition_of

    def routed(instance, name, args):
        if isinstance(instance, _FakeInstance):
            return instance._mapping.get((name, tuple(args)))
        return original(instance, name, args)

    monkeypatch.setattr(rosame26_repairer, "proposition_of", routed)


# ── the model channel, through the alignment ────────────────────────────


@pytest.mark.parametrize("domain_key", BENCH_DOMAINS)
class TestModelLabelsGoThroughTheAlignment:
    def test_labels_are_one_hot_per_row(self, domain_key, tmp_path) -> None:
        model = _head(domain_key, tmp_path)
        ps_domain = build_domain(domain_key)
        labels = model_labels_in_head_order(
            _solution_for(domain_key, ps_domain, lambda key: "none"),
            model, ps_domain,
        )
        for schema in model.action_schemas:
            label = labels[schema.name]
            assert label.shape == (int(schema.randn.shape[0]), 4)
            assert torch.all(label.sum(dim=1) == 1.0)

    def test_precedence_is_add_then_delete_then_precondition(
        self, domain_key, tmp_path
    ) -> None:
        model = _head(domain_key, tmp_path)
        ps_domain = build_domain(domain_key)
        both = ObservationM(
            *[
                {
                    (schema, predicate, tuple(binding)): 1
                    for schema in ps_domain.action_schemas
                    for predicate in ps_domain.predicates
                    for binding in ps_domain.predicate_arguments.get(
                        (schema, predicate), []
                    )
                }
                for _ in range(3)
            ]
        )
        labels = model_labels_in_head_order(both, model, ps_domain)
        for schema in model.action_schemas:
            assert torch.all(labels[schema.name][:, 1] == 1.0), schema.name

    def test_placing_by_index_differs_wherever_the_rows_reorder(
        self, domain_key, tmp_path
    ) -> None:
        """The non-vacuity check for plan §0.1a, on the model channel.

        A solution that marks one binding per schema, decoded by key versus by
        row index, must disagree on the four domains whose rows reorder and
        agree on blocksworld.
        """
        model = _head(domain_key, tmp_path)
        ps_domain = build_domain(domain_key)
        keys = schema_row_keys(model, ps_domain)

        marked = {
            (
                ps_domain.get_action_schema(name),
                ps_domain.get_predicate(rows[0][0]),
                rows[0][1],
            )
            for name, rows in keys.items()
        }
        solution = _solution_for(
            domain_key, ps_domain, lambda key: "add" if key in marked else "none"
        )
        by_key = model_labels_in_head_order(solution, model, ps_domain)

        differs = False
        for schema in model.action_schemas:
            rows = keys[schema.name]
            ps_schema = ps_domain.get_action_schema(schema.name)
            by_naive = torch.zeros(len(rows), 4)
            for index, (predicate_name, args) in enumerate(
                _naive_rows(schema, ps_schema)
            ):
                key = (ps_schema, ps_domain.get_predicate(predicate_name), args)
                by_naive[index, 1 if key in marked else 0] = 1.0
            if not torch.equal(by_key[schema.name], by_naive):
                differs = True

        assert differs == (domain_key in REORDERED), domain_key


def _naive_rows(head_schema, ps_schema):
    """Identity ``args_dl_cp``: the head's own slot numbers, taken at face value.

    This is what plan §0.1 prescribed and §0.1a corrects — the head numbers its
    parameters in sorted-type order and the CP domain in PDDL order, so passing
    one to the other unchanged addresses different arguments.
    """
    variables = {}
    counter = 1
    for one_type in head_schema.params_types:
        count = head_schema.params[one_type]
        variables[one_type] = [str(counter + i) for i in range(count)]
        counter += count
    return [
        (predicate.name, tuple(int(v) for v in proposition.split()[1:]))
        for predicate in head_schema.predicates
        for proposition in predicate.ground(variables)
    ]


class TestTheRepairerContract:
    def test_it_satisfies_the_protocol_shape(self) -> None:
        from src.milp.rosame26_training import MipRepairer

        for name in ("pseudo_labels", "clear", "update", "run_fixer"):
            assert hasattr(Rosame26MipRepairer, name), name
        assert hasattr(MipRepairer, "run_fixer")

    def test_objectives_exclude_action(self) -> None:
        """Under option B the action is observed, so nothing infers it."""
        assert "action" not in OBJECTIVES
        assert OBJECTIVES == {"state", "model"}

    def test_clear_drops_collected_traces_but_not_labels(self, tmp_path) -> None:
        """Upstream decays labels rather than discarding them each epoch."""
        repairer = Rosame26MipRepairer(
            _head("blocksworld", tmp_path), build_domain("blocksworld"), {}, {}
        )
        repairer.pseudo_labels.traces[0] = (1, torch.zeros(1, 1), torch.zeros(1))
        repairer._collected.append(object())
        repairer.clear()

        assert repairer._collected == []
        assert 0 in repairer.pseudo_labels.traces

    def test_update_is_capped_at_mip_traces(self, tmp_path) -> None:
        contexts = {
            i: TraceContext(
                instance=None, calls=[("a", []), ("a", [])],
                init_fluents=set(), goal_fluents=None,
            )
            for i in range(5)
        }
        repairer = Rosame26MipRepairer(
            _head("blocksworld", tmp_path), build_domain("blocksworld"),
            contexts, {}, mip_traces=3,
        )
        repairer.update(
            torch.arange(5), torch.rand(5, 2, 4), torch.rand(5, 3, 2),
            torch.rand(5, 4), torch.rand(5, 4),
        )
        assert len(repairer._collected) == 3

    def test_a_length_disagreement_raises(self, tmp_path) -> None:
        """Tensors and contexts built from different trace lists must not pass."""
        contexts = {
            0: TraceContext(
                instance=None, calls=[("a", []), ("a", [])],
                init_fluents=set(), goal_fluents=None,
            )
        }
        repairer = Rosame26MipRepairer(
            _head("blocksworld", tmp_path), build_domain("blocksworld"),
            contexts, {}, lengths=torch.tensor([7]),
        )
        with pytest.raises(ValueError, match="tensors and the contexts disagree"):
            repairer.update(
                torch.tensor([0]), torch.rand(1, 8, 4), torch.rand(1, 9, 2),
                torch.rand(1, 4), torch.rand(1, 4),
            )

    def test_a_solve_with_no_traces_reports_rather_than_raises(self, tmp_path) -> None:
        repairer = Rosame26MipRepairer(
            _head("blocksworld", tmp_path), build_domain("blocksworld"), {}, {}
        )
        assert repairer.run_fixer(1.0) is None
        assert repairer.rounds[-1]["status"] == "NO_TRACES"


# ── the gate: a real ragged fold through the real encoder ───────────────


class TestAgainstTheRealEncoder:
    """One real blocksworld fold, solved, with the label shapes checked.

    This is what settles §6.1's ragged-label concern. The vendored
    ``extract_sol_label`` sizes every trace from one shared ``problem.max_t``;
    reading ``repaired_states`` per trace instead makes the arithmetic land
    exactly — ``repaired_states`` returns one frame per image, dropping both
    endpoints leaves ``N - 2 = T``, and ``z`` is ``[T, S]``.
    """

    CELL = Path(
        "benchmark/running_results/blocksworld/"
        "TO=600__blocks_predefined_problems1-10_final-version/testing/"
        "fold0_numtrajs3_gtrate0"
    )

    def _fold(self, tmp_path):
        import json

        from pddl_plus_parser.lisp_parsers import DomainParser

        from benchmark.backfill_baseline import _build_prepared_trajectories
        from benchmark.backfill_common import resolve_data_dir, resolve_problem_dir
        from benchmark.baselines.image_fold_inputs import resolve_fold_inputs
        from benchmark.baselines.rosame26_data import build_fold_batch

        if not self.CELL.exists():
            pytest.skip(f"{self.CELL} not on disk")
        info = json.loads((self.CELL / "fold_info.json").read_text())
        experiment = self.CELL.parent.parent
        data_dir, _ = resolve_data_dir(experiment, None)
        prepared = _build_prepared_trajectories(
            self.CELL, resolve_problem_dir(experiment, data_dir), info
        )
        partial = DomainParser(
            self.CELL / "domain_reference.pddl", partial_parsing=True
        ).parse_domain()
        traces = resolve_fold_inputs(partial, prepared, "blocksworld")
        batch = build_fold_batch(
            traces, partial, "blocksworld", assets_root=tmp_path / "g", resize=64
        )
        return partial, traces, batch

    def _contexts(self, partial, traces, batch):
        from benchmark.baselines.rosame26_data import canonical
        from src.milp.converter import build_ps_instance

        ps_domain = build_domain("blocksworld")
        contexts = {}
        for index, name in enumerate(batch.kept):
            trace = next(t for t in traces if t.problem_name == name)
            calls = [
                (call.split()[0], call.split()[1:])
                for call in (canonical(a) for a in trace.action_strings)
            ]
            goal = {
                (p.split()[0], tuple(p.split()[1:]))
                for p in (canonical(x) for x in trace.gt_final_predicates)
            }
            init = set()
            for grounded in trace.problem.initial_state_predicates.values():
                for one in grounded:
                    if not getattr(one, "is_positive", True):
                        continue
                    init.add(
                        (
                            canonical(one.name),
                            tuple(one.object_mapping[k] for k in one.signature.keys()),
                        )
                    )
            contexts[index] = TraceContext(
                instance=build_ps_instance(ps_domain, partial, trace.problem),
                calls=calls, init_fluents=init, goal_fluents=goal,
            )
        return ps_domain, contexts

    def test_it_solves_and_labels_a_ragged_fold(self, tmp_path) -> None:
        from src.milp.rosame26_training import Rosame26Trainer, default_parameters

        partial, traces, batch = self._fold(tmp_path)
        ps_domain, contexts = self._contexts(partial, traces, batch)
        lengths = batch.batch.lengths
        assert len(set(lengths.tolist())) > 1, "this fold is not ragged"

        parameters = default_parameters(
            domain="blocksworld",
            domain_assets_root=batch.grounding.assets_root,
            epoch=2, batch_size=128, pre_mip_epoch=2, seed=1,
        )
        trainer = Rosame26Trainer(tmp_path / "t", parameters, mip_repairer=None)
        trainer.train(batch.batch)

        repairer = Rosame26MipRepairer(
            trainer.domain_model, ps_domain, contexts,
            batch.grounding.proposition_index, mip_traces=3, lengths=lengths,
        )
        with torch.no_grad():
            outputs = trainer.net(
                batch.batch.images, batch.batch.action_traces,
                batch.batch.state_traces[:, 0, :], batch.batch.state_traces[:, -1, :],
            )
        repairer.clear()
        repairer.update(
            torch.arange(len(batch.kept)), outputs["z"], outputs["a"],
            batch.batch.state_traces[:, 0, :], batch.batch.state_traces[:, -1, :],
        )
        repairer.run_fixer(60.0)

        assert repairer.rounds[-1]["status"] == "OK", repairer.rounds[-1]
        width = len(batch.grounding.proposition_index)
        for index, (_, state_label, _) in repairer.pseudo_labels.traces.items():
            assert state_label.shape == (int(lengths[index].item()), width), index

    def test_the_model_labels_cover_every_schema(self, tmp_path) -> None:
        from src.milp.rosame26_training import Rosame26Trainer, default_parameters

        partial, traces, batch = self._fold(tmp_path)
        ps_domain, contexts = self._contexts(partial, traces, batch)
        parameters = default_parameters(
            domain="blocksworld",
            domain_assets_root=batch.grounding.assets_root,
            epoch=2, batch_size=128, pre_mip_epoch=2, seed=1,
        )
        trainer = Rosame26Trainer(tmp_path / "t", parameters, mip_repairer=None)
        trainer.train(batch.batch)
        repairer = Rosame26MipRepairer(
            trainer.domain_model, ps_domain, contexts,
            batch.grounding.proposition_index, mip_traces=3,
            lengths=batch.batch.lengths,
        )
        with torch.no_grad():
            outputs = trainer.net(
                batch.batch.images, batch.batch.action_traces,
                batch.batch.state_traces[:, 0, :], batch.batch.state_traces[:, -1, :],
            )
        repairer.update(
            torch.arange(len(batch.kept)), outputs["z"], outputs["a"],
            batch.batch.state_traces[:, 0, :], batch.batch.state_traces[:, -1, :],
        )
        repairer.run_fixer(60.0)

        labels = repairer.pseudo_labels.model
        assert set(labels) == {s.name for s in trainer.domain_model.action_schemas}
        for schema in trainer.domain_model.action_schemas:
            assert labels[schema.name].shape == (int(schema.randn.shape[0]), 4)
