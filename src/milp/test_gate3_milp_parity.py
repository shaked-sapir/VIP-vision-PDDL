"""Gate 3 — the MILP-parity gate, on the pieces Phase 5 actually built.

    python -m pytest src/milp/test_gate3_milp_parity.py

Plan §9 item 3 lists five obligations. Two were written before §0.1a, and assert
things that are now known to be *wrong*; both are restated here to their intent
rather than dropped, with the reason recorded beside each.

============================  ==============================================
plan's wording                what this file asserts, and why
============================  ==============================================
labels match a direct         RESTATED. We deliberately do not call the
vendored-translator call      vendored translator: it sizes every trace from
                              one shared ``max_t`` (§6.1) and its model channel
                              is unpermuted (§0.1a). Parity with it would pin
                              the defect. Asserted instead: the labels match an
                              independent recomputation from the *same solved
                              encoder*.
the §0.1 identity mappings    RESTATED per §0.1a: identity is wrong on four of
reach ``extract_sol_*``       five domains. Asserted instead: the model channel
                              goes through the row alignment, and every row's
                              label lands on its own binding.
``trans_full_state``'s zip    RESTATED. That function is vendored and reached
is index-aligned              only by ``test_vendor_translator_contract``.
                              Asserted instead for our equivalent: the
                              proposition index and the encoder's grounding
                              agree column for column.
shapes match for a RAGGED     DIRECT. Held on a real fold whose traces differ
bundle                        in length.
``chosen = 0`` never fires    N/A by construction, and asserted as such: that
                              fallback lives in the vendored
                              ``extract_sol_label``, and under option B the
                              action is observed, so nothing infers it.
============================  ==============================================
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict

import pytest
import torch

import src.milp  # noqa: F401  (vendor sys.path bootstrap)
from src.milp.domain_assets import build_domain
from src.milp.rosame26_repairer import (
    OBJECTIVES,
    Rosame26MipRepairer,
    TraceContext,
    model_labels_in_head_order,
    reference_action_model,
    state_labels_for,
)
from src.milp.schema_row_alignment import schema_row_keys

CELL = Path(
    "benchmark/running_results/blocksworld/"
    "TO=600__blocks_predefined_problems1-10_final-version/testing/"
    "fold0_numtrajs3_gtrate0"
)


@pytest.fixture(scope="module")
def solved():
    """A real fold, trained briefly, with one MILP round solved over it."""
    from pddl_plus_parser.lisp_parsers import DomainParser

    from benchmark.backfill_baseline import _build_prepared_trajectories
    from benchmark.backfill_common import resolve_data_dir, resolve_problem_dir
    from benchmark.baselines.image_fold_inputs import resolve_fold_inputs
    from benchmark.baselines.rosame26_data import build_fold_batch, canonical
    from benchmark.baselines.rosame26_milp_runner import _init_fluents
    from src.milp.converter import build_ps_instance
    from src.milp.rosame26_training import Rosame26Trainer, default_parameters

    if not CELL.exists():
        pytest.skip(f"{CELL} not on disk")

    info = json.loads((CELL / "fold_info.json").read_text())
    experiment = CELL.parent.parent
    data_dir, _ = resolve_data_dir(experiment, None)
    prepared = _build_prepared_trajectories(
        CELL, resolve_problem_dir(experiment, data_dir), info
    )
    partial = DomainParser(
        CELL / "domain_reference.pddl", partial_parsing=True
    ).parse_domain()
    traces = resolve_fold_inputs(partial, prepared, "blocksworld")

    work = Path(tempfile.mkdtemp(prefix="gate3_"))
    fold = build_fold_batch(
        traces, partial, "blocksworld", assets_root=work / "g", resize=64
    )
    ps_domain = build_domain("blocksworld")
    by_name = {trace.problem_name: trace for trace in traces}

    contexts: Dict[int, TraceContext] = {}
    for index, name in enumerate(fold.kept):
        trace = by_name[name]
        contexts[index] = TraceContext(
            instance=build_ps_instance(ps_domain, partial, trace.problem),
            calls=[
                (call.split()[0], call.split()[1:])
                for call in (canonical(a) for a in trace.action_strings)
            ],
            init_fluents=_init_fluents(trace.problem),
            goal_fluents={
                (f.split()[0], tuple(f.split()[1:]))
                for f in (canonical(p) for p in trace.gt_final_predicates)
            }
            or None,
        )

    parameters = default_parameters(
        domain="blocksworld",
        domain_assets_root=fold.grounding.assets_root,
        epoch=3, batch_size=128, pre_mip_epoch=3, seed=8800,
    )
    trainer = Rosame26Trainer(work / "t", parameters, mip_repairer=None)
    trainer.train(fold.batch)

    repairer = Rosame26MipRepairer(
        trainer.domain_model, ps_domain, contexts,
        fold.grounding.proposition_index, mip_traces=3,
        gt_action_model=reference_action_model("blocksworld"),
        lengths=fold.batch.lengths,
    )
    with torch.no_grad():
        outputs = trainer.net(
            fold.batch.images, fold.batch.action_traces,
            fold.batch.state_traces[:, 0, :], fold.batch.state_traces[:, -1, :],
        )
    repairer.clear()
    repairer.update(
        torch.arange(len(fold.kept)), outputs["z"], outputs["a"],
        fold.batch.state_traces[:, 0, :], fold.batch.state_traces[:, -1, :],
    )
    distance = repairer.run_fixer(60.0)
    return {
        "fold": fold, "ps_domain": ps_domain, "contexts": contexts,
        "trainer": trainer, "repairer": repairer, "outputs": outputs,
        "distance": distance,
    }


class TestTheFoldIsRagged:
    """Obligation 4's precondition: a uniform bundle would prove nothing."""

    def test_traces_differ_in_length(self, solved) -> None:
        lengths = solved["fold"].batch.lengths.tolist()
        assert len(set(lengths)) > 1, lengths


class TestTheSolveSucceeded:
    def test_a_round_was_recorded(self, solved) -> None:
        assert solved["repairer"].rounds[-1]["status"] == "OK"

    def test_the_gt_diagnostic_reported(self, solved) -> None:
        assert solved["distance"] is not None


class TestShapesMatchOnARaggedBundle:
    """Obligation 4, held directly."""

    def test_each_state_label_matches_its_own_trace(self, solved) -> None:
        lengths = solved["fold"].batch.lengths
        width = len(solved["fold"].grounding.proposition_index)
        labels = solved["repairer"].pseudo_labels.traces
        assert labels, "no trace labels were written"
        for index, (_, state_label, _) in labels.items():
            assert state_label.shape == (int(lengths[index].item()), width), index

    def test_the_labels_slice_z_without_reshaping(self, solved) -> None:
        """What ``loss_pseudo_s`` does: ``z[b][:live]`` against ``label[:live]``."""
        z = solved["outputs"]["z"]
        lengths = solved["fold"].batch.lengths
        for index, (_, state_label, _) in solved["repairer"].pseudo_labels.traces.items():
            live = int(lengths[index].item())
            assert z[index][:live].shape == state_label[:live].shape

    def test_model_labels_match_each_head_width(self, solved) -> None:
        labels = solved["repairer"].pseudo_labels.model
        for schema in solved["trainer"].domain_model.action_schemas:
            assert labels[schema.name].shape == (int(schema.randn.shape[0]), 4)


class TestLabelsMatchAnIndependentRecomputation:
    """Obligation 1, restated: parity against the same solved encoder.

    Not against the vendored translator — it sizes from one shared ``max_t``
    (§6.1) and leaves the model channel unpermuted (§0.1a), so matching it would
    pin the defect rather than the contract.
    """

    def test_state_labels_recompute_identically(self, solved) -> None:
        repairer = solved["repairer"]
        contexts = solved["contexts"]
        index_map = solved["fold"].grounding.proposition_index
        collected = [item.index for item in repairer._collected]

        for position, trace_index in enumerate(collected, start=1):
            expected = state_labels_for(
                contexts[trace_index].instance,
                repairer._last_encoder.repaired_states(position),
                index_map,
            )
            _, actual, _ = repairer.pseudo_labels.traces[trace_index]
            assert torch.equal(actual, expected), trace_index

    def test_model_labels_recompute_identically(self, solved) -> None:
        repairer = solved["repairer"]
        expected = model_labels_in_head_order(
            repairer._last_encoder.action_model_sol(),
            solved["trainer"].domain_model,
            solved["ps_domain"],
        )
        for name, label in repairer.pseudo_labels.model.items():
            assert torch.equal(label, expected[name]), name


class TestTheModelChannelGoesThroughTheAlignment:
    """Obligation 2, restated per §0.1a.

    The plan asked that identity mappings reach ``extract_sol_*``. Identity is
    wrong on four of five domains, and those functions are never called. What
    must hold is that every row's label describes *that row's own binding*.
    """

    def test_every_row_agrees_with_the_solution_at_its_own_key(self, solved) -> None:
        repairer = solved["repairer"]
        ps_domain = solved["ps_domain"]
        solution = repairer._last_encoder.action_model_sol()
        keys = schema_row_keys(solved["trainer"].domain_model, ps_domain)

        for name, label in repairer.pseudo_labels.model.items():
            schema = ps_domain.get_action_schema(name)
            for row, (predicate_name, args) in enumerate(keys[name]):
                key = (schema, ps_domain.get_predicate(predicate_name), args)
                klass = int(label[row].argmax().item())
                if bool(solution.add.get(key, 0)):
                    assert klass == 1, (name, row)
                elif bool(solution.dele.get(key, 0)):
                    assert klass == 3, (name, row)
                elif bool(solution.pre.get(key, 0)):
                    assert klass == 2, (name, row)
                else:
                    assert klass == 0, (name, row)


class TestThePropositionIndexIsColumnAligned:
    """Obligation 3, restated for the path we actually use.

    ``trans_full_state`` is vendored and reached only by
    ``test_vendor_translator_contract``. Our equivalent is the head column map,
    and what matters is that it and the encoder's grounding name the same
    proposition at the same column.
    """

    def test_every_column_resolves_to_its_own_proposition(self, solved) -> None:
        from src.milp.converter import proposition_of

        index_map = solved["fold"].grounding.proposition_index
        instance = next(iter(solved["contexts"].values())).instance
        resolved = 0
        for key, column in index_map.items():
            parts = key.split()
            prop = proposition_of(instance, parts[0], parts[1:]) if parts else None
            if prop is None:
                continue  # an object this problem does not have (§4.2a′)
            resolved += 1
            assert str(prop).replace("_", " ").split()[0] == parts[0].replace("_", " ").split()[0]
        assert resolved > 0, "no column resolved; the two groundings are unrelated"

    def test_the_index_is_a_bijection_onto_its_columns(self, solved) -> None:
        index_map = solved["fold"].grounding.proposition_index
        assert sorted(index_map.values()) == list(range(len(index_map)))


class TestTheActionFallbackCannotFire:
    """Obligation 5, asserted as the reason rather than the symptom.

    ``chosen = 0`` is the vendored ``extract_sol_label``'s default when no action
    variable is set at a step. Under option B the action is observed, so this
    repairer never infers one — the action channel is not even an objective.
    """

    def test_action_is_not_a_solver_objective(self) -> None:
        assert "action" not in OBJECTIVES

    def test_action_labels_are_placeholders_not_inferences(self, solved) -> None:
        """They exist to fill the tuple; ``MIP_to_DL`` excludes ``action``."""
        for _, _, action_label in solved["repairer"].pseudo_labels.traces.values():
            assert torch.all(action_label == 0)

    def test_the_loop_does_not_consume_them(self, solved) -> None:
        assert "action" not in solved["trainer"].parameters["MIP_to_DL"]
