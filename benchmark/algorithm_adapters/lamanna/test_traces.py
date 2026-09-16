"""The trace writer emits the grammar both Lamanna learners parse.

    python -m pytest benchmark/algorithm_adapters/lamanna/test_traces.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pddl_plus_parser.lisp_parsers import DomainParser

from benchmark.algorithm_adapters.lamanna.traces import (
    format_trajectory,
    state_literals,
    write_partial_trace,
)
from src.utils.pddl_state import get_state_grounded_predicates, ground_observation_completely
from src.utils.pddl_trajectory import parse_trajectory_with_declared_types

ROOT = Path(__file__).resolve().parents[3]
BLOCKS_DOMAIN = ROOT / "src" / "domains" / "blocks" / "blocks.pddl"

PROBLEM = """(define (problem tiny) (:domain blocks)
  (:objects a - block b - block)
  (:init (clear a) (clear b) (ontable a) (ontable b) (handempty))
  (:goal (and (holding a))))
"""

TRAJECTORY = """(
(:init (clear a) (clear b) (ontable a) (ontable b) (handempty))
(operator: (pick_up a))
(:state (clear b) (ontable b) (holding a))
(operator: (stack a b))
(:state (clear a) (ontable b) (on a b) (handempty))
)
"""

# 2 blocks: on 4 + ontable 2 + clear 2 + holding 2 + handempty 1
N_ATOMS = 11


def _state_lines(text: str):
    return [line for line in text.splitlines() if line.startswith("(:state")]


def _action_lines(text: str):
    return [line for line in text.splitlines() if line.startswith("(:action")]


@pytest.fixture
def observation(tmp_path):
    (tmp_path / "tiny.pddl").write_text(PROBLEM)
    (tmp_path / "tiny.trajectory").write_text(TRAJECTORY)
    domain = DomainParser(BLOCKS_DOMAIN, partial_parsing=True).parse_domain()
    parsed = parse_trajectory_with_declared_types(
        tmp_path / "tiny.trajectory", domain, tmp_path / "tiny.pddl"
    )
    return ground_observation_completely(domain, parsed)


def _mask(state, representation: str) -> None:
    for predicate in get_state_grounded_predicates(state):
        if predicate.untyped_representation == representation:
            predicate.is_masked = True
            return
    raise AssertionError(f"{representation} not in state")


class TestShape:
    def test_states_and_actions_alternate_one_per_line(self, observation):
        text = format_trajectory(observation)
        lines = [line for line in text.splitlines() if line.strip()]
        assert lines[0] == "(:trajectory"
        assert lines[-1] == ")"
        body = lines[1:-1]
        assert [line.split()[0] for line in body] == [
            "(:state", "(:action", "(:state", "(:action", "(:state",
        ]

    def test_actions_keep_name_and_arguments(self, observation):
        assert _action_lines(format_trajectory(observation)) == [
            "(:action (pick_up a))",
            "(:action (stack a b))",
        ]

    def test_a_complete_state_lists_every_atom_with_its_sign(self, observation):
        first = _state_lines(format_trajectory(observation))[0]
        literals = re.findall(r"\(not \([^()]*\)\)|\([^()]*\)", first[len("(:state "):-1])
        assert len(literals) == N_ATOMS
        assert "(handempty)" in literals
        assert "(not (on a b))" in literals
        assert "(not (holding a))" in literals

    def test_zero_arity_atoms_have_no_stray_space(self, observation):
        text = format_trajectory(observation)
        assert "(handempty)" in text
        assert "(handempty )" not in text
        assert "(not (handempty))" in text


class TestMasking:
    def test_a_masked_atom_is_absent_in_both_signs(self, observation):
        second = observation.components[0].next_state
        _mask(second, "(holding a)")
        _mask(second, "(not (on a b))")
        line = _state_lines(format_trajectory(observation))[1]
        assert "(holding a)" not in line
        assert "(not (holding a))" not in line
        assert "(on a b)" not in line
        assert len(state_literals(second)) == N_ATOMS - 2

    def test_other_states_are_untouched(self, observation):
        _mask(observation.components[0].next_state, "(holding a)")
        lines = _state_lines(format_trajectory(observation))
        assert "(not (holding a))" in lines[0]
        assert "(not (holding a))" in lines[2]


class TestRoundTrip:
    def test_nolam_parses_what_we_write(self, observation, tmp_path, monkeypatch):
        from nolam.algorithm.ActionModel import ActionModel
        from nolam.algorithm.Learner import Learner

        _mask(observation.components[0].next_state, "(holding a)")
        path = write_partial_trace(observation, tmp_path / "out" / "t.trajectory")
        monkeypatch.chdir(tmp_path)
        model = ActionModel(input_file=str(BLOCKS_DOMAIN))
        model.init_prec_eff()
        trace = Learner().parse_trace(str(path), model)

        assert len(trace.observations) == 3
        assert [a.operator_name for a in trace.actions] == ["pick_up", "stack"]
        sizes = [
            sum(len(v) for v in obs.positive_literals.values())
            + sum(len(v) for v in obs.negative_literals.values())
            for obs in trace.observations
        ]
        assert sizes == [N_ATOMS, N_ATOMS - 1, N_ATOMS]
        assert "holding(a)" not in trace.observations[1]
        assert "not_holding(a)" not in trace.observations[1]
        assert "handempty()" in trace.observations[0]

    def test_offlam_parses_what_we_write(self, observation, tmp_path, monkeypatch):
        from offlam.src.Learner import Learner

        path = write_partial_trace(observation, tmp_path / "t.trajectory")
        monkeypatch.chdir(tmp_path)  # OffLAM's Learner writes PDDL/ relative to the cwd
        learner = Learner(input_domain_path=str(BLOCKS_DOMAIN))
        trace = learner.parse_trace(str(path))
        assert len(trace.observations) == 3
        assert [a.operator_name for a in trace.actions] == ["pick_up", "stack"]
