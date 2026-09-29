"""Tests for trajectory parsing with a complete object table."""

from pathlib import Path

import pytest
from pddl_plus_parser.lisp_parsers import DomainParser

from src.utils.pddl_state import ground_observation_completely
from src.utils.pddl_trajectory import complete_object_table, parse_trajectory_with_declared_types

GRIPPER_DOMAIN = Path(__file__).resolve().parents[1] / "domains" / "gripper" / "gripper.pddl"

# rooma is never mentioned in the initial state; it first appears after the move.
TRAJECTORY_TEXT = """(
(:init (at ball1 roomb) (at-robby roomb) (free left) (free right))
(operator: (move roomb rooma))
(:state (at ball1 roomb) (at-robby rooma) (free left) (free right))
)
"""


@pytest.fixture
def gripper_domain():
    return DomainParser(GRIPPER_DOMAIN, partial_parsing=True).parse_domain()


@pytest.fixture
def trajectory_path(tmp_path: Path) -> Path:
    path = tmp_path / "late_room.trajectory"
    path.write_text(TRAJECTORY_TEXT)
    return path


def test_object_first_seen_after_init_is_in_the_table(gripper_domain, trajectory_path):
    observation = parse_trajectory_with_declared_types(trajectory_path, gripper_domain)

    assert "rooma" in observation.grounded_objects
    assert observation.grounded_objects["rooma"].type.name == "room"
    assert observation.grounded_objects["roomb"].type.name == "room"


def test_complete_table_grounds_without_error(gripper_domain, trajectory_path):
    observation = parse_trajectory_with_declared_types(trajectory_path, gripper_domain)

    grounded = ground_observation_completely(gripper_domain, observation)

    next_state = grounded.components[0].next_state.state_predicates
    at_robby = next(preds for key, preds in next_state.items() if key.startswith("(at-robby"))
    assert {p.object_mapping["?r"] for p in at_robby} == {"rooma", "roomb"}


def test_complete_object_table_keeps_existing_entries(gripper_domain, trajectory_path):
    observation = parse_trajectory_with_declared_types(trajectory_path, gripper_domain)
    before = dict(observation.grounded_objects)

    complete_object_table(observation)

    assert observation.grounded_objects == before
