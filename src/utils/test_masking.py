"""Regression tests for object typing when loading observations.

Depot problem9's initial state mentions ``pile2`` last in ``(clear pile2)``, whose
slot is untyped, so a parse without the problem file types ``pile2`` as ``object``
and PI-SAM can no longer lift ``(at-pile pile2 d2)`` into ``drop``'s
``?pl - pile`` parameter.
"""

from pathlib import Path

import pytest
from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain

from src.pi_sam.pi_sam_learning import PISAMLearner
from src.utils.masking import load_masked_observation
from src.utils.pddl_trajectory import parse_trajectory_with_declared_types

DEPOT_DOMAIN = Path(__file__).resolve().parents[2] / "src" / "domains" / "depot" / "depot.pddl"

# One transition of depot problem9: the crane at d2 drops p1 onto the empty pile2.
PROBLEM9_DROP_TRAJECTORY = """(
(:init (at p1 d2) (at-crane c1 d1) (at-crane c2 d2) (at-pile pile1 d1) (at-pile pile2 d2) (at-truck t1 d2) (clear pile1) (clear pile2) (empty-crane c1) (holding c2 p1) (in-truck p2 t1) (in-truck p3 t1))
(operator: (drop c2 p1 pile2 d2))
(:state (at p1 d2) (at-crane c1 d1) (at-crane c2 d2) (at-pile pile1 d1) (at-pile pile2 d2) (at-truck t1 d2) (clear p1) (clear pile1) (clear pile2) (empty-crane c1) (empty-crane c2) (in-truck p2 t1) (in-truck p3 t1) (on-pile p1 pile2))
)
"""

PROBLEM9_PDDL = """(define (problem depot-d2-p3-t1-p10)
  (:domain depot)
  (:objects
    d1 d2 - depot
    t1 - truck
    c1 c2 - crane
    pile1 pile2 - pile
    p1 p2 p3 - package
  )
  (:init
    (at-truck t1 d2) (at-crane c1 d1) (empty-crane c1) (at-crane c2 d2) (holding c2 p1)
    (at-pile pile1 d1) (at-pile pile2 d2) (at p1 d2) (in-truck p2 t1) (in-truck p3 t1)
    (clear pile1) (clear pile2)
  )
  (:goal (and (on-pile p1 pile2)))
)
"""


@pytest.fixture
def depot_domain() -> Domain:
    return DomainParser(DEPOT_DOMAIN, partial_parsing=True).parse_domain()


@pytest.fixture
def problem9_files(tmp_path: Path) -> dict:
    trajectory = tmp_path / "problem9.trajectory"
    trajectory.write_text(PROBLEM9_DROP_TRAJECTORY)
    problem = tmp_path / "problem9.pddl"
    problem.write_text(PROBLEM9_PDDL)
    masking = tmp_path / "problem9.masking_info"
    masking.write_text("\n\n")  # two states, nothing masked
    return {"trajectory": trajectory, "problem": problem, "masking": masking}


def _drop_preconditions(domain: Domain, observation) -> str:
    learned, _ = PISAMLearner(domain).learn_action_model([observation])
    return str(learned.actions["drop"].preconditions)


def test_parse_without_problem_infers_pile2_as_object(depot_domain, problem9_files):
    observation = parse_trajectory_with_declared_types(problem9_files["trajectory"], depot_domain)
    assert observation.grounded_objects["pile2"].type.name == "object"


def test_parse_with_problem_takes_declared_pile_type(depot_domain, problem9_files):
    observation = parse_trajectory_with_declared_types(
        problem9_files["trajectory"], depot_domain, problem9_files["problem"]
    )
    assert observation.grounded_objects["pile2"].type.name == "pile"
    assert observation.grounded_objects["pile1"].type.name == "pile"


def test_parse_with_missing_problem_falls_back_to_inference(depot_domain, problem9_files):
    observation = parse_trajectory_with_declared_types(
        problem9_files["trajectory"], depot_domain, problem9_files["problem"].with_name("absent.pddl")
    )
    assert observation.grounded_objects["pile2"].type.name == "object"


def test_load_masked_observation_carries_declared_types(depot_domain, problem9_files):
    observation = load_masked_observation(
        problem9_files["trajectory"], problem9_files["masking"], depot_domain,
        problem_path=problem9_files["problem"],
    )
    assert observation.grounded_objects["pile2"].type.name == "pile"
    assert len(observation.components) == 1


def test_pisam_keeps_at_pile_precondition_on_drop_with_declared_types(depot_domain, problem9_files):
    with_problem = load_masked_observation(
        problem9_files["trajectory"], problem9_files["masking"], depot_domain,
        problem_path=problem9_files["problem"],
    )
    assert "at-pile" in _drop_preconditions(depot_domain, with_problem)


def test_pisam_drops_at_pile_precondition_when_pile2_is_mistyped(depot_domain, problem9_files):
    without_problem = load_masked_observation(
        problem9_files["trajectory"], problem9_files["masking"], depot_domain,
    )
    assert "at-pile" not in _drop_preconditions(depot_domain, without_problem)
