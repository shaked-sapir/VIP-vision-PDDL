"""The PI-SAM runner is registered and describes itself as a MILP-free symbolic arm.

    python -m pytest benchmark/baselines/test_pisam_runner.py
"""

from __future__ import annotations

from benchmark.baselines import BASELINE_REGISTRY, resolve_baselines
from benchmark.baselines.pisam_runner import PisamRawRunner


def test_the_registry_key_resolves_to_the_runner():
    assert BASELINE_REGISTRY["pisam"] == [PisamRawRunner]
    (runner,) = resolve_baselines(["pisam"], rosame_seed=42, batch_size=0)
    assert isinstance(runner, PisamRawRunner)


def test_the_row_label_and_factors():
    runner = PisamRawRunner()
    assert runner.name == "PISAM"
    assert runner.factors() == {"input_kind": "symbolic", "paper": "aaai24", "uses_milp": False}


def test_run_params_record_the_precondition_policy():
    assert PisamRawRunner().run_params() == {
        "negative_preconditions_policy": "hard", "pisam_seed": 42,
    }


def test_no_trajectories_yields_no_model(tmp_path):
    domain = tmp_path / "domain.pddl"
    domain.write_text(
        "(define (domain d) (:requirements :strips :typing) (:types block)\n"
        " (:predicates (clear ?x - block))\n"
        " (:action noop :parameters (?x - block) :precondition (and) :effect (and)))\n"
    )
    model, report = PisamRawRunner().learn(domain, [], tmp_path)
    assert model is None
    assert report["n_traces"] == 0
