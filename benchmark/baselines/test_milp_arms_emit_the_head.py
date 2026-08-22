"""Every MILP arm reports the *learned* model, not its solver's solution.

    python -m pytest benchmark/baselines/test_milp_arms_emit_the_head.py

The three ICAPS-24 MILP arms used to emit ``solution_to_pddl(...)`` — the CP-SAT
solution, which is logically consistent by construction and therefore scores the
*solver*. On one blocksworld fold that read precision 1.00 / recall 1.00 while
the network agreed with its own MILP on 35% of bindings.

Upstream ICAPS-26 emits the head (``network.py:300`` → ``dump_actions`` →
``extract_pddl(self.domain_model)``; ``action_model_sol`` appears only inside
``convertor.py``, building pseudo-labels), and AMLGym ships no MILP variant at
all, so nothing upstream said otherwise. All four MILP arms now agree.
"""

from __future__ import annotations

import inspect

import pytest

from benchmark.baselines import BASELINE_REGISTRY, get_baselines

#: Every arm that runs a MILP alongside a learner.
MILP_KEYS = [
    "rosame_i_milp_24",
    "rosame_milp_24",
    "rosame_milp_24_tag",
    "rosame_i_milp_26",
]


class TestNoArmEmitsTheSolverSolution:
    @pytest.mark.parametrize("key", MILP_KEYS)
    def test_the_runner_does_not_call_solution_to_pddl(self, key) -> None:
        """The direct check: the emission path must not reach that helper."""
        module = inspect.getmodule(type(get_baselines([key])[0]))
        source = inspect.getsource(module)
        calls = [
            line
            for line in source.splitlines()
            if "solution_to_pddl(" in line and not line.strip().startswith(("#", "*"))
            and "``" not in line
        ]
        assert calls == [], f"{key} still emits the solver's model: {calls}"

    def test_the_helper_still_exists_for_inspection(self) -> None:
        """Not deleted — a solver solution is still worth being able to render."""
        from benchmark.algorithm_adapters.rosame_milp import model_bridge

        assert callable(model_bridge.solution_to_pddl)


class TestTheImagedArmsAgree:
    def test_both_report_the_learned_model(self) -> None:
        """The 24 and 26 imaged arms must be comparable, which was the point."""
        from benchmark.baselines.rosame_i_milp_runner import RosameIMilpRunner

        source = inspect.getsource(RosameIMilpRunner._model_from)
        assert "rosame.to_pddl()" in source
        assert "solution_to_pddl(rosame" not in source

    def test_milp_failed_survives_as_a_diagnostic(self) -> None:
        """A cell whose solver never succeeded is its DL-only sibling."""
        from benchmark.baselines.rosame_i_milp_runner import RosameIMilpRunner

        assert "milp_failed" in inspect.getsource(RosameIMilpRunner._model_from) or (
            "final_solution" in inspect.getsource(RosameIMilpRunner._model_from)
        )
