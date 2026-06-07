import os
from collections import defaultdict
from functools import cached_property
from typing import Any, Dict, List, Set, TypeVar

from amlgym.modeling.env import Env
from tarski.grounding import LPGroundingStrategy
from tarski.io import PDDLReader as TarskiPDDLReader
from unified_planning.exceptions import UPInvalidActionError
from unified_planning.io import PDDLReader, PDDLWriter
from unified_planning.model import Fluent, Problem, UPState
from unified_planning.plans import ActionInstance
from unified_planning.shortcuts import BoolType, FALSE, TRUE, SequentialSimulator

ObservationType = TypeVar("ObservationType")


class CompatibleUPEnv(Env):
    """A UPEnv variant compatible with newer unified-planning builds."""

    problem: Problem
    _simulator: SequentialSimulator

    def __init__(self, domain_path: str, problem_path: str) -> None:
        self.problem = PDDLReader().parse_problem(domain_path, problem_path)
        self._simulator = SequentialSimulator(self.problem)
        self._all_neg_fluents = {f: FALSE() for f, _ in self.problem.initial_values.items()}
        self.all_neg_state = self._build_up_state(self._all_neg_fluents)

        # Keep AMLGym's tarski grounding logic.
        _tmp_problem = PDDLReader().parse_problem(domain_path, problem_path)
        dummy_fluent = Fluent("dummy", BoolType())
        if dummy_fluent not in _tmp_problem.fluents:
            _tmp_problem.add_fluent(dummy_fluent)
        _tmp_problem.set_initial_value(dummy_fluent, True)
        for action in _tmp_problem.actions:
            action.clear_preconditions()
            action.clear_effects()
            action.add_precondition(dummy_fluent)
            action.add_effect(dummy_fluent, True)
        _tmp_problem.clear_goals()

        tmp_domain_path = "tmp_domain.pddl"
        tmp_problem_path = "tmp_problem.pddl"
        PDDLWriter(_tmp_problem).write_domain(tmp_domain_path)
        PDDLWriter(_tmp_problem).write_problem(tmp_problem_path)
        reader = TarskiPDDLReader(raise_on_error=True)
        reader.parse_domain(tmp_domain_path)
        reader.parse_instance(tmp_problem_path)
        os.remove(tmp_domain_path)
        os.remove(tmp_problem_path)
        self._grounder = LPGroundingStrategy(reader.problem)

    def _build_up_state(self, values: Dict[Any, Any], parent: UPState | None = None) -> UPState:
        if parent is not None:
            try:
                return UPState(values, parent)
            except Exception:
                pass
        return UPState(values)

    def _str_to_action(self, action_label: str) -> ActionInstance:
        action_split = action_label.strip()[1:-1].split()
        op_name = action_split[0]
        obj_names = [o.strip() for o in action_split[1:]] if len(action_split) > 1 else []
        up_op = self.problem.action(op_name)
        up_objs = [self.problem.object(o) for o in obj_names]
        return ActionInstance(up_op, up_objs)

    def _state_from_literals(self, state: Set[str] | List[str]) -> UPState:
        pos_literals = {l for l in state if not l.startswith("(not ")}
        prob_state_fluents = {}

        for f in pos_literals:
            f_split = f[1:-1].split()
            f_name = f_split[0]
            f_objs = f_split[1:] if len(f_split) > 1 else []
            prob_f = self.problem.fluent(f_name)
            prob_args = [self.problem.object(o) for o in f_objs]
            prob_state_fluents[prob_f(*prob_args)] = TRUE()

        try:
            return self.all_neg_state.make_child(prob_state_fluents)
        except Exception:
            merged = dict(self._all_neg_fluents)
            merged.update(prob_state_fluents)
            return self._build_up_state(merged)

    def apply(self, state, action):
        if isinstance(action, str):
            action = self._str_to_action(action)

        if isinstance(state, Set) or isinstance(state, List):
            state = self._state_from_literals(state)

        try:
            next_state = self._simulator.apply(state, action)
        except UPInvalidActionError:
            next_state = None

        if next_state is None:
            return None

        values = getattr(next_state, "_values", {})
        literals = set()
        for l, v in values.items():
            l_name = l.fluent().name
            l_objs = [str(o) for o in l.args]
            l_formatted = f"({l_name})" if len(l_objs) == 0 else f"({l_name} {' '.join(l_objs)})"
            literals.add(l_formatted if v.is_true() else f"(not {l_formatted})")
        return literals

    @cached_property
    def ground_actions(self) -> Dict[str, Any]:
        return self._grounder.ground_actions()

    def applicable_actions(self, state) -> Dict[str, Set[str]]:
        if isinstance(state, Set) or isinstance(state, List):
            state = self._state_from_literals(state)

        applicable_actions = defaultdict(set)
        for op_name, param_combos in self.ground_actions.items():
            for args in param_combos:
                if self._simulator._is_applicable(
                    state,
                    self.problem.action(op_name),
                    [self.problem.object(o.lower()) for o in args],
                ):
                    action_label = f"({op_name})" if len(args) == 0 else f"({op_name} {' '.join(args)})"
                    applicable_actions[op_name].add(action_label)

        return applicable_actions
