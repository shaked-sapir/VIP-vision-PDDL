"""CP-SAT encoder variant of the ROSAME+MILP "solution fixer" with observed actions.

Adapted from the vendored ``constraint_opt/cp_sat.py`` (see vendor/UPSTREAM.md).
Differences from upstream, all motivated by our setting (ground actions observed,
trajectories from different problems, s0 + optionally GT-final anchoring):

1. **Observed actions** — no ``act`` variables at all. Each trace supplies its
   executed action per step; the step-level indicators are tied directly to the
   lifted model variables of that action's bindings. This removes the paper's
   eqs. 12, 23 and the act-side of eqs. 24–32.
2. **Per-trace instances** — each ObservationT carries its own grounded
   ``instance`` (our folds mix problems with different objects). The lifted
   pre/add/del variables remain shared across traces — the point of the joint
   solve. (Upstream assumes one shared ``traces.instance``.)
3. **Per-trace lengths** — upstream applies trace 0's length to all traces
   (``max_t = obs_t[0].step + 1``); here every trace uses its own ``step``.
4. **Optional goal anchoring** — a trace with ``goal is None`` gets no hard
   final-state constraints (its final state participates via the soft ``obs_p``
   objective instead). Traces with a goal keep upstream's hard fixing
   (paper eqs. 21–22).
5. **Config-driven undocumented constraints** — the per-schema non-empty rule
   and the redundant-add ban reproduce two constraint families present in the
   released code but absent from the paper. They are bundled in a
   :class:`encoding_config.MilpEncodingConfig` (preset ``upstream()`` = released
   behavior, ``tag()`` = the ``rosame_milp_tag`` variant); see vendor/UPSTREAM.md.

Expected trace objects: vendored ``ObservationT`` instances with two extra
attributes attached by our converter:
  - ``instance`` — the trace's own planning_structs ``Instance``;
  - ``actions``  — dict ``t -> Action`` (the observed grounded action at step t,
    1-based, t in 1..step).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import cpmpy as cp
from cpmpy import Model, boolvar, SolverLookup
from cpmpy.solvers.solver_interface import ExitStatus

from constraint_opt.factory import register_problem
from constraint_opt.util import args2str, unifies
from planning_structs.domain import ActionSchema, Predicate
from planning_structs.instance import Action, Proposition
from planning_structs.traces import ObservationM

from benchmark.algorithm_adapters.rosame_milp.encoding_config import (
    MilpEncodingConfig,
    SchemaNonemptyRule,
)


class CPSATObservedActions:
    """CP-SAT encoding of the MILP fixer with actions fixed to observations."""

    def __init__(
        self,
        domain,
        traces,
        objectives,
        config: Optional[MilpEncodingConfig] = None,
        enforce_nonempty_schemas: Optional[bool] = None,
        forbid_redundant_adds: Optional[bool] = None,
        obj_scale: int = 10**6,
        verbose: bool = False,
    ):
        self.domain = domain
        self.traces = traces
        self.n_traces = len(traces.obs_t)
        self.objectives = objectives
        # Prefer the config object; fall back to the legacy boolean kwargs so
        # older callers/tests keep working (None => upstream defaults).
        self.config = config if config is not None else self._config_from_legacy(
            enforce_nonempty_schemas, forbid_redundant_adds
        )
        self.obj_scale = obj_scale
        self.verbose = verbose

        self.model = Model()

        self.pre: Dict[Tuple[ActionSchema, Predicate, Tuple[int, ...]], Any] = {}
        self.add: Dict[Tuple[ActionSchema, Predicate, Tuple[int, ...]], Any] = {}
        self.dele: Dict[Tuple[ActionSchema, Predicate, Tuple[int, ...]], Any] = {}

        self.hol: Dict[Tuple[int, int, Proposition], Any] = {}
        self.stepadd: Dict[Tuple[int, int, Proposition], Any] = {}
        self.stepdel: Dict[Tuple[int, int, Proposition], Any] = {}
        self.steppre: Dict[Tuple[int, int, Proposition], Any] = {}

        self.solve_stats: Dict[str, Any] = {}

        self.build_domain_variables()
        self.build_domain_constraints()
        self.build_trace_variables()
        self.build_trace_constraints()
        self.build_objectives()

    # ---------- helpers ----------

    @staticmethod
    def _config_from_legacy(
        enforce_nonempty_schemas: Optional[bool],
        forbid_redundant_adds: Optional[bool],
    ) -> MilpEncodingConfig:
        """Build a config from the legacy boolean kwargs (None => upstream)."""
        rule = (
            SchemaNonemptyRule.NONE
            if enforce_nonempty_schemas is False
            else SchemaNonemptyRule.PRE_AND_ADD
        )
        return MilpEncodingConfig(
            schema_nonempty=rule,
            forbid_redundant_adds=(
                True if forbid_redundant_adds is None else forbid_redundant_adds
            ),
        )

    def _trace(self, i: int):
        return self.traces.obs_t[i - 1]

    def _steps(self, i: int) -> int:
        return self._trace(i).step

    def _instance(self, i: int):
        return self._trace(i).instance

    def _observed_action(self, i: int, t: int) -> Action:
        return self._trace(i).actions[t]

    def lodge_time(self, mess):
        self.prev_time = time.time()
        if self.verbose:
            print(mess + "...", end="")

    def report_time(self):
        if self.verbose:
            print(round(time.time() - self.prev_time, 4))

    # ---------- variables ----------

    def build_domain_variables(self) -> None:
        self.lodge_time("Building Domain Variables")
        for a in self.domain.action_schemas:
            for p in self.domain.predicates:
                for x in self.domain.predicate_arguments[(a, p)]:
                    self.pre[(a, p, x)] = boolvar(name=f"pre_{a.name}_{p.name}_{args2str(x)}")
                    self.add[(a, p, x)] = boolvar(name=f"pos_{a.name}_{p.name}_{args2str(x)}")
                    self.dele[(a, p, x)] = boolvar(name=f"neg_{a.name}_{p.name}_{args2str(x)}")
        self.report_time()

    def build_trace_variables(self) -> None:
        self.lodge_time("Building Hol/Step Variables")
        for i in range(1, self.n_traces + 1):
            inst = self._instance(i)
            steps = self._steps(i)
            for p in inst.propositions:
                for t in range(1, steps + 2):
                    self.hol[(i, t, p)] = boolvar(name=f"hol_{i}_{t}_{p}")
                for t in range(1, steps + 1):
                    self.stepadd[(i, t, p)] = boolvar(name=f"stepadd_{i}_{t}_{p}")
                    self.stepdel[(i, t, p)] = boolvar(name=f"stepdel_{i}_{t}_{p}")
                    self.steppre[(i, t, p)] = boolvar(name=f"steppre_{i}_{t}_{p}")
        self.report_time()

    # ---------- constraints ----------

    def build_domain_constraints(self) -> None:
        self.lodge_time("Building Domain Constraints")
        cons = []
        for a in self.domain.action_schemas:
            epre_terms = []
            eadd_terms = []
            for p in self.domain.predicates:
                for x in self.domain.predicate_arguments[(a, p)]:
                    # paper eq. 18: only preconditions may be deleted
                    cons.append(self.dele[(a, p, x)].implies(self.pre[(a, p, x)]))
                    # paper eq. 17: preconditions and add effects don't intersect
                    cons.append(~(self.pre[(a, p, x)] & self.add[(a, p, x)]))
                    epre_terms.append(self.pre[(a, p, x)])
                    eadd_terms.append(self.add[(a, p, x)])
            # NOT in the paper — upstream code extra (see vendor/UPSTREAM.md).
            # Rule selected via MilpEncodingConfig.schema_nonempty.
            if epre_terms:
                rule = self.config.schema_nonempty
                if rule is SchemaNonemptyRule.PRE_AND_ADD:
                    cons.append(cp.sum(epre_terms) >= 1)
                    cons.append(cp.sum(eadd_terms) >= 1)
                elif rule is SchemaNonemptyRule.ADD:
                    cons.append(cp.sum(eadd_terms) >= 1)
        self.model += cons
        self.report_time()

    def _bindings(self, action: Action, p: Proposition):
        """All lifted bindings of ``p.predicate`` to ``action`` consistent with the args."""
        schema = action.action_schema
        return [
            x
            for x in self.domain.predicate_arguments.get((schema, p.predicate), [])
            if unifies(x, p.args, action.args)
        ]

    def build_trace_constraints(self) -> None:
        cons = []

        self.lodge_time("Building Initial/Goal State Constraints")
        for i in range(1, self.n_traces + 1):
            obs = self._trace(i)
            inst = self._instance(i)
            steps = self._steps(i)
            # initial state at t=1 — hard (paper eqs. 19–20; s0 is GT by assumption)
            init_set = set(obs.init)
            for p in inst.propositions:
                if p in init_set:
                    cons.append(self.hol[(i, 1, p)])
                else:
                    cons.append(~self.hol[(i, 1, p)])
            # final state at t=steps+1 — hard only when a (GT) goal is supplied
            # (paper eqs. 21–22; ``goal is None`` leaves it to the soft objective)
            if obs.goal is not None:
                goal_set = set(obs.goal)
                for p in inst.propositions:
                    if p in goal_set:
                        cons.append(self.hol[(i, steps + 1, p)])
                    else:
                        cons.append(~self.hol[(i, steps + 1, p)])
        self.report_time()

        # Step indicators tied directly to the observed action's bindings
        # (replaces the act-coupled paper eqs. 24–32).
        self.lodge_time("Building Step Pre/Add/Del Constraints")
        for i in range(1, self.n_traces + 1):
            inst = self._instance(i)
            steps = self._steps(i)
            for t in range(1, steps + 1):
                action = self._observed_action(i, t)
                for p in inst.propositions:
                    bindings = self._bindings(action, p)
                    if bindings:
                        cons.append(self.stepadd[(i, t, p)] == cp.any(
                            [self.add[(action.action_schema, p.predicate, x)] for x in bindings]))
                        cons.append(self.stepdel[(i, t, p)] == cp.any(
                            [self.dele[(action.action_schema, p.predicate, x)] for x in bindings]))
                        cons.append(self.steppre[(i, t, p)] == cp.any(
                            [self.pre[(action.action_schema, p.predicate, x)] for x in bindings]))
                    else:
                        cons.append(~self.stepadd[(i, t, p)])
                        cons.append(~self.stepdel[(i, t, p)])
                        cons.append(~self.steppre[(i, t, p)])
                    if self.config.forbid_redundant_adds:
                        # NOT in the paper — upstream code extra; can make the GT
                        # model infeasible under legal redundant adds
                        # (see vendor/UPSTREAM.md)
                        cons.append(~(self.stepadd[(i, t, p)] & self.hol[(i, t, p)]))
        self.report_time()

        # Successor state + precondition constraints (paper eqs. 33–35)
        # and frame axioms (paper eqs. 36–37)
        self.lodge_time("Building Successor/Frame Constraints")
        for i in range(1, self.n_traces + 1):
            inst = self._instance(i)
            steps = self._steps(i)
            for p in inst.propositions:
                for t in range(1, steps + 1):
                    cons.append(self.stepadd[(i, t, p)].implies(self.hol[(i, t + 1, p)]))
                    cons.append(self.stepdel[(i, t, p)].implies(~self.hol[(i, t + 1, p)]))
                    cons.append(self.steppre[(i, t, p)].implies(self.hol[(i, t, p)]))
                    cons.append((~self.hol[(i, t, p)] & self.hol[(i, t + 1, p)]).implies(self.stepadd[(i, t, p)]))
                    cons.append((self.hol[(i, t, p)] & ~self.hol[(i, t + 1, p)]).implies(self.stepdel[(i, t, p)]))
        self.report_time()

        self.model += cons

    # ---------- objective ----------

    def build_objectives(self) -> None:
        self.lodge_time("Building Objectives")

        def w(prob: float) -> int:
            # prob in [0,1] -> integer weight; prob==0.5 (masked) weighs exactly 0
            return int(round((2.0 * float(prob) - 1.0) * self.obj_scale))

        terms = []

        if "state" in self.objectives:
            for i in range(1, self.n_traces + 1):
                obs = self._trace(i)
                for t in range(1, self._steps(i) + 2):
                    for op in obs.obs_p.get(t, []):
                        coef = w(op.prob)
                        if coef:
                            terms.append(coef * self.hol[(i, t, op.proposition)])

        if "model" in self.objectives:
            obs_m = self.traces.obs_m
            for a in self.domain.action_schemas:
                for p in self.domain.predicates:
                    for x in self.domain.predicate_arguments[(a, p)]:
                        c_pre = w(obs_m.pre[a, p, x])
                        if c_pre:
                            terms.append(c_pre * self.pre[(a, p, x)])
                        c_add = w(obs_m.add[a, p, x])
                        if c_add:
                            terms.append(c_add * self.add[(a, p, x)])
                        c_del = w(obs_m.dele[a, p, x])
                        if c_del:
                            terms.append(c_del * self.dele[(a, p, x)])

        if terms:
            self.model.maximize(cp.sum(terms))
        self.report_time()

    # ---------- solving ----------

    def make_solution_hints(self, threshold: float = 0.5):
        hint_vars, hint_vals = [], []

        for i in range(1, self.n_traces + 1):
            obs = self._trace(i)
            for t in range(1, self._steps(i) + 2):
                for op in obs.obs_p.get(t, []):
                    hint_vars.append(self.hol[(i, t, op.proposition)])
                    hint_vals.append(1 if op.prob > threshold else 0)

        obs_m = self.traces.obs_m
        for a in self.domain.action_schemas:
            for p in self.domain.predicates:
                for x in self.domain.predicate_arguments[(a, p)]:
                    hint_vars.append(self.pre[(a, p, x)])
                    hint_vals.append(1 if obs_m.pre[a, p, x] > threshold else 0)
                    hint_vars.append(self.add[(a, p, x)])
                    hint_vals.append(1 if obs_m.add[a, p, x] > threshold else 0)
                    hint_vars.append(self.dele[(a, p, x)])
                    hint_vals.append(1 if obs_m.dele[a, p, x] > threshold else 0)

        return hint_vars, hint_vals

    def solve(self, time_limit: Optional[int] = None, log_search_progress: bool = False) -> bool:
        start = time.time()
        solver = SolverLookup.get("ortools", self.model)
        hint_vars, hint_vals = self.make_solution_hints()
        solver.solution_hint(hint_vars, hint_vals)
        solver.solve(time_limit=time_limit, log_search_progress=log_search_progress)

        st = solver.status()
        self.solve_stats = {
            "exit_status": str(st.exitstatus),
            "solve_time_seconds": round(time.time() - start, 3),
            "objective_value": solver.objective_value(),
            "n_model_vars": 3 * len(self.pre),
            "n_trace_vars": len(self.hol) + 3 * len(self.stepadd),
            "n_traces": self.n_traces,
            **self.config.as_stats(),
        }

        if st.exitstatus == ExitStatus.UNKNOWN:
            print("  [MILP] No feasible solution found within time limit.")
            return False
        if st.exitstatus == ExitStatus.UNSATISFIABLE:
            print("  [MILP] Problem is infeasible.")
            return False
        return True

    # ---------- post-processing ----------

    def action_model_sol(self) -> ObservationM:
        pre = {(a, p, x): int(bool(self.pre[(a, p, x)].value())) for (a, p, x) in self.pre}
        add = {(a, p, x): int(bool(self.add[(a, p, x)].value())) for (a, p, x) in self.add}
        dele = {(a, p, x): int(bool(self.dele[(a, p, x)].value())) for (a, p, x) in self.dele}
        return ObservationM(pre, add, dele)

    def repaired_states(self, i: int):
        """The solved (repaired) state sequence of trace ``i`` — list of sets of Propositions."""
        inst = self._instance(i)
        return [
            {p for p in inst.propositions if bool(self.hol[(i, t, p)].value())}
            for t in range(1, self._steps(i) + 2)
        ]

    def _bool_value(self, var) -> bool:
        return bool(var.value())


def build_cp_sat_observed_encoding(domain, traces, objectives, **kwargs):
    """Factory entrypoint for the observed-actions CP-SAT fixer."""
    return CPSATObservedActions(domain=domain, traces=traces, objectives=objectives, **kwargs)


register_problem("cp-sat-observed", build_cp_sat_observed_encoding)
