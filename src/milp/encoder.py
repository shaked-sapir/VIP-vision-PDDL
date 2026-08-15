"""CP-SAT encoding of the MILP "solution fixer" with observed actions.

Shared by both MILP callers — the ``pisam_milp_*`` learners (MILP as denoiser)
and the ``rosame_milp*`` baselines (MILP as regularizer of a neural learner).
They differ only in the rule set (:class:`encoding_config.MilpEncodingConfig`
preset) and in what they do with the solution, never in the machinery here.

Adapted from the vendored ``constraint_opt/cp_sat.py`` (see vendor/UPSTREAM.md).
Differences from upstream, all motivated by our setting (ground actions observed,
trajectories from different problems, hard-fixed ground-truth states):

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
4. **Configurable state anchoring** — upstream hard-fixes exactly the initial
   and goal states. Here the set of hard-fixed states is per trace: a trace with
   ``goal is None`` gets no hard final state (it participates via the soft
   ``obs_p`` objective instead), and a trace may pin arbitrary known-GT
   intermediate states through its ``hard_states`` attribute. See
   :meth:`CPSATObservedActions.hard_states`; paper eqs. 19–22 are the two
   special cases.
5. **Config-driven rule set** — every constraint family that can exclude a legal
   ground-truth model (the per-schema non-empty rule, the redundant-add ban, and
   paper eq. 18 ``del => pre``) plus the optional objective terms (eq. 16
   precondition bias, reference-model channel weighting) are bundled in a
   :class:`encoding_config.MilpEncodingConfig`. Presets: ``upstream()`` =
   released behavior, ``tag()`` = the ``rosame_milp_tag`` variant,
   ``cdps_dialect()`` = all GT-excluding families dropped (used by
   ``pisam_milp_*``); see vendor/UPSTREAM.md.

Expected trace objects: vendored ``ObservationT`` instances with extra
attributes attached by our converter:
  - ``instance``    — the trace's own planning_structs ``Instance`` (required);
  - ``actions``     — dict ``t -> Action`` (the observed grounded action at step
    t, 1-based, t in 1..step) (required);
  - ``hard_states`` — optional dict ``t -> set of true Propositions`` naming
    extra states to hard-fix (see difference 4).

``traces.obs_m`` (the reference model) is optional: when absent, or when
``prior_weighting`` is ``NONE``, the model objective channel and the model-side
solution hints are simply omitted.
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

from src.milp.encoding_config import (
    MilpEncodingConfig,
    PriorWeightMode,
    SchemaNonemptyRule,
)

# CP-SAT solver determinism. The encoding usually has many equally-optimal
# solutions (the objective counts repairs, and different repair sets often cost
# the same), so which optimum comes back is a real choice, not a formality.
# Leaving it to the solver made repeated runs of the same fold disagree.
DEFAULT_RANDOM_SEED = 42
DEFAULT_NUM_WORKERS = 1


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
        # Optional warm-start override; see make_solution_hints. Aligned 1:1 with
        # traces.obs_t. Affects search order only, never the objective.
        self.state_hint_traces: Optional[list] = None

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
                    # paper eq. 18: only preconditions may be deleted. Legal PDDL
                    # allows deleting a non-precondition, so the CDPS dialect
                    # switches this off (see encoding_config).
                    if self.config.delete_implies_precondition:
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

    def hard_states(self, i: int) -> Dict[int, set]:
        """Time index (1-based, in ``1..steps+1``) -> set of true propositions to
        hard-fix for trace ``i``. Everything else at that index is forced false
        (closed-world), so a hard state is a fully known, unrepairable state.

        Always includes the initial state (GT by assumption; paper eqs. 19-20)
        and, when the trace supplies a goal, the final state (eqs. 21-22). A
        trace may additionally carry a ``hard_states`` dict — that is how our
        ``gt_anchoring: all_gt_states`` mode pins the intermediate GT states —
        which is merged on top.
        """
        obs = self._trace(i)
        steps = self._steps(i)
        fixed: Dict[int, set] = {1: set(obs.init)}
        if obs.goal is not None:
            fixed[steps + 1] = set(obs.goal)
        fixed.update(getattr(obs, "hard_states", None) or {})

        for t in fixed:
            if not 1 <= t <= steps + 1:
                raise ValueError(
                    f"hard state index {t} out of range for trace {i} "
                    f"(expected 1..{steps + 1})"
                )
        return fixed

    def build_trace_constraints(self) -> None:
        cons = []

        self.lodge_time("Building Hard-State Constraints")
        for i in range(1, self.n_traces + 1):
            inst = self._instance(i)
            for t, true_props in self.hard_states(i).items():
                for p in inst.propositions:
                    cons.append(self.hol[(i, t, p)] if p in true_props
                                else ~self.hol[(i, t, p)])
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

    def _w(self, prob: float, scale: float) -> int:
        """``prob`` in [0,1] -> integer weight; ``prob == 0.5`` (masked/unknown)
        weighs exactly 0, so uncertain slots are free."""
        return int(round((2.0 * float(prob) - 1.0) * scale))

    def _prior_bit_scale(self) -> float:
        """Per-model-bit scale of the reference-model objective channel.

        ``ROSAME``: one observed-fluent unit per bit (upstream behavior).
        ``TIEBREAK``: the whole channel is capped at ``tiebreak_mass``
        observed-fluent units, so its total swing (``2 * mass < 2``) stays below
        one fluent flip's swing (``2``) — the prior can only break ties.
        """
        if self.config.prior_weighting is PriorWeightMode.ROSAME:
            return float(self.obj_scale)
        n_bits = 3 * len(self.pre)
        if n_bits == 0:
            return 0.0
        return self.config.tiebreak_mass * self.obj_scale / n_bits

    def _model_prior_terms(self) -> list:
        """Objective terms pricing agreement with ``traces.obs_m`` (may be absent)."""
        obs_m = getattr(self.traces, "obs_m", None)
        if obs_m is None or self.config.prior_weighting is PriorWeightMode.NONE:
            return []

        scale = self._prior_bit_scale()
        self.solve_stats["prior_bit_scale"] = scale
        terms = []
        for a in self.domain.action_schemas:
            for p in self.domain.predicates:
                for x in self.domain.predicate_arguments[(a, p)]:
                    for probs, variables in (
                        (obs_m.pre, self.pre),
                        (obs_m.add, self.add),
                        (obs_m.dele, self.dele),
                    ):
                        coef = self._w(probs[a, p, x], scale)
                        if coef:
                            terms.append(coef * variables[(a, p, x)])
        return terms

    def build_objectives(self) -> None:
        self.lodge_time("Building Objectives")

        terms = []

        if "state" in self.objectives:
            for i in range(1, self.n_traces + 1):
                obs = self._trace(i)
                for t in range(1, self._steps(i) + 2):
                    for op in obs.obs_p.get(t, []):
                        coef = self._w(op.prob, self.obj_scale)
                        if coef:
                            terms.append(coef * self.hol[(i, t, op.proposition)])

        if "model" in self.objectives:
            terms.extend(self._model_prior_terms())

        if self.config.eq16:
            # ICAPS-26 eq. 16: unconditional bonus for declaring a precondition,
            # defeating the "steppre = 0" escape. Changes T' as well as the witness.
            coef = int(round(self.config.lambda_pre * self.obj_scale))
            if coef:
                terms.extend(coef * var for var in self.pre.values())

        if terms:
            self.model.maximize(cp.sum(terms))
        self.report_time()

    # ---------- solving ----------

    def make_solution_hints(self, threshold: float = 0.5):
        """Warm start: the observations themselves, plus the prior if there is one.

        ``state_hint_traces`` (set by the loop's ``frozen_with_hints`` policy) lets
        a caller hint the *state* variables from somewhere other than the encoded
        traces — e.g. a previous round's repair of the same traces. It is a hint
        only: the objective still prices flips against the encoded observations,
        so the optimum is unchanged and only the time to reach it can differ.
        """
        hint_vars, hint_vals = [], []

        hint_source = self.state_hint_traces or self.traces.obs_t
        for i in range(1, self.n_traces + 1):
            obs = hint_source[i - 1]
            for t in range(1, self._steps(i) + 2):
                for op in obs.obs_p.get(t, []):
                    variable = self.hol.get((i, t, op.proposition))
                    if variable is None:
                        continue  # a hint source may name a fluent this trace lacks
                    hint_vars.append(variable)
                    hint_vals.append(1 if op.prob > threshold else 0)

        # No reference model (e.g. pisam_milp single round) => hint the states only.
        obs_m = getattr(self.traces, "obs_m", None)
        if obs_m is None:
            return hint_vars, hint_vals

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

    def solve(
        self,
        time_limit: Optional[int] = None,
        log_search_progress: bool = False,
        random_seed: int = DEFAULT_RANDOM_SEED,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> bool:
        """Solve the encoding, returning True when a usable solution was found.

        Args:
            time_limit: wall-clock cap in seconds; None means no cap.
            log_search_progress: forward CP-SAT's search log to stdout.
            random_seed: CP-SAT's ``random_seed`` parameter. Pinned so that two
                solves of the same encoding break ties among equally-optimal
                solutions the same way.
            num_workers: CP-SAT's ``num_search_workers``. Defaults to 1 because
                parallel workers race, and whichever one finishes first decides
                which optimum is returned — seeding alone does not remove that.

        Reproducibility caveat: pinning both makes a solve that *proves optimality*
        reproducible. A solve that exhausts ``time_limit`` is still not, since the
        cap is wall-clock rather than CP-SAT deterministic time; the answer then
        depends on how much search the machine happened to fit in the budget.
        """
        start = time.time()
        solver = SolverLookup.get("ortools", self.model)
        hint_vars, hint_vals = self.make_solution_hints()
        solver.solution_hint(hint_vars, hint_vals)
        solver.solve(
            time_limit=time_limit,
            log_search_progress=log_search_progress,
            random_seed=random_seed,
            num_search_workers=num_workers,
        )

        st = solver.status()
        # update(), not reassign: build-time facts (e.g. prior_bit_scale) live here too
        self.solve_stats.update({
            "exit_status": str(st.exitstatus),
            "solve_time_seconds": round(time.time() - start, 3),
            "objective_value": solver.objective_value(),
            "n_model_vars": 3 * len(self.pre),
            "n_trace_vars": len(self.hol) + 3 * len(self.stepadd),
            "n_traces": self.n_traces,
            "random_seed": random_seed,
            "num_workers": num_workers,
            **self.config.as_stats(),
        })

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
