"""ROSAME-I+MILP baseline runner (imaged mode only).

The ICAPS-26 competitor in its native setting: the CV state predictor, the
differentiable ROSAME action model and the MILP are trained jointly from raw
state images. Inputs are :class:`RosameIBaselineRunner`'s — images, observed
actions and the GT final state, never the degraded trajectory states — and the
loop is :class:`MilpRosameI`'s two pseudo-label channels.

Each MILP round turns the CV head's current per-frame predictions into vendored
traces (:func:`cv_predictions_to_trace`), solves once, and returns both the
4-way action-model labels and the repaired interior states as CV targets.

See ``docs/rosame-i-milp-implementation-plan.md``. On simulation-mode cells (no
images on disk) the runner skips exactly as ``rosame_i`` does.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Problem

import benchmark.algorithm_adapters.rosame_milp  # noqa: F401  (vendor sys.path bootstrap)
from src.milp import encoder as _encoder_module  # noqa: F401  (factory registration)
from src.milp.converter import (
    build_ps_domain,
    build_ps_instance,
    cv_predictions_to_trace,
)
from src.milp.encoding_config import MilpEncodingConfig
from benchmark.baselines.rosame_i_runner import (
    _AUGMENT_DOMAINS,
    _DEFAULT_HYPERPARAMS,
    _HYPERPARAMS,
    _RESIZE_FROM_TABLE,
    ResizeSpec,
    RosameIBaselineRunner,
)
from benchmark.baselines.rosame_milp_runner import goal_fluents_from_trajectory

from constraint_opt.factory import resolve as resolve_encoder
from planning_structs.traces import Traces

_OBJECTIVES = {"state", "model"}

Fluents = Set[Tuple[str, Tuple[str, ...]]]


class _TraceContext:
    """Everything a trace needs to become a MILP trace, keyed by its name."""

    def __init__(self, instance, calls, init_fluents: Fluents,
                 goal_fluents: Optional[Fluents]) -> None:
        self.instance = instance
        self.calls = calls
        self.init_fluents = init_fluents
        self.goal_fluents = goal_fluents


class RosameIMilpRunner(RosameIBaselineRunner):
    """ROSAME-I trained with MILP pseudo-labels on the model and state channels."""

    _base_name: str = "ROSAME-I_MILP"

    def __init__(
        self,
        n_seeds: int = 1,
        device: Optional[str] = None,
        base_seed: int = 8800,
        psi: float = 0.99,
        pre_mip_epochs: int = 50,
        mip_interval: int = 1,
        mip_traces: int = 3,
        mip_time_limit: float = 60.0,
        agreement_stop: float = 1.0,
        encoding_config: Optional[MilpEncodingConfig] = None,
        goal_mode: str = "gt",
        milp_solver: str = "cp-sat-observed",
        resize: ResizeSpec = _RESIZE_FROM_TABLE,
    ) -> None:
        super().__init__(
            train_per_trajectory=False, n_seeds=n_seeds, device=device,
            base_seed=base_seed, resize=resize,
        )
        self.psi = psi
        self.pre_mip_epochs = pre_mip_epochs
        self.mip_interval = mip_interval
        self.mip_traces = mip_traces
        self.mip_time_limit = mip_time_limit
        self.agreement_stop = agreement_stop
        self.encoding_config = encoding_config or MilpEncodingConfig.upstream()
        self.goal_mode = goal_mode
        self.milp_solver = milp_solver

    @property
    def display_name(self) -> str:
        return "ROSAME-I+MILP"

    @property
    def color(self) -> str:
        return "#7c4bc0"

    # ------------------------------------------------------------------ inputs

    @staticmethod
    def _init_fluents(problem: Problem) -> Fluents:
        """Positive fluents of a problem's ``:init``, as ``(name, args)`` pairs."""
        fluents: Fluents = set()
        for grounded in problem.initial_state_predicates.values():
            for gp in grounded:
                if not getattr(gp, "is_positive", True):
                    continue
                args = tuple(gp.object_mapping[k] for k in gp.signature.keys())
                fluents.add((gp.name, args))
        return fluents

    def _build_contexts(
        self,
        ps_domain,
        partial_domain: Domain,
        prepared_problems: Sequence[Tuple[object, Sequence[Path], Sequence[str], Sequence[str]]],
        gt_paths: Sequence[Path],
    ) -> Tuple[Dict[str, _TraceContext], int]:
        """``trace name -> _TraceContext`` plus the count with a GT-anchored goal.

        Keys are the names :meth:`RosameI_Runner.prepare_traces` assigns.
        """
        contexts: Dict[str, _TraceContext] = {}
        n_gt_goals = 0
        for idx, (problem, image_paths, action_strings, final_preds) in enumerate(prepared_problems):
            name = image_paths[0].parent.name if image_paths else f"problem{idx}"
            calls = []
            for action_str in action_strings:
                parts = action_str.split()
                calls.append((parts[0], parts[1:]))

            # ``final_preds`` is the GT final state already resolved by
            # RosameIBaselineRunner._resolve_final_state — parsed, and conformed to
            # the reference domain's dialect. Re-reading the GT trajectory here would
            # be a *second* lookup with its own dialect policy (the hazard called out
            # in docs/rosame-i-milp-implementation-plan.md §7.7): the raw-text reader
            # cannot see whether the domain is hyphenated, so on an un-normalized
            # hyphenated domain none of its fluents ground, the goal resolves to the
            # empty set, and the encoder's closed-world hard state then forces every
            # proposition false at step+1 — infeasible on every round.
            goal = None
            if self.goal_mode == "gt" and final_preds:
                goal = {(p[0], tuple(p[1:])) for p in (s.split() for s in final_preds)}
            if goal is not None:
                n_gt_goals += 1

            contexts[name] = _TraceContext(
                instance=build_ps_instance(ps_domain, partial_domain, problem),
                calls=calls,
                init_fluents=self._init_fluents(problem),
                goal_fluents=goal,
            )
        return contexts, n_gt_goals

    # ------------------------------------------------------------------ round

    def _make_milp_round(self, rosame, ps_domain, contexts: Dict[str, _TraceContext]):
        """A ``MilpRoundFn`` closing over the trained model and the trace contexts."""
        from benchmark.algorithm_adapters.rosame_milp.model_bridge import (
            extract_model_labels,
            extract_state_labels,
            model_agreement,
            rosame_to_observation_m,
        )

        def milp_round(selected, time_limit: float):
            names = rosame.proposition_names
            obs_t = []
            solved: List[Tuple[str, _TraceContext]] = []
            for trace in selected:
                context = contexts.get(trace.name)
                if context is None:
                    continue
                built = cv_predictions_to_trace(
                    context.instance, names, rosame.predict_frames(trace),
                    context.calls, context.init_fluents, context.goal_fluents,
                )
                if built is None:
                    continue
                obs_t.append(built)
                solved.append((trace.name, context))

            if not obs_t:
                return {}, {}, 0.0, {"exit_status": "NO_TRACES"}, None

            traces = Traces(
                instance=None,
                obs_m=rosame_to_observation_m(rosame, ps_domain),
                obs_t=obs_t,
            )
            encoder = resolve_encoder(self.milp_solver)(
                ps_domain, traces, _OBJECTIVES, config=self.encoding_config
            )
            if not encoder.solve(time_limit=time_limit):
                return {}, {}, 0.0, encoder.solve_stats, None

            solution = encoder.action_model_sol()
            model_labels = extract_model_labels(rosame, ps_domain, solution)
            state_labels = {
                name: extract_state_labels(
                    context.instance, encoder.repaired_states(i + 1), names
                )
                for i, (name, context) in enumerate(solved)  # traces are 1-indexed
            }
            agreement = model_agreement(rosame, model_labels)
            return model_labels, state_labels, agreement, encoder.solve_stats, solution

        return milp_round

    # ------------------------------------------------------------------ learn

    def learn(
        self,
        domain_path: Path,
        prepared_trajectories: List[Tuple[Path, Path, Path]],
        work_dir: Path,
        timeout_seconds: int = 60,
    ) -> Tuple[Optional[str], Dict]:
        bench, partial_domain = self._bench_and_domain(domain_path)
        hp = _HYPERPARAMS.get(bench, _DEFAULT_HYPERPARAMS)
        augment = bench in _AUGMENT_DOMAINS
        resize = self._resolve_resize(bench)

        prepared_problems, gt_paths = self._resolve_inputs(
            partial_domain, prepared_trajectories, bench
        )
        if not prepared_problems:
            print(f"  [{self.name}] skipping: no images (simulation-mode cell?)")
            return None, {}

        extra: Dict = {
            "goal_mode": self.goal_mode,
            "milp_solver": self.milp_solver,
            "encoding_config": self.encoding_config.as_stats(),
            "psi": self.psi,
            "pre_mip_epochs": self.pre_mip_epochs,
            "mip_interval": self.mip_interval,
            "mip_traces": self.mip_traces,
            "n_seeds": self.n_seeds,
            "resize": resize,
        }
        try:
            ps_domain = build_ps_domain(partial_domain)
            contexts, n_gt_goals = self._build_contexts(
                ps_domain, partial_domain, prepared_problems, gt_paths
            )
            extra["n_problems"] = len(contexts)
            extra["n_gt_goals"] = n_gt_goals
        except Exception as e:
            print(f"  Warning: {self.name} MILP setup failed: {e}")
            return None, extra

        model, run_extra = self._train_seeds(
            domain_path, work_dir, timeout_seconds, prepared_problems,
            ps_domain, contexts, hp, augment, resize,
        )
        extra.update(run_extra)
        return model, extra

    def _train_seeds(
        self,
        domain_path: Path,
        work_dir: Path,
        timeout_seconds: int,
        prepared_problems,
        ps_domain,
        contexts: Dict[str, _TraceContext],
        hp: Dict[str, float],
        augment: bool,
        resize: ResizeSpec,
    ) -> Tuple[Optional[str], Dict]:
        """Train each seed's loop, keep the lowest-final-loss model."""
        from benchmark.algorithm_adapters.rosame_milp.milp_loop_i import MilpRosameI

        start = time.perf_counter()

        def timeout_check() -> bool:
            return (time.perf_counter() - start) > timeout_seconds

        def seconds_left() -> float:
            return max(0.0, timeout_seconds - (time.perf_counter() - start))

        seed_losses: Dict[int, float] = {}
        seed_models: Dict[int, str] = {}
        seed_reports: Dict[int, Dict] = {}

        for i in range(self.n_seeds):
            seed = self.base_seed + i
            if seed_models and timeout_check():
                continue
            try:
                rosame = MilpRosameI(
                    str(domain_path), device=self.device, seed=seed, psi=self.psi,
                    resize=resize,
                )
                report = rosame.learn_pooled_with_milp(
                    prepared_problems,
                    self._make_milp_round(rosame, ps_domain, contexts),
                    epochs=int(hp["epochs"]),
                    gamma=float(hp["gamma"]),
                    lambda_=float(hp["lambda_"]),
                    augment=augment,
                    pre_mip_epochs=self.pre_mip_epochs,
                    mip_interval=self.mip_interval,
                    mip_traces=self.mip_traces,
                    mip_time_limit=self.mip_time_limit,
                    agreement_stop=self.agreement_stop,
                    timeout_check=timeout_check,
                    seconds_left=seconds_left,
                )
            except Exception as e:  # keep one bad seed from killing the cell
                print(f"  [{self.name}] seed {seed} failed: {e}")
                continue

            model, failed = self._model_from(rosame, ps_domain, report)
            if not model or ":action" not in model:
                print(f"  [{self.name}] seed {seed}: invalid model, skipping")
                continue

            report["milp_failed"] = failed
            seed_losses[seed] = report["final_loss"]
            seed_models[seed] = model
            seed_reports[seed] = report
            seed_dir = Path(work_dir) / "baseline_models" / self.name / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "model.pddl").write_text(model)

        if not seed_models:
            print(f"  [{self.name}] no seed produced a valid model")
            return None, {"loop_seconds": round(time.perf_counter() - start, 2)}

        chosen = min(seed_losses, key=seed_losses.get)
        report = seed_reports[chosen]
        return seed_models[chosen], {
            "loop_seconds": round(time.perf_counter() - start, 2),
            "seeds": seed_losses,
            "chosen_seed": chosen,
            "chosen_final_loss": seed_losses[chosen],
            "n_traces": report["n_traces"],
            "milp_rounds": report["rounds"],
            "stop_reason": report["stop_reason"],
            "final_agreement": report["final_agreement"],
            "milp_failed": report["milp_failed"],
            "mip_interval_used": report["mip_interval_used"],
        }

    def _model_from(self, rosame, ps_domain, report: Dict) -> Tuple[str, bool]:
        """``(PDDL string, milp_failed)`` — the MILP's model, or plain ROSAME-I."""
        from benchmark.algorithm_adapters.rosame_milp.model_bridge import solution_to_pddl

        solution = report["final_solution"]
        if solution is not None:
            return solution_to_pddl(rosame, ps_domain, solution), False
        print(f"  [{self.name}] MILP produced no solution — falling back to ROSAME-I")
        return rosame.to_pddl(), True
